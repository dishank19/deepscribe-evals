"""
Lightweight SummaC implementation for factual consistency scoring.

This module implements a trimmed-down version of the SummaC-ZS metric
described in:
    Kryscinski et al., "Evaluating the Factual Consistency of Summaries
    with Relation Extraction" (EMNLP 2021).

It mirrors the original library’s public interface closely enough for our
evaluation pipeline while avoiding the hard dependency pinning that makes the
upstream package incompatible with modern `datasets`/`huggingface-hub`.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import nltk

MODEL_MAP: Dict[str, Dict[str, object]] = {
    "mnli": {"model_card": "roberta-large-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "mnli-base": {"model_card": "microsoft/deberta-base-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "anli": {"model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", "entailment_idx": 0, "contradiction_idx": 2},
    "vitc": {"model_card": "tals/albert-xlarge-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc-base": {"model_card": "tals/albert-base-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
}


def _ensure_punkt() -> None:
    """Download the Punkt sentence tokenizer if it is not already available."""

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _neutral_index(ent_idx: int, cont_idx: int) -> int:
    """Return the third NLI index (neutral) given entailment and contradiction."""

    return ({0, 1, 2} - {ent_idx, cont_idx}).pop()


def _batch_iter(iterator: Sequence, batch_size: int) -> Iterator[Sequence]:
    """Yield successive batches from `iterator`."""

    for start in range(0, len(iterator), batch_size):
        yield iterator[start : start + batch_size]


class SummaCZS:
    """
    Zero-shot SummaC scorer.

    The implementation here mirrors the logic in the original repository:
    1. Split the source document and hypothesis summary into chunks (sentences
       by default).
    2. Score every (source_chunk, summary_chunk) pair with an NLI model.
    3. Aggregate entailment/contradiction probabilities to produce a final
       faithfulness score.
    """

    def __init__(
        self,
        model_name: str = "vitc",
        granularity: str = "sentence",
        max_doc_chunks: int = 80,
        batch_size: int = 16,
        device: str | None = None,
        use_entailment: bool = True,
        use_contradiction: bool = True,
    ) -> None:
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unsupported model '{model_name}'. Choices: {sorted(MODEL_MAP)}")
        if granularity not in {"sentence", "paragraph", "document"}:
            raise ValueError(f"Unsupported granularity '{granularity}'")

        _ensure_punkt()

        model_info = MODEL_MAP[model_name]
        self.model_card = str(model_info["model_card"])
        self.entailment_idx = int(model_info["entailment_idx"])
        self.contradiction_idx = int(model_info["contradiction_idx"])
        self.neutral_idx = _neutral_index(self.entailment_idx, self.contradiction_idx)

        self.granularity = granularity
        self.max_doc_chunks = max_doc_chunks
        self.batch_size = batch_size
        self.use_entailment = use_entailment
        self.use_contradiction = use_contradiction

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_card, use_fast=True)
        except (ValueError, OSError):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_card, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_card).to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()

    # ---------------------------------------------------------------------
    # Text preprocessing helpers

    def _split_document(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if self.granularity == "document":
            return [text]
        if self.granularity == "paragraph":
            paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
            return [p for p in paragraphs if len(p) > 10]
        # default sentence
        return [sent for sent in nltk.sent_tokenize(text) if len(sent.strip()) > 10]

    # ---------------------------------------------------------------------
    # Core scoring logic

    def _build_dataset(self, source: str, summary: str) -> Tuple[List[Tuple[str, str, int, int]], int, int]:
        source_chunks = self._split_document(source)[: self.max_doc_chunks]
        summary_chunks = self._split_document(summary)

        if not source_chunks or not summary_chunks:
            return [], len(source_chunks), len(summary_chunks)

        dataset: List[Tuple[str, str, int, int]] = []
        for doc_idx, source_chunk in enumerate(source_chunks):
            for summary_idx, summary_chunk in enumerate(summary_chunks):
                dataset.append((source_chunk, summary_chunk, doc_idx, summary_idx))
        return dataset, len(source_chunks), len(summary_chunks)

    def _score_pairs(self, pairs: Sequence[Tuple[str, str, int, int]], doc_len: int, sum_len: int) -> np.ndarray:
        image = np.zeros((3, max(doc_len, 1), max(sum_len, 1)), dtype=np.float32)
        if not pairs:
            return image

        for batch in _batch_iter(pairs, self.batch_size):
            premises = [prem for prem, _, _, _ in batch]
            hypotheses = [hypo for _, hypo, _, _ in batch]
            tokenized = self.tokenizer(
                list(zip(premises, hypotheses)),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            with torch.no_grad():
                outputs = self.model(**tokenized)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

            for (prem, hypo, doc_idx, sum_idx), prob in zip(batch, probs):
                image[0, doc_idx, sum_idx] = prob[self.entailment_idx]
                image[1, doc_idx, sum_idx] = prob[self.contradiction_idx]
                image[2, doc_idx, sum_idx] = prob[self.neutral_idx]

        return image

    def _aggregate(self, image: np.ndarray) -> float:
        # image shape: [3, doc_len, sum_len]
        ent_scores = image[0].max(axis=0)
        con_scores = image[1].max(axis=0)

        if self.use_entailment and self.use_contradiction:
            section_scores = ent_scores - con_scores
        elif self.use_entailment:
            section_scores = ent_scores
        else:
            section_scores = 1.0 - con_scores

        if not np.any(section_scores):
            return float(section_scores.mean())
        return float(section_scores.mean())

    # ---------------------------------------------------------------------
    # Public API

    def score(self, sources: Iterable[str], summaries: Iterable[str]) -> Dict[str, List[float]]:
        results: List[float] = []
        for source, summary in zip(sources, summaries):
            pairs, doc_len, sum_len = self._build_dataset(source, summary)
            image = self._score_pairs(pairs, doc_len, sum_len)
            score = self._aggregate(image)
            results.append(score)
        return {"scores": results}


@lru_cache(maxsize=1)
def get_summac_model(model_name: str = "vitc", granularity: str = "sentence") -> SummaCZS:
    """Return a singleton SummaCZS instance."""

    return SummaCZS(model_name=model_name, granularity=granularity)
