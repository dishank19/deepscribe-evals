"""BERTScore similarity between AI and gold SOAP sections."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import torch
from bert_score import BERTScorer

from evalsuite.config import bertscore as bertscore_config
from evalsuite.metrics.registry import register

SECTION_KEYS = ("S", "O", "A", "P")


@lru_cache(maxsize=1)
def _scorer() -> BERTScorer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BERTScorer(
        lang=bertscore_config.lang,
        model_type=bertscore_config.model_type,
        rescale_with_baseline=bertscore_config.rescale_with_baseline,
        device=device,
    )


def _score_pair(candidate: str, reference: str) -> Dict[str, float]:
    candidate = (candidate or "").strip()
    reference = (reference or "").strip()
    if not candidate or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    scorer = _scorer()
    precision, recall, f1 = scorer.score([candidate], [reference])
    return {
        "precision": round(float(precision[0]), 4),
        "recall": round(float(recall[0]), 4),
        "f1": round(float(f1[0]), 4),
    }


def run_bertscore(
    transcript: str, ai_soap: Dict[str, str], gold_soap: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, List[str]]]:
    if not gold_soap:
        return {"scores": {}, "issues": {"warnings": ["No gold SOAP available."]}}

    per_section = {
        section: _score_pair(ai_soap.get(section, ""), gold_soap.get(section, ""))
        for section in SECTION_KEYS
    }

    # Average available sections; ignore those with no reference content
    available = [scores["f1"] for scores in per_section.values()]
    overall_f1 = round(sum(available) / len(available), 4) if available else 0.0

    scores: Dict[str, float] = {"overall_f1": overall_f1}
    for section, metrics in per_section.items():
        for metric_name, value in metrics.items():
            scores[f"{section}_{metric_name}"] = value

    return {"scores": scores, "issues": {}}


register("bertscore", run_bertscore)
