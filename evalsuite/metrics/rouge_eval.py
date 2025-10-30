"""ROUGE overlap between AI and gold SOAP sections."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from rouge_score import rouge_scorer

from evalsuite.config import rouge as rouge_config
from evalsuite.metrics.registry import register

SECTION_KEYS = ("S", "O", "A", "P")


@lru_cache(maxsize=1)
def _scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(
        rouge_config.metrics,
        use_stemmer=rouge_config.use_stemmer,
    )


def _score_pair(candidate: str, reference: str) -> Dict[str, float]:
    candidate = (candidate or "").strip()
    reference = (reference or "").strip()
    if not candidate or not reference:
        return {metric: 0.0 for metric in rouge_config.metrics}

    scorer = _scorer()
    results = scorer.score(reference, candidate)
    return {metric: round(results[metric].fmeasure, 4) for metric in rouge_config.metrics}


def run_rouge(
    transcript: str, ai_soap: Dict[str, str], gold_soap: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, List[str]]]:
    if not gold_soap:
        return {"scores": {}, "issues": {"warnings": ["No gold SOAP available."]}}

    per_section = {
        section: _score_pair(ai_soap.get(section, ""), gold_soap.get(section, ""))
        for section in SECTION_KEYS
    }

    scores: Dict[str, float] = {}
    for section, metrics in per_section.items():
        for metric, value in metrics.items():
            scores[f"{section}_{metric}"] = value

    # Compute macro-average across sections for the primary metric (first in list)
    primary_metric = rouge_config.metrics[0]
    available = [metrics[primary_metric] for metrics in per_section.values()]
    scores[f"overall_{primary_metric}"] = (
        round(sum(available) / len(available), 4) if available else 0.0
    )

    return {"scores": scores, "issues": {}}


register("rouge", run_rouge)
