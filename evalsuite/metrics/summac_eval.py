"""SummaC-based factual consistency scoring."""

from __future__ import annotations

from typing import Dict, List

from evalsuite.config import summac as summac_config
from evalsuite.metrics.registry import register
from evalsuite.vendor.summac import get_summac_model

SECTION_KEYS = ("S", "O", "A", "P")


def _summac_model():
    return get_summac_model(
        model_name=summac_config.model_name,
        granularity=summac_config.granularity,
    )


def _score_section(model, transcript: str, section_text: str) -> float:
    if not section_text.strip():
        return 0.0
    result = model.score([transcript], [section_text])
    raw = float(result["scores"][0])
    normalized = max(0.0, min(1.0, (raw + 1.0) / 2.0))
    return normalized


def run_summac(transcript: str, ai_soap: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    model = _summac_model()
    section_scores = {
        section: _score_section(model, transcript, ai_soap.get(section, ""))
        for section in SECTION_KEYS
    }
    overall = sum(section_scores.values()) / len(SECTION_KEYS)

    scores = {"overall": round(overall, 3)}
    scores.update({f"{section}_consistency": round(value, 3) for section, value in section_scores.items()})

    threshold_flags: List[Dict[str, float]] = []
    if overall < summac_config.min_overall:
        threshold_flags.append(
            {
                "target": "overall",
                "score": round(overall, 3),
                "threshold": summac_config.min_overall,
            }
        )
    for section, value in section_scores.items():
        if value < summac_config.min_section:
            threshold_flags.append(
                {
                    "target": section,
                    "score": round(value, 3),
                    "threshold": summac_config.min_section,
                }
            )

    issues = {"thresholds": threshold_flags} if threshold_flags else {}
    return {"scores": scores, "issues": issues}


register("summac", run_summac)


def run_summac_against_reference(
    reference_soap: Dict[str, str],
    candidate_soap: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Compute SummaC scores using the gold SOAP as the source text."""

    model = _summac_model()
    section_scores = {
        section: _score_section(model, reference_soap.get(section, ""), candidate_soap.get(section, ""))
        for section in SECTION_KEYS
    }
    overall = sum(section_scores.values()) / len(SECTION_KEYS)

    scores = {"overall": round(overall, 3)}
    scores.update({f"{section}_consistency": round(value, 3) for section, value in section_scores.items()})

    return {"scores": scores}
