"""LLM-as-a-judge metric with optional gold-note coverage."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Dict, List, Optional

import anyio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, exceptions
from pydantic_ai.models import ModelSettings

from evalsuite.config import judge as judge_config
from evalsuite.metrics.registry import register

SECTION_KEYS = ("S", "O", "A", "P")
SECTION_NAMES = {"S": "Subjective", "O": "Objective", "A": "Assessment", "P": "Plan"}


class SectionScores(BaseModel):
    consistency: float
    completeness: float
    coherence: float
    fluency: float


class SectionIssues(BaseModel):
    missing: List[str] = Field(default_factory=list)
    unsupported: List[str] = Field(default_factory=list)
    clinical_errors: List[str] = Field(default_factory=list)


class SectionEvaluation(BaseModel):
    scores: SectionScores
    issues: SectionIssues
    summary: Optional[str] = None


class CoverageEvaluation(BaseModel):
    coverage_score: float
    missing_from_ai: List[str] = Field(default_factory=list)
    extraneous_in_ai: List[str] = Field(default_factory=list)
    summary: Optional[str] = None


SECTION_SYSTEM_PROMPT = """
You are a clinical QA evaluator. Judge one SOAP section against the source transcript.
- Always return the full SectionEvaluation schema with all keys present, even when values are empty.
- Populate `scores` (consistency, completeness, coherence, fluency) with floats in [0, 5]; use 0.0 when unsure.
- List up to 3 omissions in `issues.missing`, up to 2 hallucinations in `issues.unsupported`, and at most 1 safety-critical problem in `issues.clinical_errors`.
- Provide a concise one-sentence `summary` or null when nothing notable.
- If the transcript gives no evidence for the section, still respond with the schema using zeros and explain in the summary.
Answer strictly with the schema; never add commentary outside the JSON object.
""".strip()


COVERAGE_SYSTEM_PROMPT = """
Compare one AI SOAP section to the clinician reference.
- Always return the full CoverageEvaluation schema with all keys present.
- `coverage_score` must be a float in [0, 5]; use 0.0 when unsure.
- List up to 3 omissions in `missing_from_ai` and up to 3 extras in `extraneous_in_ai`.
- Provide a one-sentence `summary` or null.
- If the reference is empty, still respond with the schema (score 0.0 and empty lists).
Answer strictly with the schema; never add commentary outside the JSON object.
""".strip()


SECTION_TEMPLATE = """
Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END_TRANSCRIPT>>>

AI {section_name} section:
<<<AI_SECTION>>>
{ai_section}
<<<END_AI_SECTION>>>
""".strip()


COVERAGE_TEMPLATE = """
Transcript (context):
<<<TRANSCRIPT>>>
{transcript}
<<<END_TRANSCRIPT>>>

Reference {section_name} section:
<<<REFERENCE>>>
{reference_section}
<<<END_REFERENCE>>>

AI {section_name} section:
<<<AI_SECTION>>>
{ai_section}
<<<END_AI_SECTION>>>
""".strip()


@lru_cache(maxsize=1)
def _model_name() -> str:
    model = judge_config.model
    if ":" in model:
        return model
    return f"cerebras:{model}"


@lru_cache(maxsize=1)
def _section_agent() -> Agent:
    return Agent(model=_model_name(), output_type=SectionEvaluation, system_prompt=SECTION_SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def _coverage_agent() -> Agent:
    return Agent(model=_model_name(), output_type=CoverageEvaluation, system_prompt=COVERAGE_SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def _model_settings() -> ModelSettings:
    return ModelSettings(temperature=0.0)


def _clamp_sections(evaluation: SectionEvaluation) -> SectionEvaluation:
    scores = evaluation.scores
    clamped_scores = SectionScores(
        consistency=float(max(0.0, min(scores.consistency, 5.0))),
        completeness=float(max(0.0, min(scores.completeness, 5.0))),
        coherence=float(max(0.0, min(scores.coherence, 5.0))),
        fluency=float(max(0.0, min(scores.fluency, 5.0))),
    )
    issues = evaluation.issues.model_copy(deep=True)
    return SectionEvaluation(scores=clamped_scores, issues=issues, summary=evaluation.summary)


def _clamp_coverage(value: float) -> float:
    if value > 5:
        value = value / 20 if value <= 100 else value
    return round(max(0.0, min(value, 5.0)), 3)


async def _score_section(section: str, transcript: str, ai_text: str) -> SectionEvaluation:
    ai_text = (ai_text or "").strip()
    if not ai_text:
        return SectionEvaluation(
            scores=SectionScores(consistency=0.0, completeness=0.0, coherence=0.0, fluency=0.0),
            issues=SectionIssues(missing=["Section is empty"], unsupported=[], clinical_errors=[]),
            summary="Section contained no content to evaluate.",
        )

    prompt = SECTION_TEMPLATE.format(transcript=transcript.strip(), section_name=SECTION_NAMES[section], ai_section=ai_text)
    last_error: Optional[Exception] = None
    retries = max(1, judge_config.max_retries)
    for _ in range(retries):
        try:
            result = await _section_agent().run(prompt, model_settings=_model_settings())
            return _clamp_sections(result.output)
        except exceptions.UnexpectedModelBehavior as exc:
            last_error = exc
            await asyncio.sleep(0.5)
            continue
        except Exception as exc:  # pragma: no cover - transport or API issues
            last_error = exc
            await asyncio.sleep(0.5)
            continue

    summary = "Judge failed" if last_error else "Judge returned no result"
    return SectionEvaluation(
        scores=SectionScores(consistency=0.0, completeness=0.0, coherence=0.0, fluency=0.0),
        issues=SectionIssues(missing=[summary], unsupported=[], clinical_errors=[]),
        summary=summary,
    )


async def _score_coverage(section: str, transcript: str, ai_text: str, reference: str) -> CoverageEvaluation:
    reference = (reference or "").strip()
    if not reference:
        return CoverageEvaluation(coverage_score=0.0, summary=None)

    prompt = COVERAGE_TEMPLATE.format(
        transcript=transcript.strip(),
        section_name=SECTION_NAMES[section],
        reference_section=reference,
        ai_section=ai_text.strip(),
    )
    last_error: Optional[Exception] = None
    retries = max(1, judge_config.max_retries)
    for _ in range(retries):
        try:
            result = await _coverage_agent().run(prompt, model_settings=_model_settings())
            coverage = result.output
            coverage.coverage_score = _clamp_coverage(coverage.coverage_score)
            return coverage
        except exceptions.UnexpectedModelBehavior as exc:
            last_error = exc
            await asyncio.sleep(0.5)
            continue
        except Exception as exc:  # pragma: no cover
            last_error = exc
            await asyncio.sleep(0.5)
            continue

    summary = "Coverage judge failed" if last_error else None
    return CoverageEvaluation(coverage_score=0.0, summary=summary)


def _average(per_section: Dict[str, SectionEvaluation], metric: str) -> float:
    values = [getattr(evaluation.scores, metric) for evaluation in per_section.values()]
    return round(sum(values) / len(values), 3) if values else 0.0


def _collect_issues(per_section: Dict[str, SectionEvaluation]) -> Dict[str, List[str]]:
    collected: Dict[str, List[str]] = {"missing": [], "unsupported": [], "clinical_errors": []}
    for key, evaluation in per_section.items():
        prefix = SECTION_NAMES[key]
        for issue_type, entries in json.loads(evaluation.issues.model_dump_json()).items():
            for entry in entries:
                collected[issue_type].append(f"{prefix}: {entry}")
    return {name: items for name, items in collected.items() if items}


def _coverage_summary(results: Dict[str, CoverageEvaluation]) -> Optional[Dict[str, List[str]]]:
    if not results:
        return None
    missing: List[str] = []
    extra: List[str] = []
    for key, result in results.items():
        prefix = SECTION_NAMES[key]
        missing.extend(f"{prefix}: {item}" for item in result.missing_from_ai)
        extra.extend(f"{prefix}: {item}" for item in result.extraneous_in_ai)
    summary: Dict[str, List[str]] = {}
    if missing:
        summary["missing_from_ai"] = missing
    if extra:
        summary["extraneous_in_ai"] = extra
    return summary or None


def run_llm_judge(transcript: str, ai_soap: Dict[str, str], gold_soap: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, List[str]]]:
    async def evaluate() -> Dict[str, Dict[str, List[str]]]:
        per_section = dict(zip(SECTION_KEYS, await asyncio.gather(*[
            _score_section(section, transcript, ai_soap.get(section, ""))
            for section in SECTION_KEYS
        ])))

        coverage_results: Dict[str, CoverageEvaluation] = {}
        if gold_soap:
            sections = []
            tasks = []
            for section in SECTION_KEYS:
                reference = gold_soap.get(section, "") or ""
                if reference.strip():
                    sections.append(section)
                    tasks.append(_score_coverage(section, transcript, ai_soap.get(section, ""), reference))
            if tasks:
                coverage_results = dict(zip(sections, await asyncio.gather(*tasks)))

        scores = {
            "consistency": _average(per_section, "consistency"),
            "completeness": _average(per_section, "completeness"),
            "coherence": _average(per_section, "coherence"),
            "fluency": _average(per_section, "fluency"),
        }
        if coverage_results:
            scores["coverage"] = round(
                sum(_clamp_coverage(result.coverage_score) for result in coverage_results.values())
                / len(coverage_results),
                3,
            )

        sections_payload = {}
        for section, evaluation in per_section.items():
            payload: Dict[str, object] = {
                "scores": json.loads(evaluation.scores.model_dump_json()),
                "issues": json.loads(evaluation.issues.model_dump_json()),
                "summary": evaluation.summary,
                "overall": round(
                    (
                        evaluation.scores.consistency
                        + evaluation.scores.completeness
                        + evaluation.scores.coherence
                        + evaluation.scores.fluency
                    )
                    / 4.0,
                    3,
                ),
            }
            if section in coverage_results:
                payload["coverage"] = json.loads(coverage_results[section].model_dump_json())
            sections_payload[section] = payload

        result: Dict[str, Dict[str, List[str]]] = {
            "scores": scores,
            "issues": _collect_issues(per_section),
            "sections": sections_payload,
        }
        coverage_summary = _coverage_summary(coverage_results)
        if coverage_summary:
            result["coverage"] = coverage_summary
        return result

    async def runner() -> Dict[str, Dict[str, List[str]]]:
        timeout = judge_config.timeout_seconds
        if timeout and timeout > 0:
            with anyio.fail_after(timeout):
                return await evaluate()
        return await evaluate()

    return anyio.run(runner)


register("llm_judge", run_llm_judge)
