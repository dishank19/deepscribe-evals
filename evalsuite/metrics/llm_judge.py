"""LLM-as-a-judge metric with optional gold-note coverage."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Dict, List, Optional

import anyio
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from pydantic_evals.evaluators.llm_as_a_judge import (
    judge_input_output,
    judge_input_output_expected,
    set_default_judge_model,
)

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


EVAL_RUBRIC = """
You are grading a SOAP section against the transcript. Return JSON:
{
  "scores": {"consistency": float, "completeness": float, "coherence": float, "fluency": float},
  "issues": {"missing": [string], "unsupported": [string], "clinical_errors": [string]},
  "summary": string|null
}
Rules: scores 0–5; `missing` top 3 omissions; `unsupported` top 2 hallucinations; `clinical_errors` at most 1 key risk (omit if none); `summary` ≤1 short sentence or null.
""".strip()


COVERAGE_RUBRIC = """
Compare AI SOAP to the clinician reference. Return JSON:
{
  "coverage_score": float,
  "missing_from_ai": [string],
  "extraneous_in_ai": [string],
  "summary": string|null
}
Rules: scores 0–5; list up to 3 items in `missing_from_ai` and `extraneous_in_ai`; summary ≤1 sentence or null.
""".strip()


FALLBACK_SECTION_SYSTEM = """
Produce the JSON schema exactly:
- Scores 0–5.
- `missing`: up to 3 key omissions.
- `unsupported`: up to 2 hallucinations.
- `clinical_errors`: at most 1 risk (omit if none).
- `summary`: ≤1 short sentence or null.
""".strip()


FALLBACK_COVERAGE_SYSTEM = """
Produce the JSON schema exactly:
- `coverage_score` 0–5.
- `missing_from_ai`: up to 3 omissions.
- `extraneous_in_ai`: up to 3 extras.
- `summary`: ≤1 short sentence or null.
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
    provider = "openai" if model.startswith("gpt") else "anthropic"
    return f"{provider}:{model}"


@lru_cache(maxsize=1)
def _model_settings() -> Optional[ModelSettings]:
    if _model_name().startswith("openai:"):
        return ModelSettings(response_format={"type": "json_object"})
    return None


@lru_cache(maxsize=1)
def _section_agent() -> Agent:
    return Agent(model=_model_name(), output_type=SectionEvaluation, system_prompt=FALLBACK_SECTION_SYSTEM)


@lru_cache(maxsize=1)
def _coverage_agent() -> Agent:
    return Agent(model=_model_name(), output_type=CoverageEvaluation, system_prompt=FALLBACK_COVERAGE_SYSTEM)


def _loads(payload: str) -> Optional[dict]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _clamp_sections(evaluation: SectionEvaluation) -> SectionEvaluation:
    data = evaluation.model_dump()
    for metric, value in data["scores"].items():
        data["scores"][metric] = float(max(0.0, min(value, 5.0)))
    return SectionEvaluation.model_validate(data)


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

    for attempt in range(judge_config.max_retries):
        try:
            grading = await judge_input_output(
                inputs=f"Transcript:\n{transcript.strip()}",
                output=f"{SECTION_NAMES[section]} section:\n{ai_text}",
                rubric=EVAL_RUBRIC,
                model=_model_name(),
                model_settings=_model_settings(),
            )
            parsed = _loads(grading.reason)
            if parsed is not None:
                return _clamp_sections(SectionEvaluation.model_validate(parsed))
        except ValidationError:
            break
        except Exception:
            await asyncio.sleep(1.5 * (attempt + 1))

    prompt = SECTION_TEMPLATE.format(transcript=transcript.strip(), section_name=SECTION_NAMES[section], ai_section=ai_text)
    fallback = await _section_agent().run(prompt, model_settings=_model_settings())
    return _clamp_sections(fallback.output)


async def _score_coverage(section: str, transcript: str, ai_text: str, reference: str) -> CoverageEvaluation:
    reference = (reference or "").strip()
    if not reference:
        return CoverageEvaluation(coverage_score=0.0, summary=None)

    for attempt in range(judge_config.max_retries):
        try:
            grading = await judge_input_output_expected(
                inputs=f"Transcript:\n{transcript.strip()}",
                output=f"AI {SECTION_NAMES[section]} section:\n{ai_text.strip()}",
                expected_output=f"Reference {SECTION_NAMES[section]} section:\n{reference}",
                rubric=COVERAGE_RUBRIC,
                model=_model_name(),
                model_settings=_model_settings(),
            )
            parsed = _loads(grading.reason)
            if parsed is not None:
                coverage = CoverageEvaluation.model_validate(parsed)
                coverage.coverage_score = _clamp_coverage(coverage.coverage_score)
                return coverage
        except ValidationError:
            break
        except Exception:
            await asyncio.sleep(1.5 * (attempt + 1))

    prompt = COVERAGE_TEMPLATE.format(
        transcript=transcript.strip(),
        section_name=SECTION_NAMES[section],
        reference_section=reference,
        ai_section=ai_text.strip(),
    )
    fallback = await _coverage_agent().run(prompt, model_settings=_model_settings())
    coverage = fallback.output
    coverage.coverage_score = _clamp_coverage(coverage.coverage_score)
    return coverage


def _average(per_section: Dict[str, SectionEvaluation], metric: str) -> float:
    values = [getattr(evaluation.scores, metric) for evaluation in per_section.values()]
    return round(sum(values) / len(values), 3) if values else 0.0


def _collect_issues(per_section: Dict[str, SectionEvaluation]) -> Dict[str, List[str]]:
    collected: Dict[str, List[str]] = {"missing": [], "unsupported": [], "clinical_errors": []}
    for key, evaluation in per_section.items():
        prefix = SECTION_NAMES[key]
        for issue_type, entries in evaluation.issues.model_dump().items():
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
    set_default_judge_model(_model_name())

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
                "scores": evaluation.scores.model_dump(),
                "issues": evaluation.issues.model_dump(),
                "summary": evaluation.summary,
                "overall": round(sum(evaluation.scores.model_dump().values()) / 4.0, 3),
            }
            if section in coverage_results:
                payload["coverage"] = coverage_results[section].model_dump()
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
