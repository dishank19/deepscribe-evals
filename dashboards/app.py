"""Streamlit dashboard for viewing evaluation metrics."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

# Ensure project root is importable when running `streamlit run`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evalsuite.config import judge as judge_cfg  # pylint: disable=wrong-import-position
from evalsuite.config import summac as summac_cfg  # pylint: disable=wrong-import-position

SECTION_KEYS = ("S", "O", "A", "P")
SECTION_NAMES = {"S": "Subjective", "O": "Objective", "A": "Assessment", "P": "Plan"}

DATA_DIR = Path("data/augmented")
DEFAULT_PATTERN = "*scored*.jsonl"
BAD_EXAMPLES_PATH = DATA_DIR / "bad_examples_scored.jsonl"
SUMMAC_GOLD_PATH = DATA_DIR / "summac_gold.jsonl"

FALLBACK_ISSUE_MARKERS = {"Judge fallback failed", "Judge fallback failed (504)"}

METRIC_DESCRIPTIONS = {
    "summac_overall": (
        "SummaC (transcript) checks whether the AI SOAP is supported by the encounter transcript. "
        "Scores close to 1.0 mean the note sticks to what was said; low scores flag potential hallucinations."
    ),
    "sumac_gold_overall": (
        "SummaC (gold) compares the AI SOAP to the clinician-edited gold note, giving a feel for how closely the AI matched the reference write-up."
    ),
    "judge_consistency": "0–5: factual alignment between the AI SOAP section and the transcript.",
    "judge_completeness": "0–5: coverage of clinically important facts from the transcript.",
    "judge_coherence": "0–5: logical SOAP organisation and readability.",
    "judge_fluency": "0–5: clarity and professional tone.",
    "judge_coverage": "0–5: when a clinician gold note is available, this reflects overlap with it.",
    "bertscore_overall_f1": "0–1: semantic similarity between the AI and gold SOAP using BERTScore F1.",
    "rouge_overall_rougeL": "0–1: ROUGE-L overlap between the AI and gold SOAP (lexical coverage).",
    "bertscore_f1": "0–1: semantic similarity between AI and gold SOAP using BERTScore F1.",
    "rouge_rougel": "0–1: ROUGE-L lexical overlap between AI and gold SOAP.",
}

OVERVIEW_SYSTEM_PROMPT = """
You are helping a clinical QA team interpret automated SOAP-note evaluations.
Given aggregate metrics in JSON, write at most 3 concise sentences that:
- describe high-level consistency/completeness trends without citing patient-specific details,
- summarize the most common issue categories generically (e.g., missing Objective data, unsupported Assessment claims),
- call out any follow-up actions or caveats (re-run judge, review low SummaC rows).
Do not mention specific diseases, genetic markers, or lab values. Return the OverviewSummary JSON exactly.
""".strip()


class OverviewSummary(BaseModel):
    summary: str


SUMMARY_MODEL_SETTINGS = ModelSettings(temperature=0.0)


@st.cache_resource(show_spinner=False)
def _overview_agent() -> Agent:
    return Agent(model=judge_cfg.model, output_type=OverviewSummary, system_prompt=OVERVIEW_SYSTEM_PROMPT)


@st.cache_data(show_spinner=False)
def generate_overview_summary(payload: Dict[str, object]) -> str:
    agent = _overview_agent()
    try:
        result = agent.run_sync(
            f"DATA SNAPSHOT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
            model_settings=SUMMARY_MODEL_SETTINGS,
        )
        return result.output.summary.strip()
    except Exception:
        return (
            "Summary unavailable right now. Review the metrics above, or rerun once the model endpoint is reachable."
        )


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            metrics = raw.get("metrics", {})
            llm_judge = metrics.get("llm_judge", {})
            baseline = llm_judge.get("baseline", {})
            row = {
                "id": raw.get("id"),
                "transcript": raw.get("transcript", ""),
                "ai_soap": raw.get("ai_soap", {}),
                "gold_soap": raw.get("gold_soap", {}),
                "summac_overall": metrics.get("summac", {}).get("scores", {}).get("overall"),
                "summac_sections": metrics.get("summac", {}).get("scores", {}),
                "summac_flags": metrics.get("summac", {}).get("issues", {}).get("thresholds", []),
                "judge_scores": llm_judge.get("scores", {}),
                "judge_sections": llm_judge.get("sections", {}),
                "judge_issues": llm_judge.get("issues", {}),
                "judge_baseline_scores": baseline.get("scores", {}),
                "judge_baseline_sections": baseline.get("sections", {}),
                "judge_deltas": llm_judge.get("deltas", {}),
                "rouge_scores": metrics.get("rouge", {}).get("scores", {}),
                "bertscore_scores": metrics.get("bertscore", {}).get("scores", {}),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # Flatten judge numeric metrics for easier filtering
    if not df.empty:
        for metric in ["consistency", "completeness", "coherence", "fluency", "coverage"]:
            df[f"judge_{metric}"] = df["judge_scores"].apply(
                lambda scores: scores.get(metric) if isinstance(scores, dict) else None
            )
            df[f"judge_baseline_{metric}"] = df["judge_baseline_scores"].apply(
                lambda scores: scores.get(metric) if isinstance(scores, dict) else None
            )
            df[f"judge_delta_{metric}"] = df["judge_deltas"].apply(
                lambda scores: scores.get(metric) if isinstance(scores, dict) else None
            )
        df["rouge_overall_rougeL"] = df["rouge_scores"].apply(
            lambda scores: scores.get("overall_rougeL") if isinstance(scores, dict) else None
        )
        df["bertscore_overall_f1"] = df["bertscore_scores"].apply(
            lambda scores: scores.get("overall_f1") if isinstance(scores, dict) else None
        )
        df["rouge_flag"] = df["rouge_overall_rougeL"].apply(
            lambda value: value is not None and value < 0.2
        )
        df["bertscore_flag"] = df["bertscore_overall_f1"].apply(
            lambda value: value is not None and value < 0.3
        )

        sumac_gold_mapping = load_sumac_gold(SUMMAC_GOLD_PATH)
        if sumac_gold_mapping:
            df["sumac_gold_overall"] = df["id"].astype(str).apply(
                lambda rid: sumac_gold_mapping.get(rid, {}).get("overall")
            )
            for section in SECTION_KEYS:
                df[f"sumac_gold_{section}"] = df["id"].astype(str).apply(
                    lambda rid: sumac_gold_mapping.get(rid, {}).get(f"{section}_consistency")
                )
    return df


@st.cache_data(show_spinner=False)
def load_sumac_gold(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    mapping: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            mapping[str(raw.get("id"))] = raw.get("sumac_gold", {})
    return mapping


@st.cache_data(show_spinner=False)
def load_bad_examples_summary(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    df = load_dataset(path)
    if df.empty:
        return None
    rouge_series = pd.to_numeric(df["rouge_overall_rougeL"], errors="coerce")
    bert_series = pd.to_numeric(df["bertscore_overall_f1"], errors="coerce")
    judge_series = pd.to_numeric(df["judge_consistency"], errors="coerce")
    judge_baseline_series = pd.to_numeric(df.get("judge_baseline_consistency"), errors="coerce")
    judge_delta_series = pd.to_numeric(df.get("judge_delta_consistency"), errors="coerce")
    summary = {
        "rows": len(df),
        "rouge_avg": round(rouge_series.mean(), 3) if rouge_series.notna().any() else None,
        "bertscore_avg": round(bert_series.mean(), 3) if bert_series.notna().any() else None,
        "judge_consistency": round(judge_series.mean(), 3) if judge_series.notna().any() else None,
        "judge_baseline_consistency": round(judge_baseline_series.mean(), 3) if judge_baseline_series.notna().any() else None,
        "judge_delta_consistency": round(judge_delta_series.mean(), 3) if judge_delta_series.notna().any() else None,
        "rouge_flags": int(df["rouge_flag"].fillna(False).sum()) if "rouge_flag" in df else 0,
        "bertscore_flags": int(df["bertscore_flag"].fillna(False).sum()) if "bertscore_flag" in df else 0,
    }
    return summary


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Aggregated metrics")
    n_rows = len(df)
    primary_cards = st.columns(5)

    sumac_series = pd.to_numeric(df["summac_overall"], errors="coerce")
    avg_sumac = sumac_series.mean()
    below = sumac_series.lt(summac_cfg.min_overall).sum() if not sumac_series.empty else 0
    with primary_cards[0]:
        st.metric(
            "SummaC (transcript)",
            f"{avg_sumac:.3f}" if pd.notna(avg_sumac) else "—",
            f"{below}/{n_rows} below {summac_cfg.min_overall}",
        )

    sumac_gold_series = pd.to_numeric(df.get("sumac_gold_overall"), errors="coerce")
    avg_sumac_gold = sumac_gold_series.mean()
    with primary_cards[1]:
        st.metric("SummaC (gold)", f"{avg_sumac_gold:.3f}" if pd.notna(avg_sumac_gold) else "—")

    bert_series = pd.to_numeric(df["bertscore_overall_f1"], errors="coerce")
    avg_bertscore = bert_series.mean()
    with primary_cards[2]:
        st.metric("BERTScore F1", f"{avg_bertscore:.3f}" if pd.notna(avg_bertscore) else "—")

    rouge_series = pd.to_numeric(df["rouge_overall_rougeL"], errors="coerce")
    avg_rouge = rouge_series.mean()
    with primary_cards[3]:
        st.metric("ROUGE-L", f"{avg_rouge:.3f}" if pd.notna(avg_rouge) else "—")

    coverage_series = pd.to_numeric(df["judge_coverage"], errors="coerce") if "judge_coverage" in df else pd.Series(dtype=float)
    avg_cov = coverage_series.mean() if not coverage_series.empty else float("nan")
    baseline_cov = pd.to_numeric(df.get("judge_baseline_coverage"), errors="coerce").mean() if "judge_baseline_coverage" in df else float("nan")
    delta_cov = pd.to_numeric(df.get("judge_delta_coverage"), errors="coerce").mean() if "judge_delta_coverage" in df else float("nan")
    coverage_delta_text = None
    if pd.notna(delta_cov):
        if pd.notna(baseline_cov):
            coverage_delta_text = f"{delta_cov:+.3f} vs gold {baseline_cov:.3f}"
        else:
            coverage_delta_text = f"{delta_cov:+.3f}"
    with primary_cards[4]:
        st.metric(
            "Judge coverage",
            f"{avg_cov:.3f}" if pd.notna(avg_cov) else "—",
            coverage_delta_text,
        )

    judge_metrics = ["consistency", "completeness", "coherence", "fluency"]
    judge_ai_means: Dict[str, Optional[float]] = {}
    judge_baseline_means: Dict[str, Optional[float]] = {}
    judge_delta_means: Dict[str, Optional[float]] = {}
    for metric in judge_metrics:
        judge_ai_means[metric] = pd.to_numeric(df.get(f"judge_{metric}"), errors="coerce").mean()
        judge_baseline_means[metric] = pd.to_numeric(df.get(f"judge_baseline_{metric}"), errors="coerce").mean()
        judge_delta_means[metric] = pd.to_numeric(df.get(f"judge_delta_{metric}"), errors="coerce").mean()

    judge_cols = st.columns(len(judge_metrics))
    for col, metric in zip(judge_cols, judge_metrics):
        label = f"Judge {metric}"
        avg_value = judge_ai_means.get(metric)
        avg_baseline = judge_baseline_means.get(metric)
        avg_delta = judge_delta_means.get(metric)
        delta_text = None
        if pd.notna(avg_delta):
            if pd.notna(avg_baseline):
                delta_text = f"{avg_delta:+.3f} vs gold {avg_baseline:.3f}"
            else:
                delta_text = f"{avg_delta:+.3f}"
        col.metric(label, f"{avg_value:.3f}" if pd.notna(avg_value) else "—", delta_text)

    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown(f"- **SummaC overall** – {METRIC_DESCRIPTIONS['summac_overall']}")
        st.markdown(
            "- **Judge consistency/completeness/coherence/fluency** – "
            " consistency (truthfulness), completeness (coverage), coherence (structure), fluency (writing quality) scored 0–5."
        )
        st.markdown(f"- **Judge coverage** – {METRIC_DESCRIPTIONS['judge_coverage']}")
        st.caption("Counts in parentheses show how many notes triggered each issue.")

    issue_counts = {"missing": Counter(), "unsupported": Counter(), "clinical_errors": Counter()}
    for sections in df["judge_sections"]:
        if not isinstance(sections, dict):
            continue
        for section_key, section_data in sections.items():
            issues = section_data.get("issues", {})
            for category in issue_counts:
                filtered = [
                    entry for entry in issues.get(category, []) if entry not in FALLBACK_ISSUE_MARKERS
                ]
                issue_counts[category].update(filtered)

    judge_scores_summary = {
        metric: (round(float(value), 3) if pd.notna(value) else None)
        for metric, value in {
            **{m: judge_ai_means.get(m) for m in judge_metrics},
            "coverage": avg_cov,
        }.items()
    }
    judge_baseline_summary = {
        metric: (round(float(value), 3) if pd.notna(value) else None)
        for metric, value in {
            **{m: judge_baseline_means.get(m) for m in judge_metrics},
            "coverage": baseline_cov,
        }.items()
    }
    judge_delta_summary = {
        metric: (round(float(value), 3) if pd.notna(value) else None)
        for metric, value in {
            **{m: judge_delta_means.get(m) for m in judge_metrics},
            "coverage": delta_cov,
        }.items()
    }
    sumac_flag_count = int(
        sum(len(flags or []) for flags in df["summac_flags"] if isinstance(flags, list))
    )
    issue_counts_payload = {
        category: issue_counts[category].most_common(5)
        for category in issue_counts
    }
    coverage_available = int(df["gold_soap"].apply(lambda x: bool(x)).sum())

    rouge_available = int(rouge_series.notna().sum())
    bert_available = int(bert_series.notna().sum())

    summary_context = {
        "rows": n_rows,
        "sumac": {
            "average": round(avg_sumac, 3) if pd.notna(avg_sumac) else None,
            "below_threshold": int(below),
            "threshold": summac_cfg.min_overall,
            "flagged_sections": sumac_flag_count,
        },
        "sumac_gold": {
            "average": round(avg_sumac_gold, 3) if pd.notna(avg_sumac_gold) else None,
        },
        "judge_scores": judge_scores_summary,
        "judge_baseline": judge_baseline_summary,
        "judge_deltas": judge_delta_summary,
        "issue_counts": issue_counts_payload,
        "coverage": {
            "available_rows": coverage_available,
            "average_score": judge_scores_summary.get("coverage"),
            "baseline": judge_baseline_summary.get("coverage"),
            "delta": judge_delta_summary.get("coverage"),
        },
        "rouge": {
            "available_rows": rouge_available,
            "average_rougeL": round(rouge_series.mean(), 3)
            if rouge_series.notna().any()
            else None,
        },
        "bertscore": {
            "available_rows": bert_available,
            "average_f1": round(bert_series.mean(), 3)
            if bert_series.notna().any()
            else None,
        },
    }

    summary_text = generate_overview_summary(summary_context)
    st.divider()
    st.subheader("LLM overview")
    st.write(summary_text)

    bad_examples_summary = load_bad_examples_summary(BAD_EXAMPLES_PATH)
    if bad_examples_summary:
        delta_text = ""
        if (
            bad_examples_summary.get("judge_delta_consistency") is not None
            and bad_examples_summary.get("judge_baseline_consistency") is not None
        ):
            delta_text = (
                f" (Δ {bad_examples_summary['judge_delta_consistency']:+.3f} vs gold "
                f"{bad_examples_summary['judge_baseline_consistency']:.3f})"
            )
        st.info(
            "Synthetic bad examples "
            f"({bad_examples_summary['rows']} rows): "
            f"ROUGE-L avg {bad_examples_summary['rouge_avg']}, "
            f"BERTScore F1 avg {bad_examples_summary['bertscore_avg']}, "
            f"Judge consistency avg {bad_examples_summary['judge_consistency']}{delta_text}. "
            f"Flags → ROUGE {bad_examples_summary['rouge_flags']}/{bad_examples_summary['rows']}, "
            f"BERTScore {bad_examples_summary['bertscore_flags']}/{bad_examples_summary['rows']}."
        )

    with st.expander("Frequent judge findings", expanded=False):
        cols = st.columns(3)
        for idx, category in enumerate(issue_counts):
            with cols[idx]:
                st.markdown(f"**{category.replace('_', ' ').title()}**")
                top_items = issue_counts[category].most_common(5)
                if not top_items:
                    st.caption("No findings")
                for item, count in top_items:
                    st.write(f"- {item} ({count})")


def render_row_view(df: pd.DataFrame) -> None:
    st.subheader("Row explorer")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_sumac = st.slider("Min SummaC overall", 0.0, 1.0, 0.0, 0.05)
    with col2:
        min_consistency = st.slider("Min judge consistency", 0.0, 5.0, 0.0, 0.25)
    with col3:
        show_only_flags = st.checkbox("Show only flagged rows (low ROUGE/BERTScore)", value=False)

    filtered = df[
        df["summac_overall"].fillna(0) >= min_sumac
    ]
    filtered = filtered[filtered["judge_consistency"].fillna(0) >= min_consistency]
    if show_only_flags:
        filtered = filtered[(filtered["rouge_flag"] == True) | (filtered["bertscore_flag"] == True)]

    display_cols = [
        "id",
        "summac_overall",
        "sumac_gold_overall",
        "judge_consistency",
        "judge_delta_consistency",
        "judge_completeness",
        "judge_delta_completeness",
        "judge_coherence",
        "judge_delta_coherence",
        "judge_fluency",
        "judge_delta_fluency",
        "judge_coverage",
        "judge_delta_coverage",
        "bertscore_overall_f1",
        "rouge_overall_rougeL",
    ]
    if not filtered.empty:
        table = filtered[display_cols]
        rename_map = {
            "summac_overall": "SummaC (transcript)",
            "sumac_gold_overall": "SummaC (gold)",
            "judge_consistency": "Judge consistency",
            "judge_delta_consistency": "Δ Consistency",
            "judge_completeness": "Judge completeness",
            "judge_delta_completeness": "Δ Completeness",
            "judge_coherence": "Judge coherence",
            "judge_delta_coherence": "Δ Coherence",
            "judge_fluency": "Judge fluency",
            "judge_delta_fluency": "Δ Fluency",
            "judge_coverage": "Judge coverage",
            "judge_delta_coverage": "Δ Coverage",
            "bertscore_overall_f1": "BERTScore F1",
            "rouge_overall_rougeL": "ROUGE-L",
        }
        st.dataframe(table.rename(columns=rename_map), width="stretch")
    else:
        st.dataframe(pd.DataFrame(columns=display_cols), width="stretch")

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    selected_id = st.selectbox("Inspect row", filtered["id"])
    row = filtered[filtered["id"] == selected_id].iloc[0]

    st.markdown(f"### Row {selected_id}")
    st.caption("SummaC, judge, and reference scores")
    metric_specs = [
        ("summac_overall", "SummaC (transcript)", None),
        ("sumac_gold_overall", "SummaC (gold)", None),
        ("judge_consistency", "Judge consistency", "consistency"),
        ("judge_completeness", "Judge completeness", "completeness"),
        ("judge_coherence", "Judge coherence", "coherence"),
        ("judge_fluency", "Judge fluency", "fluency"),
        ("judge_coverage", "Judge coverage", "coverage"),
        ("bertscore_overall_f1", "BERTScore F1", None),
        ("rouge_overall_rougeL", "ROUGE-L", None),
    ]
    score_cols = st.columns(len(metric_specs))
    for col, (column_name, label, judge_key) in zip(score_cols, metric_specs):
        value = row.get(column_name)
        col.metric(label, f"{value:.3f}" if pd.notna(value) else "—")
        description = METRIC_DESCRIPTIONS.get(column_name) or METRIC_DESCRIPTIONS.get(label.lower().replace(" ", "_"))
        caption_parts: List[str] = []
        if judge_key:
            baseline_value = row.get(f"judge_baseline_{judge_key}")
            delta_value = row.get(f"judge_delta_{judge_key}")
            if pd.notna(baseline_value):
                caption_parts.append(f"Gold {baseline_value:.3f}")
            if pd.notna(delta_value):
                caption_parts.append(f"Δ {delta_value:+.3f}")
        if description:
            caption_parts.append(description)
        if caption_parts:
            col.caption("; ".join(caption_parts))

    st.caption("Text artefacts")
    with st.expander("Transcript", expanded=False):
        st.write(row["transcript"])
    with st.expander("AI SOAP", expanded=False):
        for section in SECTION_KEYS:
            st.markdown(f"**{SECTION_NAMES[section]}**")
            st.write(row["ai_soap"].get(section, ""))
    if isinstance(row.get("gold_soap"), dict) and row["gold_soap"]:
        with st.expander("Gold SOAP", expanded=False):
            for section in SECTION_KEYS:
                st.markdown(f"**{SECTION_NAMES[section]}**")
                st.write(row["gold_soap"].get(section, ""))

    st.caption("Judge section breakdown")
    judge_sections = row["judge_sections"] or {}
    for section in SECTION_KEYS:
        section_data = judge_sections.get(section, {})
        if not section_data:
            continue
        with st.expander(f"{SECTION_NAMES[section]} issues", expanded=False):
            scores = section_data.get("scores", {})
            st.markdown("**Scores**")
            st.write({k: round(v, 3) for k, v in scores.items()})
            for category, items in section_data.get("issues", {}).items():
                filtered_items = [item for item in items if item not in FALLBACK_ISSUE_MARKERS]
                if not filtered_items:
                    continue
                st.markdown(f"- **{category.replace('_', ' ').title()}**")
                for item in filtered_items:
                    st.write(f"  - {item}")
            baseline_scores = section_data.get("baseline_scores")
            baseline_overall = section_data.get("baseline_overall")
            delta_overall = section_data.get("delta_overall")
            summary = section_data.get("summary")
            if baseline_scores:
                st.markdown("**Gold baseline**")
                st.write({"overall": baseline_overall, **baseline_scores})
            if delta_overall is not None:
                st.caption(f"Δ overall {delta_overall:+.3f}")
            if summary:
                st.caption(summary)


def main() -> None:
    st.set_page_config(page_title="DeepScribe Metrics", layout="wide")
    st.title("DeepScribe Evaluation Dashboard")

    files = sorted(DATA_DIR.glob(DEFAULT_PATTERN))
    if not files:
        st.error(f"No files matching '{DEFAULT_PATTERN}' in {DATA_DIR.resolve()}")
        return

    selected = st.sidebar.selectbox("Dataset", files, format_func=lambda p: p.name)
    df = load_dataset(selected)

    view = st.sidebar.radio("View", ("Overview", "Rows"))
    if view == "Overview":
        render_overview(df)
    else:
        render_row_view(df)


if __name__ == "__main__":
    main()
