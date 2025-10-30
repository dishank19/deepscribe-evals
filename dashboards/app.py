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

FALLBACK_ISSUE_MARKERS = {"Judge fallback failed", "Judge fallback failed (504)"}

METRIC_DESCRIPTIONS = {
    "summac_overall": (
        "SummaC is a factual consistency score based on Natural Language Inference. "
        "It estimates how well each SOAP section is supported by the transcript. "
        "Scores close to 1.0 indicate strong grounding; lower scores point to missing or hallucinatory content."
    ),
    "judge_consistency": "0–5: factual alignment between the AI SOAP section and the transcript.",
    "judge_completeness": "0–5: coverage of clinically important facts from the transcript.",
    "judge_coherence": "0–5: logical SOAP organisation and readability.",
    "judge_fluency": "0–5: clarity and professional tone.",
    "judge_coverage": "0–5: when a clinician gold note is available, this reflects overlap with it.",
}

OVERVIEW_SYSTEM_PROMPT = """
You are helping a clinical QA team interpret automated SOAP-note evaluations.
Given dataset statistics captured in JSON, write at most 3 tight sentences covering:
- overall consistency and coverage trends,
- common missing or unsupported findings,
- any reliability caveats or follow-up actions.
Return the OverviewSummary JSON exactly.
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
            row = {
                "id": raw.get("id"),
                "transcript": raw.get("transcript", ""),
                "ai_soap": raw.get("ai_soap", {}),
                "gold_soap": raw.get("gold_soap", {}),
                "summac_overall": metrics.get("summac", {}).get("scores", {}).get("overall"),
                "summac_sections": metrics.get("summac", {}).get("scores", {}),
                "summac_flags": metrics.get("summac", {}).get("issues", {}).get("thresholds", []),
                "judge_scores": metrics.get("llm_judge", {}).get("scores", {}),
                "judge_sections": metrics.get("llm_judge", {}).get("sections", {}),
                "judge_issues": metrics.get("llm_judge", {}).get("issues", {}),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # Flatten judge numeric metrics for easier filtering
    if not df.empty:
        for metric in ["consistency", "completeness", "coherence", "fluency", "coverage"]:
            df[f"judge_{metric}"] = df["judge_scores"].apply(
                lambda scores: scores.get(metric) if isinstance(scores, dict) else None
            )
    return df


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Aggregated metrics")
    n_rows = len(df)
    cards = st.columns(3)

    with cards[0]:
        avg_sumac = df["summac_overall"].mean()
        below = df["summac_overall"].lt(summac_cfg.min_overall).sum()
        st.metric(
            "SummaC overall",
            f"{avg_sumac:.3f}" if pd.notna(avg_sumac) else "—",
            f"{below}/{n_rows} below {summac_cfg.min_overall}",
        )
    with cards[1]:
        avg_consistency = df["judge_consistency"].mean()
        st.metric("Judge consistency", f"{avg_consistency:.3f}" if pd.notna(avg_consistency) else "—")
    with cards[2]:
        if df["judge_coverage"].notna().any():
            avg_cov = df["judge_coverage"].mean()
            st.metric("Coverage vs. gold", f"{avg_cov:.3f}" if pd.notna(avg_cov) else "—")
        else:
            st.metric("Coverage vs. gold", "—")

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

    judge_means: Dict[str, Optional[float]] = {}
    for metric in ["consistency", "completeness", "coherence", "fluency", "coverage"]:
        column = f"judge_{metric}"
        if column in df and df[column].notna().any():
            mean_value = df[column].mean()
            judge_means[metric] = round(float(mean_value), 3) if pd.notna(mean_value) else None
        else:
            judge_means[metric] = None
    sumac_flag_count = int(
        sum(len(flags or []) for flags in df["summac_flags"] if isinstance(flags, list))
    )
    issue_counts_payload = {
        category: issue_counts[category].most_common(5)
        for category in issue_counts
    }
    coverage_available = int(df["gold_soap"].apply(lambda x: bool(x)).sum())

    summary_context = {
        "rows": n_rows,
        "sumac": {
            "average": round(avg_sumac, 3) if pd.notna(avg_sumac) else None,
            "below_threshold": int(below),
            "threshold": summac_cfg.min_overall,
            "flagged_sections": sumac_flag_count,
        },
        "judge_scores": judge_means,
        "issue_counts": issue_counts_payload,
        "coverage": {
            "available_rows": coverage_available,
            "average_score": judge_means.get("coverage"),
        },
    }

    summary_text = generate_overview_summary(summary_context)
    st.divider()
    st.subheader("LLM overview")
    st.write(summary_text)

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
    col1, col2 = st.columns(2)
    with col1:
        min_sumac = st.slider("Min SummaC overall", 0.0, 1.0, 0.0, 0.05)
    with col2:
        min_consistency = st.slider("Min judge consistency", 0.0, 5.0, 0.0, 0.25)

    filtered = df[
        df["summac_overall"].fillna(0) >= min_sumac
    ]
    filtered = filtered[filtered["judge_consistency"].fillna(0) >= min_consistency]

    display_cols = ["id", "summac_overall", "judge_consistency", "judge_completeness", "judge_coherence", "judge_fluency"]
    if not filtered.empty:
        table = filtered[display_cols]
        st.dataframe(
            table.rename(columns=lambda c: c.replace("judge_", "Judge ").replace("_", " ").title()),
            width="stretch",
        )
    else:
        st.dataframe(pd.DataFrame(columns=display_cols), width="stretch")

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    selected_id = st.selectbox("Inspect row", filtered["id"])
    row = filtered[filtered["id"] == selected_id].iloc[0]

    st.markdown(f"### Row {selected_id}")
    st.caption("SummaC and judge scores")
    score_cols = st.columns(5)
    metrics = [
        ("Summac overall", row.get("summac_overall")),
        ("Judge consistency", row.get("judge_consistency")),
        ("Judge completeness", row.get("judge_completeness")),
        ("Judge coherence", row.get("judge_coherence")),
        ("Judge fluency", row.get("judge_fluency")),
    ]
    for col, (label, value) in zip(score_cols, metrics):
        col.metric(label, f"{value:.3f}" if pd.notna(value) else "—")
        key = label.lower().replace(" ", "_")
        description = METRIC_DESCRIPTIONS.get(key)
        if description:
            col.caption(description)

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
            summary = section_data.get("summary")
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
