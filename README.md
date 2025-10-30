DeepScribe SOAP Evaluation Suite
================================

Quick-start guide for building the augmented dataset, running the evaluation metrics, and inspecting results.

Environment Setup
-----------------
- Install dependencies once with `uv sync`.
- Activate the managed venv if you need direct access: `source .venv/bin/activate`.
- Always wrap project commands with `uv run …` so they execute inside the managed environment.

Data Processing Pipeline
------------------------
1. **Source transcripts & gold notes.** We pull rows from `omi-health/medical-dialogue-to-soap-summary` via `datasets`. The split and slice are controlled by `scripts/build_dataset.py`.
2. **Gold SOAP parsing.** `evalsuite/dataset.py::parse_soap` normalises the labelled blocks (`S/O/A/P`) into structured dictionaries, preserving multi-line content.
3. **AI SOAP generation.** `evalsuite/gen/pydantic_claude.py` streams each transcript through a Pydantic-AI agent (Claude backend) that returns schema-validated SOAP sections. Generation retries are handled inside the agent wrapper so malformed responses never hit disk.
4. **Augmented dataset write.** `scripts/build_dataset.py` emits JSONL files under `data/augmented/<split>.jsonl` containing `transcript`, `gold_soap`, `ai_soap`, and an empty `metrics` stub. By default we cap at 100 rows (`--limit 100`) to keep iteration fast; drop the flag to cover the full split.
5. **Synthetic stress set.** `data/augmented/bad_examples.jsonl` holds five transcripts that were not part of the first 100 rows. Their AI SOAP sections are intentionally degraded (empty sections, hallucinated facts, etc.) so we can sanity-check the evaluator on obviously bad output. Generate it with `scripts/create_bad_examples.py`.

Common dataset operations:
```
# Build the first 100 rows
uv run python scripts/build_dataset.py --split train --limit 100

# Sample a record
uv run python view_dataset.py --split train --index 0

# Create the synthetic “bad” sample set (uses rows beyond the first 100)
uv run python scripts/create_bad_examples.py

# Score the synthetic rows with the full metric stack
uv run python -m evalsuite.runners.run_batch \
  --split bad_examples \
  --metrics summac,rouge,bertscore,llm_judge \
  --output data/augmented/bad_examples_scored.jsonl
```
> The synthetic command above reuses the build script with an offset and a helper that overwrites the AI SOAP with deliberately poor content. See the “Evaluator Quality & Synthetic Data” section for usage details.

Evaluation Metrics & Code Layout
--------------------------------
- **SummaC (`evalsuite/metrics/summac_eval.py`)**  
  Compares each AI section to the transcript with an NLI model. The overall gate now uses a 0.45 threshold: transcripts contain a lot of clinical shorthand, so exact entailment scores sit lower than general summarisation corpora. A low bar keeps recall high—SummaC acts as a fast “smoke check” before heavier metrics run. Because SummaC only sees the transcript, it cannot spot mismatches against the gold note; we rely on ROUGE/BERTScore for that.
- **ROUGE (`evalsuite/metrics/rouge_eval.py`)**  
  Uses `rouge-score` to compute ROUGE-L per section and overall. Scores below 0.2 usually signal missing sections or entirely different content; these rows are flagged in the dashboard.
- **BERTScore (`evalsuite/metrics/bert_score_eval.py`)**  
  Computes precision/recall/F1 against the gold SOAP (`roberta-large`, baseline-rescaled). An overall F1 under 0.3 is a strong indication the generated note deviates semantically, so we flag those rows as well.
- **LLM Judge (`evalsuite/metrics/llm_judge.py`)**  
  Cerebras-based Pydantic agent that delivers section-level scores (consistency, completeness, coherence, fluency), issue lists, and optional coverage when a gold note is available. Retries + schema enforcement ensure every row yields structured output.

Metric registration lives in `evalsuite/metrics/registry.py` and is invoked automatically by the runners. Configuration (model names, thresholds) is centralised in `evalsuite/config.py`.

Running Metrics
---------------
### Batch Runner
Evaluate any chunk of the augmented dataset.
```
uv run python -m evalsuite.runners.run_batch \
  --split train \
  --metrics summac,rouge,bertscore,llm_judge \
  --output data/augmented/train_scored.jsonl
```
Key arguments:
- `--split`: augmented file to read (looks under `data/augmented/<split>.jsonl`).
- `--metrics`: comma-separated registry names (`summac`, `rouge`, `bertscore`, `llm_judge`, …).
- `--start`, `--limit`: optional slice controls for chunked execution.
- `--output`: destination file. Use `.jsonl` for newline-delimited records or `.json` for a single array.
- `--append`: only valid with `.jsonl`; appends new results instead of overwriting.

### Single Row Runner
Re-run metrics for a specific record id (0-indexed when generated locally):
```
uv run python -m evalsuite.runners.run_one \
  --id 42 \
  --split train \
  --metrics summac,rouge,bertscore,llm_judge \
  --output data/augmented/single_row.json
```
Omit `--output` to print the evaluated record to stdout (gold SOAP is hidden by default).

Outputs
-------
- Generated dataset(s): `data/augmented/<split>.jsonl`
- Scored outputs: `data/augmented/<split>_scored.jsonl` (or `.json`)
- Synthetic stress set: `data/augmented/bad_examples.jsonl`
- Dashboard config expects scored files under `data/augmented/`.
- To backfill new metrics onto an existing scored file without recomputing others, load the JSONL, call the target metric functions (e.g., ROUGE, BERTScore), and rewrite the updated rows in place.

Backfill Judge Baselines
------------------------
Use the helper script to refresh existing scored files so the LLM judge stores gold-note baselines and AI–gold deltas:
```
uv run python scripts/backfill_judge_baseline.py --input data/augmented/train_scored.jsonl
uv run python scripts/backfill_judge_baseline.py --input data/augmented/bad_examples_scored.jsonl
```
When `--output` is omitted the script rewrites the input file in place; only the `llm_judge` payload is updated.

SummaC vs Gold Note
-------------------
Grab the transcript-vs-gold SummaC scores so you can eyeball how far the AI deviates from the clinician note:
```
uv run python scripts/compute_sumac_gold.py \
  --input data/augmented/train.jsonl \
  --output data/augmented/summac_gold.jsonl
```
The dashboard reads `summac_gold.jsonl` automatically and shows both the transcript-based SummaC and the gold comparison side by side.

Dashboard & Reporting
---------------------
Launch the Streamlit dashboard against any scored file:
```
uv run streamlit run dashboards/app.py
```
Features:
- Aggregated cards for SummaC, judge, ROUGE-L, and BERTScore.
- LLM-generated overview summary derived from the aggregate stats.
- “Show only flagged rows” toggle surfaces the subset with low ROUGE/BERTScore for rapid triage.
- Row explorer exposes filters, per-section judge breakdowns, and links to transcripts/gold notes.

Metrics Cheat Sheet
-------------------
- **SummaC (threshold 0.45)** – fast transcript vs. AI gate; below-threshold rows suggest hallucination risk or missing transcript coverage.
- **ROUGE-L (flag < 0.2)** – low lexical overlap with the gold note; often signals omitted sections or entirely different content.
- **BERTScore F1 (flag < 0.3)** – low semantic similarity to the gold note; catches paraphrased omissions that ROUGE might miss.
- **LLM Judge** – richest signal: factuality, completeness, coherence, fluency, issues, and gold-note coverage. Each row stores the clinician baseline (treated as the 5/5 reference) and the AI−gold delta so you can see how far the generated note trails the gold note.

Why these metrics (plain speak)
-------------------------------
- **ROUGE + BERTScore first.** They’re not the smartest graders, but they’re cheap and loud. If ROUGE and BERT both tank, something is seriously wrong (bad generation, wrong prompt, API hiccup) and we don’t waste LLM calls digging further. ROUGE looks at literal overlap, BERTScore checks the vibe/semantics, so together they catch both word-level and meaning-level misses.
- **SummaC on the transcript, then on the gold.** SummaC-ZS is lightweight compared to fancier NLI rigs like UniEval. I run it straight against the transcript to smoke-test hallucinations; the format mismatch means scores skew low, but that’s fine because we only care about relative changes. I also run a second pass against the gold note (see `summac_gold.jsonl`) so we can spot rows where the AI simply diverged from what the clinician wrote, even if it’s technically “supported” by the transcript.
- **LLM judge for nuance.** The judge is strictly transcript vs. AI. Gold serves as the 5/5 baseline so we can report “AI completeness is −1.2 vs gold” without mixing references in the prompt. The judge spits out missing/unsupported/clinical-error bullet points per section, which makes triage human-friendly when ROUGE/BERTScore are too blunt.
- **Synthetic “bad” set.** I generate five deliberately crummy SOAPs (minimal prompt, no safeguards) and park them in `data/augmented/bad_examples*.jsonl`. When the metrics light up red on that set I know the pipeline is still behaving; if they ever start giving those notes 4s and 5s we broke something.

Evaluator Quality & Synthetic Data
----------------------------------
- **Ground-truth sanity check:** Feed the gold SOAP back through the judge (treat gold as “AI output”) and confirm no issues are flagged. This validates the prompt and temperature settings.
- **Negative control:** Use `data/augmented/bad_examples.jsonl`, where AI SOAP sections are intentionally incorrect. SummaC/ROUGE/BERTScore should all tank, and the judge should list substantive issues. Re-run this file after prompt or model changes.
- **Targeted mutations:** For ad-hoc QA, modify a single section (e.g., replace Assessment with “Patient is stable”) and confirm both deterministic metrics and the judge call out the omission.
- **Discrepancy tracking:** If deterministic metrics flag a row that the judge says is fine, log the case and manually inspect—those become regression tests for future prompt iterations.

Follow this loop whenever we update prompts, swap generators, or onboard a new evaluator model to maintain confidence in the scoring stack.
