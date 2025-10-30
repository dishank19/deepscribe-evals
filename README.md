DeepScribe SOAP Evaluation Suite
================================

Quick-start guide for building the augmented dataset, running the evaluation metrics, and inspecting results.

Environment
-----------
- Install dependencies once with `uv sync`.
- Activate the managed venv if you need direct access: `source .venv/bin/activate`.
- All project commands should be wrapped with `uv run …` so they execute inside the synced environment.

Build The Augmented Dataset
---------------------------
1. Create SOAP generations (defaults to 100 train rows):
   ```
   uv run python scripts/build_dataset.py --split train --limit 100
   ```
   Arguments:
   - `--split`: HF split name (`train`, `validation`, `test`).
   - `--limit`: optional cap for quicker runs. Omit to process the full split.
   - Output: `data/augmented/<split>.jsonl` with transcript, gold SOAP, and AI SOAP sections.

2. Inspect a sample row:
   ```
   uv run python view_dataset.py --split train --index 0
   ```

Run Metrics
-----------
### Batch Runner
Evaluate any chunk of the augmented dataset.
```
uv run python -m evalsuite.runners.run_batch \
  --split train \
  --metrics summac,llm_judge \
  --output data/augmented/train_scored.jsonl
```
Key arguments:
- `--split`: which augmented file to read (looks under `data/augmented/<split>.jsonl`).
- `--metrics`: comma-separated registry names (`summac`, `llm_judge`).
- `--start`, `--limit`: optional slice controls for chunked execution.
- `--output`: destination file. Use `.jsonl` for newline-delimited records or `.json` for a single array.
- `--append`: only valid with `.jsonl`; appends new results instead of overwriting.

### Single Row Runner
Re-run metrics for a specific record id (0-indexed when generated locally):
```
uv run python -m evalsuite.runners.run_one \
  --id 42 \
  --split train \
  --metrics summac,llm_judge \
  --output data/augmented/single_row.json
```
Omit `--output` to print the evaluated record to stdout (gold SOAP is hidden by default).

Visualise Results
-----------------
Launch the Streamlit dashboard against any scored file:
```
uv run streamlit run dashboards/app.py
```
Choose the JSONL from the sidebar to see overview stats and drill into individual rows.

Outputs
-------
- Generated dataset: `data/augmented/<split>.jsonl`
- Scored outputs: `data/augmented/<split>_scored.jsonl` (or `.json`)
- Dashboard config expects scored files under `data/augmented/`.

Next Steps
----------
- Periodically sample the scored file to confirm the LLM judge still returns valid payloads after prompt/model tweaks.
- Add regression tests for dataset parsing and metric aggregation.
- Write the evaluation rationale tying SummaC + LLM judge back to the project goals.
