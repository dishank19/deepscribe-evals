# Repository Guidelines

## Project Structure & Module Organization
Runtime scripts live at the top level. `main.py` is the minimal entry point; extend it or add reusable code under a `deepscribe/` package so interfaces stay clean. `view_dataset.py` contains dataset inspection helpers for `omi-health/medical-dialogue-to-soap-summary`; keep additional dataset utilities alongside it. Project metadata and dependencies are tracked in `pyproject.toml` with the locked resolver state in `uv.lock`. Reserve `README.md` for user-facing notes.

## Build, Test, and Development Commands
Create the environment with `uv sync`, which installs the dependencies declared in `pyproject.toml`. Use `uv run python main.py` for a smoke check of the entry point. Explore dataset samples via `uv run python view_dataset.py --split train --index 0`. Always prefer `uv run` so scripts execute inside the managed environment.

## Coding Style & Naming Conventions
Target Python 3.12 features and adhere to PEP 8: 4-space indentation, snake_case for functions and variables, CapWords for classes, and UPPER_CASE for constants like `SOAP_LABELS`. Keep modules small and cohesive; share helpers instead of duplicating parsing logic. Store large assets or notebooks under dedicated `docs/` or `data/` directories and exclude transient artifacts from version control.

## Testing Guidelines
Tests are not yet in place; add them with `pytest`. Mirror the package layout under `tests/`, e.g., `tests/test_view_dataset.py` for `view_dataset.py`. Run suites using `uv run pytest`. Name tests by behavior (`test_parse_soap_sections_handles_missing_labels`) and cover parsing edge cases, dataset boundary checks, and failure messaging.

## Commit & Pull Request Guidelines
Write commits in the imperative mood with focused scopes (`Add SOAP parsing helpers`). Expand in the body when behavior changes or new dependencies are introduced. Pull requests should state motivation, summarize functional impact, and link related issues. Attach relevant command output or dataset snippets so reviewers can reproduce and verify your results.

---

# Evaluation Suite Notes

## Focus Areas
- Extend each OMI transcript with an AI-generated SOAP note using Claude plus structured output guarantees.
- Use SummaC scores as an inexpensive factuality signal ahead of deeper checks.
- Layer an LLM judge for richer scoring and issue extraction once the fast gate is in place.
- Emit JSONL so dashboards and analytics can plug in later without rework.

## Current Scaffold
- `evalsuite/config.py`, `evalsuite/dataset.py`, `evalsuite/gen/pydantic_claude.py`, `evalsuite/metrics/`, and `evalsuite/runners/` hold the callable pieces.
- CLI wrappers (`scripts/build_dataset.py`, `scripts/eval_one.sh`, `scripts/eval_batch.sh`) exercise the flow; `data/augmented/` houses outputs.
- Metrics registry currently wires stubbed `summac` and `llm_judge` implementations; swap them with real integrations as they come online.

## Upcoming Tasks
- Evaluate whether additional deterministic metrics (coverage, lexical overlap) add value alongside SummaC + judge.
- Optional: add a lightweight deterministic check (coverage or lexical overlap) and start building reporting notebooks/dashboards from the JSONL outputs.

## Dataset Handling Notes
- `scripts/build_dataset.py` now defaults to writing the first 100 transcripts; override `--limit` to adjust. Outputs land in `data/augmented/{split}.jsonl`.
- Evaluation focuses on transcript ↔ gold SOAP ↔ AI SOAP comparisons, so split boundaries are less critical. Keep the subset consistent for reproducibility.
- Generator now retries up to three times when Claude misses required fields; transient API overloads may require rerunning the build command.

## Metric Status
- SummaC metric now uses an in-repo copy of the zero-shot scorer (transformers + NLI). Default backbone is `microsoft/deberta-base-mnli`; threshold flags surface under `issues["flags"]`.
- LLM judge uses `pydantic-evals` with concise rubrics (top-3 missing / top-2 unsupported / top-1 clinical error), runs section evaluations in parallel, enforces a configurable timeout (`config.judge.timeout_seconds`), and optionally reports gold-note coverage.
