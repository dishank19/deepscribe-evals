"""Command-line entry point to run metrics on a single row."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Iterable

from evalsuite.dataset import find_row, write_jsonl
from evalsuite.metrics import bert_score_eval, llm_judge, rouge_eval, summac_eval  # noqa: F401 - ensure registration
from evalsuite.metrics.registry import get_metric


def default_split_path(split: str) -> Path:
    """Location of the augmented split file."""

    return Path(f"data/augmented/{split}.jsonl")


def run_metrics(record: dict, metric_names: Iterable[str]) -> dict:
    """Run the selected metrics against the provided record."""

    metrics = dict(record.get("metrics", {}))
    for name in metric_names:
        metric_fn = get_metric(name)
        params = inspect.signature(metric_fn).parameters
        if len(params) >= 3:
            metrics[name] = metric_fn(record["transcript"], record["ai_soap"], record.get("gold_soap"))
        else:
            metrics[name] = metric_fn(record["transcript"], record["ai_soap"])

    record["metrics"] = metrics
    return record


def output_record(record: dict, path: Path | None) -> None:
    """Persist or print the record."""

    if path is None:
        display = dict(record)
        display.pop("gold_soap", None)
        print(json.dumps(display, ensure_ascii=False, indent=2))
        return

    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return

    write_jsonl([record], path, mode="a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metrics for a single SOAP row.")
    parser.add_argument("--id", required=True, help="Row identifier")
    parser.add_argument("--split", default="train", help="Augmented split file (default: train)")
    parser.add_argument("--metrics", default="summac", help="Comma-separated list of metrics to run")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for evaluated output; prints to stdout when omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = default_split_path(args.split)
    record = find_row(input_path, args.id)
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    updated = run_metrics(record, metric_names)
    output_path = Path(args.output) if args.output else None
    output_record(updated, output_path)


if __name__ == "__main__":
    main()
