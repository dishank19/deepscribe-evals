"""Run a set of metrics across an augmented dataset split."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Iterable, Iterator, List

from tqdm import tqdm

from evalsuite.dataset import read_jsonl, write_jsonl
from evalsuite.metrics import llm_judge, summac_eval  # noqa: F401 - ensure registration
from evalsuite.metrics.registry import get_metric


def default_split_path(split: str) -> Path:
    """Location of the augmented split file."""

    return Path(f"data/augmented/{split}.jsonl")


def load_records(path: Path, start: int = 0, limit: int | None = None) -> Iterator[dict]:
    """Stream records from a JSONL file with an optional start offset."""

    yielded = 0
    for idx, record in enumerate(read_jsonl(path)):
        if idx < start:
            continue
        if limit is not None and yielded >= limit:
            break
        yielded += 1
        yield record


def run_metrics(records: Iterable[dict], metric_names: List[str]) -> Iterator[dict]:
    """Yield records with metrics populated."""

    for record in tqdm(records, desc="Evaluating", unit="row", leave=False):
        metrics = dict(record.get("metrics", {}))
        for name in metric_names:
            metric_fn = get_metric(name)
            params = inspect.signature(metric_fn).parameters
            if len(params) >= 3:
                metrics[name] = metric_fn(record["transcript"], record["ai_soap"], record.get("gold_soap"))
            else:
                metrics[name] = metric_fn(record["transcript"], record["ai_soap"])
        record["metrics"] = metrics
        yield record


def emit_records(records: Iterable[dict], path: Path | None, append: bool = False) -> None:
    """Write evaluated records to disk or stdout."""

    if path is None:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
        return

    mode = "a" if append else "w"
    write_jsonl(records, path, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metrics across a batch of SOAP rows.")
    parser.add_argument("--split", default="train", help="Augmented split to read (default: train)")
    parser.add_argument("--metrics", default="summac", help="Comma-separated metrics to run")
    parser.add_argument("--start", type=int, default=0, help="Starting index (0-based) within the dataset")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of records")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path; prints JSONL to stdout when omitted",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = default_split_path(args.split)
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    records = load_records(input_path, start=args.start, limit=args.limit)
    evaluated = list(run_metrics(records, metric_names))
    output_path = Path(args.output) if args.output else None
    emit_records(evaluated, output_path, append=args.append)


if __name__ == "__main__":
    main()
