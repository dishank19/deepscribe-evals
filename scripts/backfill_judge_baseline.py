"""Refresh llm_judge metrics to include gold baselines and deltas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evalsuite.metrics.llm_judge import run_llm_judge


def backfill(input_path: Path, output_path: Path) -> None:
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.setdefault("metrics", {})
            metrics["llm_judge"] = run_llm_judge(
                record.get("transcript", ""),
                record.get("ai_soap", {}),
                record.get("gold_soap"),
            )
            rows.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill llm_judge baseline metrics into a scored JSONL file.")
    parser.add_argument("--input", required=True, help="Path to the scored JSONL file to refresh.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to rewriting the input file in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    if output_path == input_path:
        temp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        backfill(input_path, temp_path)
        temp_path.replace(input_path)
    else:
        backfill(input_path, output_path)


if __name__ == "__main__":
    main()
