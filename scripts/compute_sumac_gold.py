"""Compute SummaC scores using the clinician gold SOAP as reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ensure project modules importable
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evalsuite.metrics.summac_eval import run_summac_against_reference
from evalsuite.dataset import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SummaC vs gold SOAP and write JSONL output.")
    parser.add_argument(
        "--input",
        default="data/augmented/train.jsonl",
        help="Augmented dataset (with ai_soap + gold_soap) to read.",
    )
    parser.add_argument(
        "--output",
        default="data/augmented/summac_gold.jsonl",
        help="Destination JSONL file for SummaC vs gold scores.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Optional offset for large files (0-based).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on rows processed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, record in enumerate(read_jsonl(input_path)):
            if idx < args.start:
                continue
            if args.limit is not None and (idx - args.start) >= args.limit:
                break
            gold = record.get("gold_soap") or {}
            ai = record.get("ai_soap") or {}
            result = run_summac_against_reference(gold, ai)
            payload = {
                "id": record.get("id", idx),
                "sumac_gold": result["scores"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
if __name__ == "__main__":
    main()
