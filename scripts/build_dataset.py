"""CLI script to build an augmented dataset with AI-generated SOAP notes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evalsuite.config import DEFAULT_BUILD_LIMIT  # noqa: E402
from evalsuite.dataset import build_augmented_rows, write_jsonl  # noqa: E402
from evalsuite.gen.pydantic_claude import generate_ai_soap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build augmented SOAP dataset.")
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument(
        "--output",
        default="data/augmented/train.jsonl",
        help="Destination path for augmented rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    limit = args.limit if args.limit is not None else DEFAULT_BUILD_LIMIT
    rows = build_augmented_rows(args.split, generate_ai_soap, limit=limit)
    write_jsonl(rows, output_path)


if __name__ == "__main__":
    main()
