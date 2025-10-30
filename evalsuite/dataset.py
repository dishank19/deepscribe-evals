"""
Helper functions for working with the SOAP evaluation dataset.

Keeps the data flow lightweight: load from Hugging Face, derive JSON rows, and
read/write JSONL files without additional abstractions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from datasets import load_dataset

SECTION_KEYS = ("S", "O", "A", "P")


def load_split(split: str) -> Iterable[dict]:
    """Yield records from the Hugging Face dataset split."""

    dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary", split=split)
    for record in dataset:
        yield record


def parse_soap(raw: str) -> Dict[str, str]:
    """Convert a SOAP note blob into a dict keyed by S/O/A/P."""

    sections: Dict[str, List[str]] = {key: [] for key in SECTION_KEYS}
    current: Optional[str] = None

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        prefix = stripped[:2].upper()
        if prefix in {f"{key}:" for key in SECTION_KEYS}:
            current = prefix[0]
            content = stripped[2:].strip()
            if content:
                sections[current].append(content)
            continue

        if current:
            sections[current].append(stripped)

    return {key: "\n".join(parts).strip() for key, parts in sections.items()}


def build_augmented_rows(
    split: str,
    generate_fn,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    """Yield rows containing transcript, gold SOAP, and generated SOAP."""

    for idx, record in enumerate(load_split(split)):
        if limit is not None and idx >= limit:
            break

        transcript = record["dialogue"]
        gold = parse_soap(record["soap"])
        ai = generate_fn(transcript)
        yield {
            "id": str(record.get("id", idx)),
            "transcript": transcript,
            "gold_soap": gold,
            "ai_soap": ai,
            "metrics": {},
            "meta": {},
        }


def read_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSON objects from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def find_row(path: Path, row_id: str) -> dict:
    """Return a row from path matching the requested id."""

    for record in read_jsonl(path):
        if str(record.get("id")) == row_id:
            return record
    raise KeyError(f"Row '{row_id}' not found in {path}")


def write_jsonl(records: Iterable[dict], path: Path, mode: str = "w") -> None:
    """Write JSON objects to path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
