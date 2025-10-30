"""Generate intentionally poor AI SOAP notes for evaluator stress testing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evalsuite.dataset import parse_soap
from evalsuite.gen.pydantic_claude import SoapNote
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel


def build_bad_rows(start: int, count: int, split: str) -> Iterable[dict]:
    dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary", split=split)
    agent = Agent(
        model=AnthropicModel("claude-haiku-4-5-20251001"),
        output_type=SoapNote,
        system_prompt=(
            "Write a concise SOAP note (one or two sentences per section) using only facts from the transcript. "
            "Return JSON with keys S, O, A, P."
        ),
    )
    for offset in range(start, start + count):
        record = dataset[int(offset)]
        transcript = record["dialogue"]
        ai_note = agent.run_sync(transcript).output.model_dump()
        yield {
            "id": f"bad-{offset}",
            "transcript": record["dialogue"],
            "gold_soap": parse_soap(record["soap"]),
            "ai_soap": ai_note,
            "metrics": {},
            "meta": {"source_index": int(offset), "type": "synthetic_bad"},
        }


def main() -> None:
    output = Path("data/augmented/bad_examples.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(build_bad_rows(start=200, count=5, split="train"))
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    print(f"Wrote {len(rows)} synthetic rows to {output}")


if __name__ == "__main__":
    main()
