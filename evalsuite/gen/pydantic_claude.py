"""Generate SOAP notes using Claude with Pydantic AI structured output."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from pydantic import BaseModel, Field
from pydantic_ai import Agent, exceptions
from pydantic_ai.models.anthropic import AnthropicModel
from time import sleep


class SoapNote(BaseModel):
    """Structured SOAP note returned by the generator."""

    S: str = Field(description="Subjective findings")
    O: str = Field(description="Objective findings")
    A: str = Field(description="Assessment/diagnosis")
    P: str = Field(description="Plan or follow-up actions")


SYSTEM_PROMPT = """
You are an experienced clinical documentation specialist.
- Read the provided patient encounter transcript carefully.
- Extract only information stated or implied in the transcript; do not invent facts.
- Write a SOAP note with four sections: S, O, A, P.
- Each section should be well-structured prose suitable for a clinician.
- Return JSON that exactly matches the provided schema.
- Every key (S, O, A, P) must be present even if no findings are noted.
"""

USER_TEMPLATE = """
Transcript:
\"\"\"{transcript}\"\"\"

Respond ONLY with JSON shaped as:
{{
  "S": "...",
  "O": "...",
  "A": "...",
  "P": "..."
}}
"""



def _soap_agent() -> Agent:
    """Create a single Claude agent reused across calls."""

    model = AnthropicModel("claude-haiku-4-5-20251001")
    return Agent(
        model=model,
        output_type=SoapNote,
        system_prompt=SYSTEM_PROMPT.strip(),
        retries=3,
        output_retries=3,
    )


def generate_ai_soap(transcript: str) -> Dict[str, str]:
    """Generate a SOAP note for the supplied transcript."""

    agent = _soap_agent()
    prompt = USER_TEMPLATE.format(transcript=transcript.strip())
    for attempt in range(3):
        try:
            result = agent.run_sync(prompt)
            return result.output.model_dump()
        except exceptions.UnexpectedModelBehavior as error:
            if attempt == 2:
                raise
            sleep(1.5)
