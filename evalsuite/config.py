"""Configuration settings for the SOAP evaluation suite."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used across the suite."""

    data_root: Path = Path("data")
    augmented_root: Path = Path("data/augmented")
    cache_root: Path = Path(".cache/evalsuite")


@dataclass(frozen=True)
class SummaCConfig:
    """Parameters for the SummaC fast factuality gate."""

    model_name: str = "mnli-base"
    granularity: str = "sentence"
    min_overall: float = 0.6
    min_section: float = 0.55


@dataclass(frozen=True)
class JudgeConfig:
    """Settings for the LLM-as-a-judge pipeline."""

    model: str = "cerebras:gpt-oss-120b"
    prompt_version: str = "v0"
    enable_section_cache: bool = True
    max_retries: int = 3
    timeout_seconds: float = 240.0


paths = Paths()
summac = SummaCConfig()
judge = JudgeConfig()

DEFAULT_BUILD_LIMIT = 100
