from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from project root .env if present.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_DOTENV_PATH, override=False)
