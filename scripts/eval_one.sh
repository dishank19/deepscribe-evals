#!/usr/bin/env bash

set -euo pipefail

SPLIT=${1:-train}
ROW_ID=${2:-0}
METRICS=${3:-summac}

uv run --timeout 600 python -m evalsuite.runners.run_one --split "${SPLIT}" --id "${ROW_ID}" --metrics "${METRICS}"
