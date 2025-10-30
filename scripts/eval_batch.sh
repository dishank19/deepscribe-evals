#!/usr/bin/env bash

set -euo pipefail

SPLIT=${1:-train}
METRICS=${2:-summac}
LIMIT=${3:-}

CMD=(uv run --timeout 600 python -m evalsuite.runners.run_batch --split "${SPLIT}" --metrics "${METRICS}")

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}"
