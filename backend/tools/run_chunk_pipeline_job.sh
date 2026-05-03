#!/bin/zsh
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_chunk_pipeline_job.sh <parent_project_id> <log_path>" >&2
  exit 64
fi

PARENT_PROJECT_ID="$1"
LOG_PATH="$2"

ROOT="/Users/jamieobala/Documents/Panelia"
PYTHON_BIN="$ROOT/backend/.venv/bin/python"
RUNNER="$ROOT/backend/tools/run_chunk_pipeline.py"

mkdir -p "$(dirname "$LOG_PATH")"

export HOME="/Users/jamieobala"
export PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPATH="$ROOT/backend"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting parent $PARENT_PROJECT_ID"
  cd "$ROOT"
  exec "$PYTHON_BIN" "$RUNNER" --manifest-parent "$PARENT_PROJECT_ID"
} >>"$LOG_PATH" 2>&1
