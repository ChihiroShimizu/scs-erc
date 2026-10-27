#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$PACKAGE_DIR")"
cd "$PROJECT_DIR"
PY="${S7R_PYTHON:-$PROJECT_DIR/.venv-server/bin/python3}"
STAGE="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
GPU_ID="${GPU_ID:-4}"
if [[ ! -x "$PY" ]]; then
  echo "Missing Python: $PY. Use the existing .venv-server environment." >&2
  exit 1
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo 'GPU_ID must be one physical GPU index.' >&2
  exit 2
fi
LOG_DIR="$PROJECT_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/meld7r1500_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log}"
STATUS_FILE="${STATUS_FILE:-${LOG_FILE%.log}.status}"
mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATUS_FILE")"
case "$STAGE" in
  run|smoke|train-seed|evaluate-seed)
    ln -sfn "$LOG_FILE" "$LOG_DIR/latest_meld7r1500.log"
    ln -sfn "$STATUS_FILE" "$LOG_DIR/latest_meld7r1500.status"
    ;;
esac
finish() {
  local code=$?
  printf '%s\n' "$code" > "$STATUS_FILE"
}
trap finish EXIT
printf 'Stage=%s GPU_ID=%s\nLog: %s\nExit status: %s\n' "$STAGE" "$GPU_ID" "$LOG_FILE" "$STATUS_FILE"
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -u "$PACKAGE_DIR/run_stage7r.py" "$STAGE" "$@" 2>&1 | tee "$LOG_FILE"
codes=("${PIPESTATUS[@]}")
code="${codes[0]}"
if [[ "$code" -eq 0 && "${codes[1]}" -ne 0 ]]; then code="${codes[1]}"; fi
exit "$code"
