#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$PACKAGE_DIR")"
STAGE="${1:-run}"
GPU_ID="${GPU_ID:-4}"
SESSION_NAME="${SESSION_NAME:-meld7r1500}"
if [[ ! "$SESSION_NAME" =~ ^[a-zA-Z0-9_-]+$ || ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo 'Use a simple session name and one integer GPU_ID.' >&2; exit 2
fi
if ! command -v tmux >/dev/null; then echo 'tmux is not installed.' >&2; exit 1; fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session already exists: $SESSION_NAME. Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi
LOG_DIR="$PROJECT_DIR/artifacts/logs"; mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_${STAMP}.log"
STATUS_FILE="$LOG_DIR/${SESSION_NAME}_${STAMP}.status"
printf -v TASK 'cd %q && exec env GPU_ID=%q LOG_FILE=%q STATUS_FILE=%q bash %q %q' \
  "$PROJECT_DIR" "$GPU_ID" "$LOG_FILE" "$STATUS_FILE" "$PACKAGE_DIR/run.sh" "$STAGE"
printf -v SETUP 'tmux set-window-option -t %q remain-on-exit off; %s' "$SESSION_NAME:" "$TASK"
tmux new-session -d -s "$SESSION_NAME" "$SETUP"
printf 'Started session: %s\nLog: %s\nExit status: %s\nAttach: tmux attach -t %s\n' \
  "$SESSION_NAME" "$LOG_FILE" "$STATUS_FILE" "$SESSION_NAME"
echo 'The session exits when the task ends. Logs and exit status remain on disk.'
