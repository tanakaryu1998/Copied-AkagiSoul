#!/bin/zsh
cd "$(dirname "$0")"

PYTHON_BIN="./runtime/python/aarch64-apple-darwin/bin/python3"
TRAINER_SCRIPT="./auto_learn/trainer.py"

(
  LAST_COUNT=$(find ./history/games -type f 2>/dev/null | wc -l)
  while kill -0 $$ 2>/dev/null; do
    sleep 15
    CURRENT_COUNT=$(find ./history/games -type f 2>/dev/null | wc -l)
    if [ "$CURRENT_COUNT" -gt "$LAST_COUNT" ]; then
      LAST_COUNT=$CURRENT_COUNT
      "$PYTHON_BIN" "$TRAINER_SCRIPT" &
    fi
  done
) &
WATCHER_PID=$!

trap "kill $WATCHER_PID 2>/dev/null" EXIT

./akagi "$@"