#!/bin/zsh
cd "$(dirname "$0")"
mkdir -p auto_learn history/virtual_games
PID_FILE="auto_learn/self_play.pid"
NEED_START=1
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        NEED_START=0
    fi
fi
if [ "$NEED_START" -eq 1 ]; then
    VENV_PY="./mjai_bot/local-adaptive/.venv/bin/python"
    if [ ! -x "$VENV_PY" ]; then
        VENV_PY="./runtime/python/aarch64-apple-darwin/bin/python3"
    fi
    nohup nice -n 15 "$VENV_PY" auto_learn/self_play.py > auto_learn/self_play.log 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    echo "仮想自己対戦学習をバックグラウンドで開始しました (PID: $NEW_PID)"
    echo "実戦の動作に影響しない低優先度モードで自動対局と学習を回します。"
fi
