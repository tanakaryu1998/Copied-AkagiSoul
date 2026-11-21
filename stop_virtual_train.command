#!/bin/zsh
cd "$(dirname "$0")"
PID_FILE="auto_learn/self_play.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        pkill -P "$PID" 2>/dev/null
        kill "$PID" 2>/dev/null
        rm -f "$PID_FILE"
        echo "並列仮想自己対戦学習をすべて停止しました (親PID: $PID)"
    else
        pkill -f "auto_learn/self_play.py" 2>/dev/null
        rm -f "$PID_FILE"
        echo "仮想自己対戦学習はすでに停止しています。"
    fi
else
    pkill -f "auto_learn/self_play.py" 2>/dev/null
    echo "稼働中の仮想自己対戦プロセスを終了しました。"
fi
