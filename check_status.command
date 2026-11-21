#!/bin/zsh
cd "$(dirname "$0")"
clear
echo "=================================================="
echo "   Akagi 自律学習AI ＆ 実戦・並列仮想戦績モニター"
echo "=================================================="
./runtime/python/aarch64-apple-darwin/bin/python3 -c '
import glob, json, os, subprocess
from datetime import datetime

pid_file = "auto_learn/self_play.pid"
virt_running = False
worker_cnt = 0
if os.path.exists(pid_file):
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, 0)
        virt_running = True
        out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True).strip()
        worker_cnt = len(out.splitlines()) if out else 0
    except Exception:
        pass

v_files_all = glob.glob("history/virtual_games/virtual_*.jsonl")
v4_files = [f for f in v_files_all if "virtual_4p_" in os.path.basename(f)]
v3_files = [f for f in v_files_all if "virtual_3p_" in os.path.basename(f)]

latest_v_str = "なし"
if v_files_all:
    latest_mtime = max(os.path.getmtime(f) for f in v_files_all)
    latest_v_str = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")

if virt_running:
    print(f"■ 裏側並列仮想対局 : 【稼働中】({worker_cnt}卓並行で自動対局中)")
else:
    print("■ 裏側並列仮想対局 : 【停止中】start_akagi.command 起動で自動開始")
print(f"  仮想試合累計実行 : 計 {len(v_files_all)} 半荘 (4麻: {len(v4_files)}戦 / 3麻: {len(v3_files)}戦)")
print(f"  最新仮想試合完了 : {latest_v_str}")

state_file = "auto_learn/train_state.json"
if not os.path.exists(state_file):
    print("\nまだ学習データがありません。")
else:
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for mode, label, goal, v_cur in [
        ("4p", "四人麻雀 (4P)", 2.50, len(v4_files)),
        ("3p", "三人麻雀 (3P)", 2.00, len(v3_files))
    ]:
        info = data.get(mode, {})
        print(f"\n■ {label}")
        if not info:
            print("  状態       : 対局データ待機中")
            continue
        files = info.get("file_count", 0)
        v_learned = info.get("virtual_count", 0)
        v_pending = max(0, v_cur - v_learned)
        samples = info.get("val_samples", 0)
        promoted = info.get("promoted", False)
        status_str = "【本番稼働中】純粋ResNet転移学習AI" if promoted else "【育成中】純正AI稼働・実戦100戦まで裏で学習中"
        print(f"  稼働状態   : {status_str}")
        print(f"  実戦半荘数 : {files} / 100 戦 (運用開始条件)")
        print(f"  仮想試合数 : 実行済 {v_cur} 戦 (学習吸収済: {v_learned} 戦 / 次回学習待機: {v_pending} 戦)")
        print(f"  検証打牌数 : {samples} / 1500 打牌 (昇格基準)")
        perf = info.get("performance", {})
        if perf:
            print(f"  実戦平均順 : {perf.get("avg_rank", 0):.2f} 位 (目標: {goal:.2f} 未満)")
            print(f"  トップ/ラス: トップ率 {perf.get("top_rate", 0)*100:.1f}% / ラス率 {perf.get("last_rate", 0)*100:.1f}%")
            print(f"  和了/放銃  : 和了率 {perf.get("win_rate", 0)*100:.1f}% / 放銃率 {perf.get("deal_in_rate", 0)*100:.1f}%")
        if "last_mean_acc" in info:
            print(f"  打牌一致率 : {info["last_mean_acc"]*100:.2f}% (目標基準: 72.00%)")
            print(f"  手牌読み率 : {info.get("opp_read_acc", 0)*100:.2f}% (他家手牌・当たり牌予測)")
            print(f"  連続合格数 : {info.get("streak", 0)} / 2 回連続達成")
        if "updated_at" in info:
            print(f"  最終学習時 : {info["updated_at"][:19].replace("T", " ")}")
'
echo ""
echo "=================================================="
echo "確認が終わったらこのウィンドウを閉じてください。"
