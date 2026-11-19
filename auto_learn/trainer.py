import fcntl
import glob
import json
import os
import re
import shutil
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_core import TILE_TO_IDX, MahjongMultiTaskNet, build_feature_vector, normalize_tile

MIN_VAL_SAMPLES = 1500
PROMOTION_THRESHOLD = 0.78
IMPROVEMENT_MARGIN = 0.005
REQUIRED_STREAK = 2

def parse_game_file(filepath):
    samples_x = []
    samples_y_policy = []
    samples_y_danger = []

    hands = [[], [], [], []]
    my_discards = [[0] * 34 for _ in range(4)]
    all_discards = [0] * 34
    tedashi_by_seat = [[0] * 34 for _ in range(4)]
    melds_by_seat = [[0] * 34 for _ in range(4)]
    dora_vec = [0.0] * 34
    riichi_flags = [0.0] * 4
    kita_counts = [0] * 4
    turn_counts = [0] * 4
    last_discard_idx = -1
    bakaze = 27
    oya = 0
    num_players = 4

    kyoku_buffer = []
    danger_tiles_in_kyoku = set()

    def flush_kyoku():
        if not kyoku_buffer:
            return
        danger_vec = [1.0 if i in danger_tiles_in_kyoku else 0.0 for i in range(34)]
        for bx, by in kyoku_buffer:
            samples_x.append(bx)
            samples_y_policy.append(by)
            samples_y_danger.append(danger_vec)
        kyoku_buffer.clear()
        danger_tiles_in_kyoku.clear()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")
            if etype == "start_kyoku":
                flush_kyoku()
                my_discards = [[0] * 34 for _ in range(4)]
                all_discards = [0] * 34
                tedashi_by_seat = [[0] * 34 for _ in range(4)]
                melds_by_seat = [[0] * 34 for _ in range(4)]
                dora_vec = [0.0] * 34
                riichi_flags = [0.0] * 4
                kita_counts = [0] * 4
                turn_counts = [0] * 4
                last_discard_idx = -1
                bakaze = TILE_TO_IDX.get(ev.get("bakaze", "E"), 27)
                oya = ev.get("oya", 0)
                dora_marker = ev.get("dora_marker")
                if dora_marker:
                    b = normalize_tile(dora_marker)
                    if b in TILE_TO_IDX:
                        dora_vec[TILE_TO_IDX[b]] = 1.0
                tehais = ev.get("tehais", [])
                non_empty = [t for t in tehais if len(t) > 0]
                if len(non_empty) == 3 or len(tehais) == 3:
                    num_players = 3
                else:
                    num_players = 4
                for i in range(min(4, len(tehais))):
                    hands[i] = list(tehais[i])
            elif etype == "tsumo":
                actor = ev.get("actor")
                pai = ev.get("pai")
                if actor is not None and 0 <= actor < 4 and pai:
                    turn_counts[actor] += 1
                    hands[actor].append(pai)
            elif etype == "dahai":
                actor = ev.get("actor")
                pai = ev.get("pai")
                if actor is not None and 0 <= actor < 4 and pai:
                    base = normalize_tile(pai)
                    target_idx = TILE_TO_IDX.get(base, -1)
                    if target_idx >= 0 and len(hands[actor]) >= 13:
                        opp_tedashi = [0] * 34
                        opp_melds = [0] * 34
                        for s in range(num_players):
                            if s != actor:
                                for k in range(34):
                                    opp_tedashi[k] += tedashi_by_seat[s][k]
                                    opp_melds[k] += melds_by_seat[s][k]
                        jikaze = 27 + ((actor - oya) % num_players)
                        vec = build_feature_vector(
                            hands[actor],
                            my_discards[actor],
                            all_discards,
                            opp_tedashi,
                            opp_melds,
                            dora_vec,
                            last_discard_idx,
                            bakaze,
                            jikaze,
                            turn_counts[actor],
                            riichi_flags,
                            kita_counts[:3],
                            num_players
                        )
                        kyoku_buffer.append((vec, target_idx))
                    if target_idx >= 0:
                        last_discard_idx = target_idx
                        all_discards[target_idx] += 1
                        my_discards[actor][target_idx] += 1
                        if not ev.get("tsumogiri", False):
                            tedashi_by_seat[actor][target_idx] += 1
                    if pai in hands[actor]:
                        hands[actor].remove(pai)
            elif etype == "reach":
                actor = ev.get("actor")
                if actor is not None and 0 <= actor < 4:
                    riichi_flags[actor] = 1.0
            elif etype == "nukidora":
                actor = ev.get("actor")
                if actor is not None and 0 <= actor < 4:
                    kita_counts[actor] += 1
                    if "N" in hands[actor]:
                        hands[actor].remove("N")
            elif etype in ("pon", "chi", "kan", "daiminkan", "kakan", "ankan"):
                actor = ev.get("actor")
                if actor is not None and 0 <= actor < 4:
                    pai = ev.get("pai")
                    if pai:
                        idx = TILE_TO_IDX.get(normalize_tile(pai), -1)
                        if idx >= 0:
                            melds_by_seat[actor][idx] += 1
                    for c in ev.get("consumed", []):
                        c_idx = TILE_TO_IDX.get(normalize_tile(c), -1)
                        if c_idx >= 0:
                            melds_by_seat[actor][c_idx] += 1
                        if c in hands[actor]:
                            hands[actor].remove(c)
            elif etype == "hora":
                pai = ev.get("pai")
                if pai:
                    idx = TILE_TO_IDX.get(normalize_tile(pai), -1)
                    if idx >= 0:
                        danger_tiles_in_kyoku.add(idx)
                for th in ev.get("tehais", []):
                    for t in th:
                        idx = TILE_TO_IDX.get(normalize_tile(t), -1)
                        if idx >= 0:
                            danger_tiles_in_kyoku.add(idx)
            elif etype == "ryukyoku":
                for th in ev.get("tehais", []):
                    for t in th:
                        idx = TILE_TO_IDX.get(normalize_tile(t), -1)
                        if idx >= 0:
                            danger_tiles_in_kyoku.add(idx)
                flush_kyoku()
            elif etype == "end_kyoku":
                flush_kyoku()

    flush_kyoku()
    mode = "3p" if num_players == 3 else "4p"
    return mode, samples_x, samples_y_policy, samples_y_danger

def list_history_files(history_dir):
    patterns = [
        os.path.join(history_dir, "**", "*.jsonl"),
        os.path.join(history_dir, "**", "*.json"),
        os.path.join(history_dir, "**", "*.mjson")
    ]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))
    return sorted(files)

def evaluate_blocks(model, val_x, val_yp, val_yd, device):
    model.eval()
    total_len = len(val_x)
    if total_len < MIN_VAL_SAMPLES:
        return 0.0, 0.0, 0.0
    block_size = total_len // 3
    block_accs = []
    danger_accs = []
    with torch.no_grad():
        for b in range(3):
            start = b * block_size
            end = total_len if b == 2 else (b + 1) * block_size
            bx = torch.tensor(val_x[start:end], dtype=torch.float32, device=device)
            byp = torch.tensor(val_yp[start:end], dtype=torch.long, device=device)
            byd = torch.tensor(val_yd[start:end], dtype=torch.float32, device=device)
            p_logits, d_logits = model(bx)
            preds = p_logits.argmax(dim=1)
            acc = (preds == byp).sum().item() / byp.size(0)
            block_accs.append(acc)
            d_preds = (torch.sigmoid(d_logits) >= 0.5).float()
            d_acc = (d_preds == byd).float().mean().item()
            danger_accs.append(d_acc)
    return sum(block_accs) / 3.0, min(block_accs), sum(danger_accs) / 3.0

def promote_bot_in_config(config_path, mode):
    if not os.path.exists(config_path):
        return
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    if mode == "4p":
        updated = re.sub(r'active_4p\s*=\s*"akagi-native"', 'active_4p = "local-adaptive"', content)
    else:
        updated = re.sub(r'active_3p\s*=\s*"akagi-native3p"', 'active_3p = "local-adaptive3p"', content)
    if updated != content:
        tmp_cfg = config_path + ".tmp"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            f.write(updated)
        os.replace(tmp_cfg, config_path)

def train_mode(base_dir, mode, all_x, all_yp, all_yd, file_count, state_all):
    active_model_path = os.path.join(base_dir, "auto_learn", "models", f"active_model_{mode}.pt")
    shadow_model_path = os.path.join(base_dir, "auto_learn", "models", f"shadow_model_{mode}.pt")
    archive_dir = os.path.join(base_dir, "auto_learn", "archive")
    config_path = os.path.join(base_dir, "configs", "config.toml")

    mode_state = state_all.get(mode, {})
    if file_count <= mode_state.get("file_count", 0):
        return mode_state

    split = int(len(all_x) * 0.80)
    val_x = all_x[split:]
    val_yp = all_yp[split:]
    val_yd = all_yd[split:]
    if len(val_x) < MIN_VAL_SAMPLES:
        mode_state["file_count"] = file_count
        mode_state["val_samples"] = len(val_x)
        return mode_state

    train_ds = TensorDataset(
        torch.tensor(all_x[:split], dtype=torch.float32),
        torch.tensor(all_yp[:split], dtype=torch.long),
        torch.tensor(all_yd[:split], dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MahjongMultiTaskNet().to(device)
    if os.path.exists(shadow_model_path):
        try:
            model.load_state_dict(torch.load(shadow_model_path, map_location=device, weights_only=True))
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=2e-4)
    policy_criterion = nn.CrossEntropyLoss()
    danger_criterion = nn.BCEWithLogitsLoss()

    for _ in range(8):
        model.train()
        for bx, byp, byd in train_loader:
            bx, byp, byd = bx.to(device), byp.to(device), byd.to(device)
            optimizer.zero_grad()
            p_logits, d_logits = model(bx)
            loss_p = policy_criterion(p_logits, byp)
            loss_d = danger_criterion(d_logits, byd)
            loss = loss_p + 0.5 * loss_d
            loss.backward()
            optimizer.step()

    mean_acc, min_block_acc, danger_acc = evaluate_blocks(model, val_x, val_yp, val_yd, device)
    best_acc = mode_state.get("best_acc", 0.0)
    promoted = mode_state.get("promoted", False)
    streak = mode_state.get("streak", 0)

    target_bar = max(PROMOTION_THRESHOLD, best_acc + IMPROVEMENT_MARGIN) if promoted else PROMOTION_THRESHOLD
    min_block_bar = target_bar - 0.025

    if mean_acc >= target_bar and min_block_acc >= min_block_bar:
        streak += 1
        tmp_shadow = shadow_model_path + ".tmp"
        torch.save(model.state_dict(), tmp_shadow)
        os.replace(tmp_shadow, shadow_model_path)

        if streak >= REQUIRED_STREAK:
            if os.path.exists(active_model_path):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(active_model_path, os.path.join(archive_dir, f"model_{mode}_{stamp}.pt"))
            shutil.copy2(shadow_model_path, active_model_path)
            promote_bot_in_config(config_path, mode)
            best_acc = mean_acc
            promoted = True
            streak = 0
    else:
        streak = 0
        if not promoted and mean_acc > best_acc:
            tmp_shadow = shadow_model_path + ".tmp"
            torch.save(model.state_dict(), tmp_shadow)
            os.replace(tmp_shadow, shadow_model_path)
            best_acc = mean_acc

    return {
        "file_count": file_count,
        "val_samples": len(val_x),
        "last_mean_acc": round(mean_acc, 4),
        "last_min_block_acc": round(min_block_acc, 4),
        "danger_read_acc": round(danger_acc, 4),
        "best_acc": round(best_acc, 4),
        "streak": streak,
        "promoted": promoted,
        "updated_at": datetime.now().isoformat()
    }

def run_training():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    history_dir = os.path.join(base_dir, "history", "games")
    state_path = os.path.join(base_dir, "auto_learn", "train_state.json")
    os.makedirs(os.path.join(base_dir, "auto_learn", "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "auto_learn", "archive"), exist_ok=True)

    time.sleep(2.0)

    files = list_history_files(history_dir)
    if not files:
        return

    state_all = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as sf:
                state_all = json.load(sf)
        except Exception:
            state_all = {}

    data_by_mode = {"4p": ([], [], []), "3p": ([], [], [])}
    counts_by_mode = {"4p": 0, "3p": 0}

    for fp in files:
        mode, sx, syp, syd = parse_game_file(fp)
        counts_by_mode[mode] += 1
        data_by_mode[mode][0].extend(sx)
        data_by_mode[mode][1].extend(syp)
        data_by_mode[mode][2].extend(syd)

    for mode in ("4p", "3p"):
        state_all[mode] = train_mode(
            base_dir,
            mode,
            data_by_mode[mode][0],
            data_by_mode[mode][1],
            data_by_mode[mode][2],
            counts_by_mode[mode],
            state_all
        )

    with open(state_path, "w", encoding="utf-8") as sf:
        json.dump(state_all, sf, indent=2)

def main():
    lock_path = os.path.join(os.path.dirname(__file__), "trainer.lock")
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        run_training()

if __name__ == "__main__":
    main()