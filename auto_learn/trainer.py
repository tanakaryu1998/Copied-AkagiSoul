import os
import sys

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SITE = os.path.join(_BASE, "runtime", "python", "aarch64-apple-darwin", "lib", "python3.12", "site-packages")
if os.path.exists(_SITE) and _SITE not in sys.path:
    sys.path.append(_SITE)

import copy
import fcntl
import glob
import json
import re
import shutil
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import load_file
from model_core import (
    AkagiResNet,
    compact_idx_from_34,
    encode_observation,
    tile34_from_str
)

torch.set_num_threads(2)

MIN_GAMES = 100
MIN_VAL_SAMPLES = 1500
PROMOTION_THRESHOLD = 0.72
IMPROVEMENT_MARGIN = 0.005
REQUIRED_STREAK = 2
MAX_VIRTUAL_PER_MODE = 80

def rank_weights_for_scores(final_scores, num_players):
    indexed = sorted(enumerate(final_scores[:num_players]), key=lambda x: x[1], reverse=True)
    if num_players == 4:
        base_map = [1.65, 1.25, 0.90, 0.25]
    else:
        base_map = [1.60, 1.05, 0.35]
    weights = [1.0] * num_players
    ranks = [num_players] * num_players
    start_pts = 35000.0 if num_players == 3 else 25000.0
    for rank_idx, (seat, pts) in enumerate(indexed):
        ranks[seat] = rank_idx + 1
        bonus = max(-0.15, min(0.25, (pts - start_pts) / 100000.0))
        weights[seat] = max(0.15, base_map[rank_idx] + bonus)
    return weights, ranks

def get_current_rank(cur_scores, actor, num_players):
    my_s = cur_scores[actor]
    higher = sum(1 for s in cur_scores[:num_players] if s > my_s)
    return higher + 1

def situation_multiplier(cur_scores, actor, num_players, round_wind, kyoku_index, opp_riichi_count):
    mult = 1.0
    is_south_or_later = (round_wind >= 1)
    is_oorasu = (num_players == 4 and round_wind == 1 and kyoku_index >= 3) or (
        num_players == 3 and round_wind == 1 and kyoku_index >= 2
    )
    my_score = cur_scores[actor]
    sorted_s = sorted(cur_scores[:num_players], reverse=True)
    if len(sorted_s) >= 2:
        min_gap = min(
            abs(my_score - s)
            for s in sorted_s
            if s != my_score or sorted_s.count(my_score) > 1
        )
        if min_gap <= 8000:
            mult *= 1.20
    if is_oorasu:
        mult *= 1.35
    elif is_south_or_later:
        mult *= 1.15
    if opp_riichi_count > 0:
        mult *= 1.15
    return mult

def encode_action_id(etype, pai, consumed, num_players):
    if etype == "dahai":
        t34 = tile34_from_str(pai)
        return compact_idx_from_34(t34, num_players)
    if etype == "reach":
        return 27 if num_players == 3 else 37
    if etype == "pon":
        return 28 if num_players == 3 else 41
    if etype == "chi" and num_players == 4:
        p34 = tile34_from_str(pai)
        c34 = sorted([tile34_from_str(c) for c in consumed if tile34_from_str(c) >= 0])
        if len(c34) == 2:
            if p34 < c34[0]:
                return 38
            if c34[0] < p34 < c34[1]:
                return 39
            return 40
    if etype in ("kan", "daiminkan", "kakan", "ankan"):
        target_tile = pai if pai else (consumed[0] if consumed else None)
        if target_tile:
            c_idx = compact_idx_from_34(tile34_from_str(target_tile), num_players)
            if c_idx is not None:
                return (29 + c_idx) if num_players == 3 else (42 + c_idx)
    if etype == "hora":
        return 56 if num_players == 3 else 79
    if etype == "ryukyoku":
        return 57 if num_players == 3 else 80
    if etype == "nukidora" and num_players == 3:
        return 59
    return None

def reconstruct_kyoku_hands(kyoku_events, num_players, my_seat):
    n_ev = len(kyoku_events)
    snapshots = [{} for _ in range(n_ev)]
    revealed_final = {}

    for ev in reversed(kyoku_events):
        if ev.get("type") in ("hora", "ryukyoku"):
            tehais = ev.get("tehais", [])
            for s in range(min(num_players, len(tehais))):
                if s != my_seat and len(tehais[s]) > 0 and "?" not in tehais[s]:
                    revealed_final[s] = list(tehais[s])
            break

    for s, final_hand in revealed_final.items():
        cur_h = list(final_hand)
        for idx in range(n_ev - 1, -1, -1):
            ev = kyoku_events[idx]
            snapshots[idx][s] = list(cur_h)
            etype = ev.get("type")
            actor = ev.get("actor")
            if actor != s:
                continue
            pai = ev.get("pai")
            consumed = ev.get("consumed", [])
            if etype == "dahai" and pai:
                cur_h.append(pai)
            elif etype == "tsumo" and pai:
                if pai in cur_h:
                    cur_h.remove(pai)
            elif etype in ("pon", "chi", "daiminkan", "ankan"):
                for c in consumed:
                    cur_h.append(c)
            elif etype == "kakan" and pai:
                cur_h.append(pai)
            elif etype == "nukidora":
                cur_h.append("N")

    return snapshots, revealed_final

def parse_game_file(filepath, recency_weight=1.0, is_virtual=False):
    events = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not events:
        return "4p", [], [], [], [], [], [], None

    num_players = 4
    my_seat = 0
    for ev in events:
        if ev.get("type") == "start_game" and "id" in ev:
            my_seat = ev.get("id", 0)
        if ev.get("type") == "start_kyoku":
            tehais = ev.get("tehais", [])
            non_empty = [t for t in tehais if len(t) > 0]
            if len(non_empty) == 3 or len(tehais) == 3:
                num_players = 3
            break

    default_score = 35000 if num_players == 3 else 25000
    scores = [default_score] * 4
    for ev in reversed(events):
        if "scores" in ev and isinstance(ev["scores"], list) and len(ev["scores"]) >= num_players:
            scores = list(ev["scores"])
            break
    seat_weights, seat_ranks = rank_weights_for_scores(scores, num_players)

    if is_virtual:
        for s in range(num_players):
            if seat_ranks[s] == 1:
                my_seat = s
                break

    source_scale = 0.55 if is_virtual else 1.0

    kyokus = []
    cur_k = []
    for ev in events:
        if ev.get("type") == "start_kyoku":
            if cur_k:
                kyokus.append(cur_k)
            cur_k = [ev]
        elif cur_k:
            cur_k.append(ev)
    if cur_k:
        kyokus.append(cur_k)

    samples_x = []
    samples_y = []
    samples_w = []
    samples_pmask = []
    samples_yr = []
    samples_rmask = []

    cur_scores = [default_score] * 4
    total_kyoku = 0
    my_deal_ins = 0
    my_wins = 0
    wind_map = {"E": 0, "S": 1, "W": 2, "N": 3}
    t_dim = 27 if num_players == 3 else 34

    for k_events in kyokus:
        total_kyoku += 1
        snapshots = None if is_virtual else reconstruct_kyoku_hands(k_events, num_players, my_seat)[0]

        winner_seat = None
        loser_seat = None
        delta_scores = None
        win_pai_compact = None
        for ev in k_events:
            if ev.get("type") == "hora":
                winner_seat = ev.get("actor")
                loser_seat = ev.get("target")
                delta_scores = ev.get("deltas", [])
                win_pai = ev.get("pai")
                if winner_seat != my_seat and win_pai:
                    win_pai_compact = compact_idx_from_34(tile34_from_str(win_pai), num_players)
                if winner_seat == my_seat:
                    my_wins += 1
                if loser_seat == my_seat and winner_seat != my_seat:
                    my_deal_ins += 1
                break

        hands = [[], [], [], []]
        drawn_tiles = [None] * 4
        discards = [[], [], [], []]
        melds = [[], [], [], []]
        riichi_declared = [False] * 4
        riichi_tile = [None] * 4
        kita_counts = [0] * 4
        kyoku_start_scores = list(cur_scores)
        dora_indicators = []
        last_discard = None
        round_wind = 0
        oya = 0
        honba = 0
        riichi_sticks = 0
        turn_count = 0
        kyoku_index = 0

        for step_idx, ev in enumerate(k_events):
            etype = ev.get("type")
            if etype == "start_kyoku":
                round_wind = wind_map.get(ev.get("bakaze", "E"), 0)
                oya = ev.get("oya", 0)
                honba = ev.get("honba", 0)
                riichi_sticks = ev.get("kyotaku", 0)
                kyoku_index = max(0, ev.get("kyoku", 1) - 1)
                if "scores" in ev and len(ev["scores"]) >= num_players:
                    cur_scores = list(ev["scores"])
                    kyoku_start_scores = list(ev["scores"])
                dm = ev.get("dora_marker")
                dora_indicators = [dm] if dm else []
                tehais = ev.get("tehais", [])
                for i in range(min(4, len(tehais))):
                    hands[i] = list(tehais[i])
            elif etype == "dora":
                dm = ev.get("dora_marker")
                if dm:
                    dora_indicators.append(dm)
            elif etype == "tsumo":
                actor = ev.get("actor")
                pai = ev.get("pai")
                if actor is not None and 0 <= actor < num_players and pai:
                    turn_count += 1
                    drawn_tiles[actor] = pai
                    hands[actor].append(pai)
            elif etype in ("dahai", "reach", "pon", "chi", "kan", "daiminkan", "kakan", "ankan", "hora", "nukidora"):
                actor = ev.get("actor")
                pai = ev.get("pai")
                consumed = ev.get("consumed", [])

                if etype == "dahai" and pai:
                    discards[actor].append(pai)
                    last_discard = pai
                    if riichi_declared[actor] and riichi_tile[actor] is None:
                        riichi_tile[actor] = pai
                    if pai in hands[actor]:
                        hands[actor].remove(pai)
                    elif "?" in hands[actor]:
                        hands[actor].remove("?")
                    drawn_tiles[actor] = None
                elif etype == "reach":
                    riichi_declared[actor] = True
                elif etype == "reach_accepted":
                    riichi_sticks += 1
                    cur_scores[actor] -= 1000
                elif etype == "nukidora":
                    kita_counts[actor] += 1
                    if "N" in hands[actor]:
                        hands[actor].remove("N")
                    elif "?" in hands[actor]:
                        hands[actor].remove("?")
                elif etype in ("pon", "chi", "kan", "daiminkan", "kakan", "ankan"):
                    if pai:
                        melds[actor].append(pai)
                    for c in consumed:
                        melds[actor].append(c)
                        if c in hands[actor]:
                            hands[actor].remove(c)
                        elif "?" in hands[actor]:
                            hands[actor].remove("?")
                    drawn_tiles[actor] = None

                if actor is not None and 0 <= actor < num_players and len(hands[my_seat]) > 0 and "?" not in hands[my_seat]:
                    opp_target_vec = [0.0] * t_dim
                    has_opp_target = 0.0

                    if is_virtual:
                        for s in range(num_players):
                            if s != my_seat and len(hands[s]) > 0 and "?" not in hands[s]:
                                has_opp_target = 1.0
                                for t_str in hands[s]:
                                    c_idx = compact_idx_from_34(tile34_from_str(t_str), num_players)
                                    if c_idx is not None:
                                        opp_target_vec[c_idx] = 1.0
                    elif snapshots is not None:
                        for s, r_hand in snapshots[step_idx].items():
                            if s != my_seat and len(r_hand) > 0:
                                has_opp_target = 1.0
                                for t_str in r_hand:
                                    c_idx = compact_idx_from_34(tile34_from_str(t_str), num_players)
                                    if c_idx is not None:
                                        opp_target_vec[c_idx] = 1.0

                    if win_pai_compact is not None:
                        opp_target_vec[win_pai_compact] = 1.0
                        has_opp_target = 1.0

                    is_my_decision = (actor == my_seat) and (etype in ("dahai", "reach", "pon", "chi", "kan", "daiminkan", "kakan", "ankan", "hora", "nukidora"))
                    act_id = encode_action_id(etype, pai, consumed, num_players) if is_my_decision else 0

                    if (is_my_decision and act_id is not None) or (has_opp_target > 0.0 and etype == "dahai"):
                        seats_rel = []
                        opp_riichi_cnt = 0
                        for offset in range(num_players):
                            s = (my_seat + offset) % num_players
                            if offset > 0 and riichi_declared[s]:
                                opp_riichi_cnt += 1
                            seats_rel.append({
                                "discards": discards[s],
                                "melds": melds[s],
                                "riichi": riichi_declared[s],
                                "riichi_tile": riichi_tile[s],
                                "score": cur_scores[s],
                                "kita": kita_counts[s]
                            })
                        my_seat_wind = (my_seat - oya) % num_players
                        d_tile = drawn_tiles[my_seat] if (drawn_tiles[my_seat] and drawn_tiles[my_seat] != "?") else None
                        obs_vec = encode_observation(
                            num_players=num_players,
                            hand_tiles=hands[my_seat],
                            drawn_tile=d_tile,
                            waits_34=[],
                            is_tenpai=riichi_declared[my_seat],
                            dora_indicators=dora_indicators,
                            seats_rel=seats_rel,
                            last_discard=last_discard,
                            round_wind=round_wind,
                            seat_wind=my_seat_wind,
                            honba=honba,
                            riichi_sticks=riichi_sticks,
                            turn_count=turn_count,
                            is_dealer=(my_seat == oya),
                            kyoku_index=kyoku_index,
                            self_riichi=riichi_declared[my_seat]
                        )
                        sit_mult = situation_multiplier(
                            cur_scores, my_seat, num_players, round_wind, kyoku_index, opp_riichi_cnt
                        )
                        if is_my_decision and etype in ("pon", "chi", "kan", "daiminkan", "kakan", "ankan"):
                            sit_mult *= 1.15

                        adj = 1.0
                        if loser_seat is not None and my_seat == loser_seat and winner_seat != loser_seat:
                            adj = 0.35
                        elif winner_seat is not None and my_seat == winner_seat:
                            gain = delta_scores[my_seat] if delta_scores and my_seat < len(delta_scores) else 0
                            cur_rank = get_current_rank(kyoku_start_scores, my_seat, num_players)
                            if cur_rank == 1:
                                adj = 1.35 if gain <= 5200 else 1.22
                            elif cur_rank >= num_players - 1:
                                adj = 1.40 if gain >= 8000 else 1.12
                            else:
                                adj = 1.25 if gain >= 8000 else 1.22

                        samples_x.append(obs_vec)
                        samples_y.append(act_id if act_id is not None else 0)
                        samples_w.append(seat_weights[my_seat] * sit_mult * adj * recency_weight * source_scale)
                        samples_pmask.append(1.0 if (is_my_decision and act_id is not None) else 0.0)
                        samples_yr.append(opp_target_vec)
                        samples_rmask.append(has_opp_target)

    mode = "3p" if num_players == 3 else "4p"
    stats = None if is_virtual else {
        "rank": seat_ranks[my_seat] if my_seat < len(seat_ranks) else num_players,
        "score": scores[my_seat] if my_seat < len(scores) else default_score,
        "kyoku": max(1, total_kyoku),
        "wins": my_wins,
        "deal_ins": my_deal_ins
    }
    return mode, samples_x, samples_y, samples_w, samples_pmask, samples_yr, samples_rmask, stats

def list_history_files(history_dir):
    if not os.path.exists(history_dir):
        return []
    patterns = [
        os.path.join(history_dir, "**", "*.jsonl"),
        os.path.join(history_dir, "**", "*.json"),
        os.path.join(history_dir, "**", "*.mjson")
    ]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))
    return sorted(files)

def evaluate_blocks(model, val_x_np, val_y_np, val_pmask_np, val_yr_np, val_rmask_np, c_dim, t_dim, device):
    model.eval()
    own_indices = np.where(val_pmask_np > 0.5)[0]
    if len(own_indices) < MIN_VAL_SAMPLES:
        return 0.0, 0.0, 0.0
    block_size = len(own_indices) // 3
    block_accs = []
    with torch.no_grad():
        for b in range(3):
            start = b * block_size
            end = len(own_indices) if b == 2 else (b + 1) * block_size
            sub_idx = own_indices[start:end]
            bx = torch.from_numpy(val_x_np[sub_idx]).view(-1, c_dim, t_dim).to(device)
            by = torch.from_numpy(val_y_np[sub_idx]).to(device)
            preds = model(bx).argmax(dim=1)
            acc = (preds == by).sum().item() / by.size(0)
            block_accs.append(acc)

        read_indices = np.where(val_rmask_np > 0.5)[0]
        mean_read = 0.0
        if len(read_indices) > 0:
            read_indices = read_indices[:2000]
            rx = torch.from_numpy(val_x_np[read_indices]).view(-1, c_dim, t_dim).to(device)
            ry = torch.from_numpy(val_yr_np[read_indices]).to(device)
            _, r_logits = model(rx, return_hand_read=True)
            r_preds = (torch.sigmoid(r_logits) >= 0.45).float()
            mean_read = (r_preds == ry).float().mean().item()

    return sum(block_accs) / 3.0, min(block_accs), mean_read

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

def summarize_game_stats(stats_list, num_players):
    if not stats_list:
        return {}
    n = len(stats_list)
    avg_rank = sum(s["rank"] for s in stats_list) / n
    top_rate = sum(1 for s in stats_list if s["rank"] == 1) / n
    last_rate = sum(1 for s in stats_list if s["rank"] == num_players) / n
    total_k = sum(s["kyoku"] for s in stats_list)
    win_rate = sum(s["wins"] for s in stats_list) / max(1, total_k)
    deal_in_rate = sum(s["deal_ins"] for s in stats_list) / max(1, total_k)
    return {
        "avg_rank": round(avg_rank, 2),
        "top_rate": round(top_rate, 4),
        "last_rate": round(last_rate, 4),
        "win_rate": round(win_rate, 4),
        "deal_in_rate": round(deal_in_rate, 4)
    }

def train_mode(base_dir, mode, all_x, all_y, all_w, all_pmask, all_yr, all_rmask, real_count, virtual_count, stats_list, state_all):
    num_players = 3 if mode == "3p" else 4
    active_model_path = os.path.join(base_dir, "auto_learn", "models", f"active_model_{mode}.pt")
    shadow_model_path = os.path.join(base_dir, "auto_learn", "models", f"shadow_model_{mode}.pt")
    pretrained_path = os.path.join(base_dir, "auto_learn", "pretrained", f"akagi{num_players}p.safetensors")
    archive_dir = os.path.join(base_dir, "auto_learn", "archive")
    config_path = os.path.join(base_dir, "configs", "config.toml")

    perf = summarize_game_stats(stats_list, num_players)
    mode_state = state_all.get(mode, {})
    prev_real = mode_state.get("file_count", 0)
    prev_virt = mode_state.get("virtual_count", 0)
    if real_count <= prev_real and virtual_count <= prev_virt:
        mode_state["performance"] = perf
        return mode_state

    if not all_x:
        mode_state["file_count"] = real_count
        mode_state["virtual_count"] = virtual_count
        mode_state["performance"] = perf
        return mode_state

    print(f"[{mode.upper()}] 配列変換中 ({len(all_x)} ステップ)...", flush=True)
    x_np = np.asarray(all_x, dtype=np.float32)
    y_np = np.asarray(all_y, dtype=np.int64)
    w_np = np.asarray(all_w, dtype=np.float32)
    pm_np = np.asarray(all_pmask, dtype=np.float32)
    yr_np = np.asarray(all_yr, dtype=np.float32)
    rm_np = np.asarray(all_rmask, dtype=np.float32)

    split = int(len(x_np) * 0.80)
    val_x_np = x_np[split:]
    val_y_np = y_np[split:]
    val_pm_np = pm_np[split:]
    val_yr_np = yr_np[split:]
    val_rm_np = rm_np[split:]

    own_val_count = int(np.sum(val_pm_np > 0.5))
    if own_val_count < MIN_VAL_SAMPLES:
        print(f"[{mode.upper()}] 検証打牌数不足 ({own_val_count}/{MIN_VAL_SAMPLES}) のため蓄積のみ記録します。", flush=True)
        mode_state["file_count"] = real_count
        mode_state["virtual_count"] = virtual_count
        mode_state["val_samples"] = own_val_count
        mode_state["performance"] = perf
        mode_state["updated_at"] = datetime.now().isoformat()
        return mode_state

    device = torch.device("cpu")
    model = AkagiResNet(num_players).to(device)
    anchor_model = AkagiResNet(num_players).to(device)

    if os.path.exists(pretrained_path):
        base_sd = load_file(pretrained_path, device="cpu")
        anchor_model.load_state_dict(base_sd, strict=False)
        if os.path.exists(shadow_model_path):
            model.load_state_dict(torch.load(shadow_model_path, map_location=device, weights_only=True), strict=False)
        else:
            model.load_state_dict(base_sd, strict=False)
    elif os.path.exists(shadow_model_path):
        model.load_state_dict(torch.load(shadow_model_path, map_location=device, weights_only=True), strict=False)
        anchor_model.load_state_dict(copy.deepcopy(model.state_dict()), strict=False)

    anchor_model.eval()
    for p in anchor_model.parameters():
        p.requires_grad = False

    c_dim = model.channels
    t_dim = model.tile_dim

    train_ds = TensorDataset(
        torch.from_numpy(x_np[:split]),
        torch.from_numpy(y_np[:split]),
        torch.from_numpy(w_np[:split]),
        torch.from_numpy(pm_np[:split]),
        torch.from_numpy(yr_np[:split]),
        torch.from_numpy(rm_np[:split])
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    for ep in range(3):
        model.train()
        for bx, by, bw, bpm, byr, brm in train_loader:
            bx = bx.view(-1, c_dim, t_dim)
            optimizer.zero_grad()
            p_logits, r_logits = model(bx, return_hand_read=True)

            ce_loss = F.cross_entropy(p_logits, by, reduction="none")
            weighted_policy_loss = (ce_loss * bw * bpm).sum() / (bpm.sum() + 1e-6)

            with torch.no_grad():
                anchor_probs = F.softmax(anchor_model(bx), dim=1)
            kl_per_sample = F.kl_div(F.log_softmax(p_logits, dim=1), anchor_probs, reduction="none").sum(dim=1)
            kl_loss = (kl_per_sample * bpm).sum() / (bpm.sum() + 1e-6)

            read_bce = F.binary_cross_entropy_with_logits(r_logits, byr, reduction="none").mean(dim=1)
            read_loss = (read_bce * brm).sum() / (brm.sum() + 1e-6)

            loss = weighted_policy_loss + 0.25 * kl_loss + 0.45 * read_loss
            loss.backward()
            optimizer.step()
        print(f"[{mode.upper()}] エポック {ep + 1}/3 完了", flush=True)

    mean_acc, min_block_acc, opp_read_acc = evaluate_blocks(
        model, val_x_np, val_y_np, val_pm_np, val_yr_np, val_rm_np, c_dim, t_dim, device
    )
    print(f"[{mode.upper()}] 評価完了 -> 打牌一致率: {mean_acc*100:.2f}% / 手牌読み率: {opp_read_acc*100:.2f}%", flush=True)

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

        if streak >= REQUIRED_STREAK and real_count >= MIN_GAMES:
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
        "file_count": real_count,
        "virtual_count": virtual_count,
        "val_samples": own_val_count,
        "last_mean_acc": round(mean_acc, 4),
        "last_min_block_acc": round(min_block_acc, 4),
        "opp_read_acc": round(opp_read_acc, 4),
        "best_acc": round(best_acc, 4),
        "streak": streak,
        "promoted": promoted,
        "performance": perf,
        "updated_at": datetime.now().isoformat()
    }

def run_training():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    real_dir = os.path.join(base_dir, "history", "games")
    virtual_dir = os.path.join(base_dir, "history", "virtual_games")
    state_path = os.path.join(base_dir, "auto_learn", "train_state.json")
    os.makedirs(os.path.join(base_dir, "auto_learn", "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "auto_learn", "archive"), exist_ok=True)

    real_files = list_history_files(real_dir)
    virtual_files_all = list_history_files(virtual_dir)
    if not real_files and not virtual_files_all:
        print("学習対象ファイルが見つかりません。", flush=True)
        return

    v4_all = [f for f in virtual_files_all if "virtual_4p_" in os.path.basename(f)]
    v3_all = [f for f in virtual_files_all if "virtual_3p_" in os.path.basename(f)]
    virtual_files = v4_all[-MAX_VIRTUAL_PER_MODE:] + v3_all[-MAX_VIRTUAL_PER_MODE:]

    print(f"牌譜読み込み開始 (実戦: {len(real_files)}戦 / 仮想抽出: {len(virtual_files)}戦)...", flush=True)

    state_all = {}
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as sf:
                state_all = json.load(sf)
        except Exception:
            state_all = {}

    data_by_mode = {"4p": ([], [], [], [], [], []), "3p": ([], [], [], [], [], [])}
    real_counts = {"4p": 0, "3p": 0}
    virt_counts = {"4p": len(v4_all), "3p": len(v3_all)}
    stats_by_mode = {"4p": [], "3p": []}

    for idx, fp in enumerate(virtual_files):
        recency_w = 0.85 + 0.25 * ((idx + 1) / max(1, len(virtual_files)))
        mode, sx, sy, sw, spm, syr, srm, _ = parse_game_file(fp, recency_weight=recency_w, is_virtual=True)
        data_by_mode[mode][0].extend(sx)
        data_by_mode[mode][1].extend(sy)
        data_by_mode[mode][2].extend(sw)
        data_by_mode[mode][3].extend(spm)
        data_by_mode[mode][4].extend(syr)
        data_by_mode[mode][5].extend(srm)

    for idx, fp in enumerate(real_files):
        recency_w = 0.90 + 0.30 * ((idx + 1) / max(1, len(real_files)))
        mode, sx, sy, sw, spm, syr, srm, st = parse_game_file(fp, recency_weight=recency_w, is_virtual=False)
        real_counts[mode] += 1
        data_by_mode[mode][0].extend(sx)
        data_by_mode[mode][1].extend(sy)
        data_by_mode[mode][2].extend(sw)
        data_by_mode[mode][3].extend(spm)
        data_by_mode[mode][4].extend(syr)
        data_by_mode[mode][5].extend(srm)
        if st is not None:
            stats_by_mode[mode].append(st)

    for mode in ("4p", "3p"):
        state_all[mode] = train_mode(
            base_dir,
            mode,
            data_by_mode[mode][0],
            data_by_mode[mode][1],
            data_by_mode[mode][2],
            data_by_mode[mode][3],
            data_by_mode[mode][4],
            data_by_mode[mode][5],
            real_counts[mode],
            virt_counts[mode],
            stats_by_mode[mode],
            state_all
        )

    with open(state_path, "w", encoding="utf-8") as sf:
        json.dump(state_all, sf, indent=2)
    print("学習データの保存が完了しました。", flush=True)

def main():
    lock_path = os.path.join(os.path.dirname(__file__), "trainer.lock")
    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("別の学習プロセスが実行中のためスキップしました。", flush=True)
            return
        run_training()

if __name__ == "__main__":
    main()
