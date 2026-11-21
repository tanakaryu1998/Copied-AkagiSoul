import copy
import json
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLED_SITE_PACKAGES = os.path.join(
    BASE_DIR, "runtime", "python", "aarch64-apple-darwin", "lib", "python3.12", "site-packages"
)
if os.path.exists(BUNDLED_SITE_PACKAGES) and BUNDLED_SITE_PACKAGES not in sys.path:
    sys.path.append(BUNDLED_SITE_PACKAGES)

sys.path.insert(0, os.path.join(BASE_DIR, "mjai_bot", "local-adaptive"))
sys.path.insert(0, os.path.join(BASE_DIR, "auto_learn"))

import glob
for _vp in glob.glob(os.path.join(BASE_DIR, "mjai_bot", "local-adaptive", ".venv", "lib", "python*", "site-packages")):
    if _vp not in sys.path:
        sys.path.append(_vp)
import torch
from safetensors.torch import load_file
import base_engine
from model_core import (
    AkagiResNet,
    TILE_TYPES,
    compact_idx_from_34,
    encode_observation,
    tile34_from_str
)

NUM_WORKERS = 3

def build_wall(num_players):
    wall = []
    for idx34, name in enumerate(TILE_TYPES):
        if num_players == 3 and 1 <= idx34 <= 7:
            continue
        for copy_i in range(4):
            if copy_i == 0 and idx34 in (4, 13, 22):
                wall.append(f"5{'mps'[idx34 // 9]}r")
            else:
                wall.append(name)
    random.shuffle(wall)
    return wall

def hand_to_34_and_aka(hand_list):
    h34 = [0] * 34
    aka = {"m": 0, "p": 0, "s": 0}
    for p in hand_list:
        t34 = tile34_from_str(p)
        if 0 <= t34 < 34:
            h34[t34] += 1
        if p == "5mr":
            aka["m"] += 1
        elif p == "5pr":
            aka["p"] += 1
        elif p == "5sr":
            aka["s"] += 1
    return h34, aka

def extract_points(cost_obj, is_dealer, is_tsumo, num_players):
    if cost_obj is None:
        return 0
    if isinstance(cost_obj, int):
        return cost_obj
    if isinstance(cost_obj, (tuple, list)):
        if len(cost_obj) >= 2:
            if is_tsumo:
                if is_dealer:
                    return int(cost_obj[0]) * (num_players - 1)
                return int(cost_obj[0]) + int(cost_obj[1]) * max(1, num_players - 2)
            return int(cost_obj[0])
        if len(cost_obj) == 1:
            return int(cost_obj[0])
    for attr in ("total", "main", "ron", "tsumo_oya"):
        if hasattr(cost_obj, attr):
            val = getattr(cost_obj, attr)
            if isinstance(val, int) and val > 0:
                return val
    return 8000 if is_dealer else 5200

class VirtualSeatAgent:
    def __init__(self, model, device, temperature=0.0):
        self.model = model
        self.device = device
        self.temperature = temperature

    def predict(self, obs_vec, c_dim, t_dim):
        tx = torch.tensor([obs_vec], dtype=torch.float32, device=self.device).view(1, c_dim, t_dim)
        with torch.no_grad():
            logits = self.model(tx)[0]
            if self.temperature > 0.0:
                noise = torch.randn_like(logits) * self.temperature
                logits = logits + noise
            return logits.cpu().tolist()

def load_virtual_models(num_players, device):
    mode = "3p" if num_players == 3 else "4p"
    pretrained_path = os.path.join(BASE_DIR, "auto_learn", "pretrained", f"akagi{num_players}p.safetensors")
    shadow_path = os.path.join(BASE_DIR, "auto_learn", "models", f"shadow_model_{mode}.pt")

    base_model = AkagiResNet(num_players).to(device)
    if os.path.exists(pretrained_path):
        base_model.load_state_dict(load_file(pretrained_path, device=str(device)), strict=False)
    base_model.eval()

    learner_model = AkagiResNet(num_players).to(device)
    if os.path.exists(shadow_path):
        try:
            learner_model.load_state_dict(torch.load(shadow_path, map_location=device, weights_only=True), strict=False)
        except Exception:
            learner_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    else:
        learner_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    learner_model.eval()

    agents = [VirtualSeatAgent(learner_model, device, temperature=0.20)]
    for _ in range(num_players - 1):
        agents.append(VirtualSeatAgent(base_model, device, temperature=0.35))
    return agents

def simulate_hanchan(num_players, agents):
    events = [{"type": "start_game", "id": 0}]
    start_score = 35000 if num_players == 3 else 25000
    scores = [start_score] * num_players
    honba = 0
    riichi_sticks = 0
    wind_names = ["E", "S"]
    c_dim = 37 if num_players == 3 else 39
    t_dim = 27 if num_players == 3 else 34

    for round_wind in range(2):
        kyoku_idx = 0
        while kyoku_idx < num_players:
            if any(s < 0 for s in scores):
                break
            oya = kyoku_idx
            wall = build_wall(num_players)
            dead_wall_size = 14
            live_end = len(wall) - dead_wall_size
            dora_indicators = [wall[live_end]]
            ptr = 0

            hands = []
            for _ in range(num_players):
                hands.append(wall[ptr:ptr + 13])
                ptr += 13

            discards = [[] for _ in range(num_players)]
            melds = [[] for _ in range(num_players)]
            meld_counts = [0] * num_players
            riichi_declared = [False] * num_players
            riichi_tile = [None] * num_players
            kita_counts = [0] * num_players
            last_discard = None
            turn_count = 0
            dealer_keep = False
            kyoku_ended = False

            events.append({
                "type": "start_kyoku",
                "bakaze": wind_names[round_wind],
                "kyoku": kyoku_idx + 1,
                "honba": honba,
                "kyotaku": riichi_sticks,
                "oya": oya,
                "dora_marker": dora_indicators[0],
                "scores": list(scores),
                "tehais": [list(h) for h in hands]
            })

            actor = oya
            while ptr < live_end and not kyoku_ended:
                drawn = wall[ptr]
                ptr += 1
                turn_count += 1
                hands[actor].append(drawn)
                events.append({"type": "tsumo", "actor": actor, "pai": drawn})

                while True:
                    h34, aka = hand_to_34_and_aka(hands[actor])
                    drawn_idx = tile34_from_str(drawn) if drawn else -1
                    dummy_state = base_engine.State(actor_id=actor)
                    dummy_state.dora_markers = list(dora_indicators)

                    if 0 <= drawn_idx < 34:
                        res = base_engine._hand_value(dummy_state, h34, aka, drawn_idx, is_tsumo=True)
                        if res.cost is not None:
                            pts = max(1000, extract_points(res.cost, actor == oya, True, num_players))
                            deltas = [0] * num_players
                            if actor == oya:
                                share = ((pts + (num_players - 2)) // (num_players - 1) + 99) // 100 * 100
                                for s in range(num_players):
                                    if s == actor:
                                        deltas[s] = share * (num_players - 1) + honba * 300 + riichi_sticks * 1000
                                    else:
                                        deltas[s] = -(share + honba * 100)
                            else:
                                oya_pay = ((pts // 2) + 99) // 100 * 100
                                ko_pay = ((pts // 4) + 99) // 100 * 100
                                for s in range(num_players):
                                    if s == actor:
                                        deltas[s] = oya_pay + ko_pay * (num_players - 2) + honba * 300 + riichi_sticks * 1000
                                    elif s == oya:
                                        deltas[s] = -(oya_pay + honba * 100)
                                    else:
                                        deltas[s] = -(ko_pay + honba * 100)
                            for s in range(num_players):
                                scores[s] += deltas[s]
                            events.append({
                                "type": "hora",
                                "actor": actor,
                                "target": actor,
                                "pai": drawn,
                                "deltas": deltas,
                                "scores": list(scores),
                                "tehais": [list(h) for h in hands]
                            })
                            riichi_sticks = 0
                            if actor == oya:
                                dealer_keep = True
                                honba += 1
                            else:
                                honba = 0
                            kyoku_ended = True
                            break

                    if riichi_declared[actor]:
                        discard_pai = drawn
                        hands[actor].remove(discard_pai)
                        discards[actor].append(discard_pai)
                        last_discard = discard_pai
                        events.append({"type": "dahai", "actor": actor, "pai": discard_pai, "tsumogiri": True})
                        break

                    waits_34 = []
                    for i in range(34):
                        if h34[i] > 0:
                            t = list(h34)
                            t[i] -= 1
                            if base_engine._shanten_with_melds(t, meld_counts[actor]) == 0:
                                for w in base_engine._tenpai_waits(t, meld_counts[actor]):
                                    if w not in waits_34:
                                        waits_34.append(w)

                    seats_rel = []
                    for offset in range(num_players):
                        s = (actor + offset) % num_players
                        seats_rel.append({
                            "discards": discards[s],
                            "melds": melds[s],
                            "riichi": riichi_declared[s],
                            "riichi_tile": riichi_tile[s],
                            "score": scores[s],
                            "kita": kita_counts[s]
                        })

                    obs = encode_observation(
                        num_players=num_players,
                        hand_tiles=hands[actor],
                        drawn_tile=drawn,
                        waits_34=waits_34,
                        is_tenpai=len(waits_34) > 0,
                        dora_indicators=dora_indicators,
                        seats_rel=seats_rel,
                        last_discard=last_discard,
                        round_wind=round_wind,
                        seat_wind=(actor - oya) % num_players,
                        honba=honba,
                        riichi_sticks=riichi_sticks,
                        turn_count=turn_count,
                        is_dealer=(actor == oya),
                        kyoku_index=kyoku_idx,
                        self_riichi=riichi_declared[actor]
                    )
                    logits = agents[actor].predict(obs, c_dim, t_dim)
                    candidates = []

                    if num_players == 3 and "N" in hands[actor] and ptr < live_end:
                        candidates.append((logits[59], ("kita", "N")))

                    if meld_counts[actor] == 0 and not riichi_declared[actor] and (live_end - ptr) >= 4 and scores[actor] >= 1000:
                        for i in range(34):
                            if h34[i] > 0:
                                t = list(h34)
                                t[i] -= 1
                                if base_engine._shanten_with_melds(t, 0) == 0:
                                    r_id = 27 if num_players == 3 else 37
                                    candidates.append((logits[r_id], ("reach", None)))
                                    break

                    seen_34 = set()
                    for p_str in hands[actor]:
                        i34 = tile34_from_str(p_str)
                        if i34 in seen_34:
                            continue
                        seen_34.add(i34)
                        c_idx = compact_idx_from_34(i34, num_players)
                        if c_idx is not None:
                            chosen_str = p_str
                            if p_str.endswith("r"):
                                plain = p_str[:-1]
                                if plain in hands[actor]:
                                    chosen_str = plain
                            candidates.append((logits[c_idx], ("dahai", chosen_str)))

                    candidates.sort(key=lambda x: x[0], reverse=True)
                    act_type, act_pai = candidates[0][1]

                    if act_type == "kita":
                        hands[actor].remove("N")
                        kita_counts[actor] += 1
                        events.append({"type": "nukidora", "actor": actor, "pai": "N"})
                        if ptr >= live_end:
                            break
                        drawn = wall[ptr]
                        ptr += 1
                        hands[actor].append(drawn)
                        events.append({"type": "tsumo", "actor": actor, "pai": drawn})
                        continue

                    if act_type == "reach":
                        riichi_declared[actor] = True
                        events.append({"type": "reach", "actor": actor})
                        best_d = None
                        for p_str in hands[actor]:
                            i34 = tile34_from_str(p_str)
                            t = list(h34)
                            t[i34] -= 1
                            if base_engine._shanten_with_melds(t, 0) == 0:
                                c_idx = compact_idx_from_34(i34, num_players)
                                sc = logits[c_idx] if c_idx is not None else -1e9
                                if best_d is None or sc > best_d[0]:
                                    best_d = (sc, p_str)
                        discard_pai = best_d[1] if best_d else drawn
                        hands[actor].remove(discard_pai)
                        discards[actor].append(discard_pai)
                        last_discard = discard_pai
                        riichi_tile[actor] = discard_pai
                        events.append({"type": "dahai", "actor": actor, "pai": discard_pai, "tsumogiri": discard_pai == drawn})
                        scores[actor] -= 1000
                        riichi_sticks += 1
                        events.append({"type": "reach_accepted", "actor": actor})
                        break

                    hands[actor].remove(act_pai)
                    discards[actor].append(act_pai)
                    last_discard = act_pai
                    events.append({"type": "dahai", "actor": actor, "pai": act_pai, "tsumogiri": act_pai == drawn})
                    break

                if kyoku_ended:
                    break

                for offset in range(1, num_players):
                    other = (actor + offset) % num_players
                    oh34, oaka = hand_to_34_and_aka(hands[other])
                    win_idx = tile34_from_str(last_discard)
                    if 0 <= win_idx < 34:
                        test_h = list(oh34)
                        test_h[win_idx] += 1
                        dummy_state = base_engine.State(actor_id=other)
                        dummy_state.dora_markers = list(dora_indicators)
                        res = base_engine._hand_value(dummy_state, test_h, oaka, win_idx, is_tsumo=False)
                        if res.cost is not None:
                            pts = max(1000, extract_points(res.cost, other == oya, False, num_players))
                            deltas = [0] * num_players
                            total_gain = pts + honba * 300 + riichi_sticks * 1000
                            deltas[other] = total_gain
                            deltas[actor] = -(pts + honba * 300)
                            for s in range(num_players):
                                scores[s] += deltas[s]
                            ron_hand = list(hands[other]) + [last_discard]
                            all_h = [list(h) for h in hands]
                            all_h[other] = ron_hand
                            events.append({
                                "type": "hora",
                                "actor": other,
                                "target": actor,
                                "pai": last_discard,
                                "deltas": deltas,
                                "scores": list(scores),
                                "tehais": all_h
                            })
                            riichi_sticks = 0
                            if other == oya:
                                dealer_keep = True
                                honba += 1
                            else:
                                honba = 0
                            kyoku_ended = True
                            break

                if kyoku_ended:
                    break

                called = False
                if (live_end - ptr) >= 1:
                    base_d = last_discard[:-1] if last_discard.endswith("r") else last_discard
                    for offset in range(1, num_players):
                        other = (actor + offset) % num_players
                        if riichi_declared[other]:
                            continue
                        matching = [p for p in hands[other] if (p[:-1] if p.endswith("r") else p) == base_d]
                        if len(matching) >= 2:
                            seats_rel = []
                            for o2 in range(num_players):
                                s = (other + o2) % num_players
                                seats_rel.append({
                                    "discards": discards[s],
                                    "melds": melds[s],
                                    "riichi": riichi_declared[s],
                                    "riichi_tile": riichi_tile[s],
                                    "score": scores[s],
                                    "kita": kita_counts[s]
                                })
                            obs = encode_observation(
                                num_players=num_players,
                                hand_tiles=hands[other],
                                drawn_tile=None,
                                waits_34=[],
                                is_tenpai=False,
                                dora_indicators=dora_indicators,
                                seats_rel=seats_rel,
                                last_discard=last_discard,
                                round_wind=round_wind,
                                seat_wind=(other - oya) % num_players,
                                honba=honba,
                                riichi_sticks=riichi_sticks,
                                turn_count=turn_count,
                                is_dealer=(other == oya),
                                kyoku_index=kyoku_idx,
                                self_riichi=False
                            )
                            logits = agents[other].predict(obs, c_dim, t_dim)
                            pon_id = 28 if num_players == 3 else 41
                            pass_id = 58 if num_players == 3 else 81
                            if logits[pon_id] > logits[pass_id]:
                                consumed = matching[:2]
                                for c in consumed:
                                    hands[other].remove(c)
                                    melds[other].append(c)
                                melds[other].append(last_discard)
                                meld_counts[other] += 1
                                events.append({
                                    "type": "pon",
                                    "actor": other,
                                    "target": actor,
                                    "pai": last_discard,
                                    "consumed": consumed
                                })
                                actor = other
                                obs2 = encode_observation(
                                    num_players=num_players,
                                    hand_tiles=hands[actor],
                                    drawn_tile=None,
                                    waits_34=[],
                                    is_tenpai=False,
                                    dora_indicators=dora_indicators,
                                    seats_rel=seats_rel,
                                    last_discard=last_discard,
                                    round_wind=round_wind,
                                    seat_wind=(actor - oya) % num_players,
                                    honba=honba,
                                    riichi_sticks=riichi_sticks,
                                    turn_count=turn_count,
                                    is_dealer=(actor == oya),
                                    kyoku_index=kyoku_idx,
                                    self_riichi=False
                                )
                                logits2 = agents[actor].predict(obs2, c_dim, t_dim)
                                best_d = None
                                for p_str in hands[actor]:
                                    i34 = tile34_from_str(p_str)
                                    c_idx = compact_idx_from_34(i34, num_players)
                                    sc = logits2[c_idx] if c_idx is not None else -1e9
                                    if best_d is None or sc > best_d[0]:
                                        best_d = (sc, p_str)
                                discard_pai = best_d[1]
                                hands[actor].remove(discard_pai)
                                discards[actor].append(discard_pai)
                                last_discard = discard_pai
                                events.append({"type": "dahai", "actor": actor, "pai": discard_pai, "tsumogiri": False})
                                actor = (actor + 1) % num_players
                                called = True
                                break

                if not called:
                    actor = (actor + 1) % num_players

            if not kyoku_ended:
                tenpai_seats = []
                for s in range(num_players):
                    h34, _ = hand_to_34_and_aka(hands[s])
                    if base_engine._shanten_with_melds(h34, meld_counts[s]) == 0:
                        tenpai_seats.append(s)
                deltas = [0] * num_players
                if 0 < len(tenpai_seats) < num_players:
                    pot = 3000 if num_players == 4 else 2000
                    gain = pot // len(tenpai_seats)
                    loss = pot // (num_players - len(tenpai_seats))
                    for s in range(num_players):
                        deltas[s] = gain if s in tenpai_seats else -loss
                        scores[s] += deltas[s]
                if oya in tenpai_seats:
                    dealer_keep = True
                honba += 1
                events.append({
                    "type": "ryukyoku",
                    "deltas": deltas,
                    "scores": list(scores),
                    "tehais": [list(h) for h in hands]
                })

            events.append({"type": "end_kyoku", "scores": list(scores)})
            if not dealer_keep:
                kyoku_idx += 1

    events.append({"type": "end_game", "scores": list(scores)})
    return events

def worker_loop(worker_id, stop_event):
    torch.set_num_threads(1)
    v_dir = os.path.join(BASE_DIR, "history", "virtual_games")
    os.makedirs(v_dir, exist_ok=True)
    device = torch.device("cpu")
    trainer_script = os.path.join(BASE_DIR, "auto_learn", "trainer.py")
    py_bin = sys.executable

    random.seed(int(time.time() * 1000) + worker_id * 997)
    game_idx = worker_id

    while not stop_event.is_set():
        num_players = 4 if (game_idx % 2 == 0) else 3
        agents = load_virtual_models(num_players, device)
        events = simulate_hanchan(num_players, agents)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(v_dir, f"virtual_{num_players}p_w{worker_id}_{stamp}.jsonl")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, separators=(",", ":")) + "\n")
        os.replace(tmp_path, out_path)

        game_idx += 1
        if worker_id == 0 and game_idx % 40 == 0:
            import glob as _g
            for _pat in ("virtual_4p_*.jsonl", "virtual_3p_*.jsonl"):
                _vfiles = sorted(_g.glob(os.path.join(v_dir, _pat)), key=os.path.getmtime)
                for _old in _vfiles[:-200]:
                    try:
                        os.remove(_old)
                    except OSError:
                        pass
            subprocess.Popen([py_bin, trainer_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.1)

def main():
    stop_event = mp.Event()

    def handle_sig(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    workers = []
    for wid in range(NUM_WORKERS):
        p = mp.Process(target=worker_loop, args=(wid, stop_event), daemon=True)
        p.start()
        workers.append(p)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    finally:
        stop_event.set()
        for p in workers:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()

if __name__ == "__main__":
    main()