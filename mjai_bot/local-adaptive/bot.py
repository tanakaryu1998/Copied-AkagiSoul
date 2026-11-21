import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUNDLED_SITE_PACKAGES = os.path.join(
    BASE_DIR, "runtime", "python", "aarch64-apple-darwin", "lib", "python3.12", "site-packages"
)
if os.path.exists(BUNDLED_SITE_PACKAGES) and BUNDLED_SITE_PACKAGES not in sys.path:
    sys.path.append(BUNDLED_SITE_PACKAGES)

sys.path.insert(0, os.path.join(BASE_DIR, "auto_learn"))

import torch
import base_engine
from model_core import (
    TILE_TYPES,
    compact_idx_from_34,
    encode_observation,
    load_pretrained_resnet,
    tile34_from_str,
    tile_str_from_compact
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, model_source = load_pretrained_resnet(BASE_DIR, "4p", device)
model.eval()

class PureNeuralSession4P:
    def __init__(self):
        self.state = base_engine.State(actor_id=base_engine._initial_actor_id())
        self.drawn_tile = None
        self.last_discard = None
        self.riichi_tiles = [None] * 4
        self.scores = [25000] * 4
        self.honba = 0
        self.riichi_sticks = 0
        self.kyoku_index = 0
        self.turn_count = 0
        self.last_opp_read_probs = None

    def observe(self, ev):
        etype = ev.get("type")
        if etype == "start_kyoku":
            self.drawn_tile = None
            self.last_discard = None
            self.riichi_tiles = [None] * 4
            self.last_opp_read_probs = None
            self.honba = ev.get("honba", 0)
            self.riichi_sticks = ev.get("kyotaku", 0)
            self.kyoku_index = max(0, ev.get("kyoku", 1) - 1)
            self.turn_count = 0
            if "scores" in ev and len(ev["scores"]) >= 4:
                self.scores = list(ev["scores"])
        elif etype == "tsumo":
            if ev.get("actor") == self.state.actor_id:
                self.drawn_tile = ev.get("pai")
                self.turn_count += 1
        elif etype == "dahai":
            actor = ev.get("actor")
            pai = ev.get("pai")
            self.last_discard = pai
            if actor == self.state.actor_id:
                self.drawn_tile = None
            if actor is not None and 0 <= actor < 4:
                if actor in getattr(self.state, "reach_declared_seats", set()) and self.riichi_tiles[actor] is None:
                    self.riichi_tiles[actor] = pai
        elif etype == "reach_accepted":
            actor = ev.get("actor")
            self.riichi_sticks += 1
            if actor is not None and 0 <= actor < 4:
                self.scores[actor] -= 1000
        elif etype in ("hora", "ryukyoku"):
            self.verify_opponent_prediction(ev)
        self.state.consume(ev)

    def verify_opponent_prediction(self, ev):
        if self.last_opp_read_probs is None:
            return
        my_seat = self.state.actor_id
        winner = ev.get("actor")
        win_pai = ev.get("pai")
        tehais = ev.get("tehais", [])

        actual_indices = set()
        for s in range(min(4, len(tehais))):
            if s != my_seat and len(tehais[s]) > 0 and "?" not in tehais[s]:
                for t_str in tehais[s]:
                    c_idx = compact_idx_from_34(tile34_from_str(t_str), 4)
                    if c_idx is not None:
                        actual_indices.add(c_idx)

        win_prob_str = ""
        win_prob_val = None
        if ev.get("type") == "hora" and winner != my_seat and win_pai:
            w_idx = compact_idx_from_34(tile34_from_str(win_pai), 4)
            if w_idx is not None:
                actual_indices.add(w_idx)
                win_prob_val = self.last_opp_read_probs[w_idx] * 100.0
                win_prob_str = f"当たり牌[{win_pai}]危険予測: {win_prob_val:.1f}% / "

        if not actual_indices:
            return

        ranked = sorted(range(34), key=lambda i: self.last_opp_read_probs[i], reverse=True)
        top8 = set(ranked[:8])
        hits = len(top8 & actual_indices)
        precision = (hits / 8.0) * 100.0
        top3_tiles = ",".join(tile_str_from_compact(i, 4) for i in ranked[:3])

        base_engine.notify(
            "info",
            "他家手牌読み 答え合わせ (4P)",
            f"{win_prob_str}上位予測一致: {hits}/8種 ({precision:.0f}%) [予測上位: {top3_tiles}]",
            id="opp-read-4p"
        )

        log_path = os.path.join(BASE_DIR, "auto_learn", "opp_read_log.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps({
                    "mode": "4p",
                    "type": ev.get("type"),
                    "win_pai": win_pai if winner != my_seat else None,
                    "win_pai_prob": round(win_prob_val, 2) if win_prob_val is not None else None,
                    "top8_hits": hits,
                    "top3_pred": top3_tiles,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def build_logits(self):
        actor = self.state.actor_id
        hand_tiles = []
        for idx34, cnt in enumerate(self.state.hand34):
            base = TILE_TYPES[idx34]
            aka_cnt = 0
            if idx34 in (4, 13, 22):
                suit = "mps"[idx34 // 9]
                aka_cnt = self.state.aka_in_hand.get(suit, 0)
            for k in range(cnt):
                if k < aka_cnt:
                    hand_tiles.append(f"5{'mps'[idx34 // 9]}r")
                else:
                    hand_tiles.append(base)

        waits_34 = []
        if sum(self.state.hand34) % 3 == 1:
            if base_engine._shanten_with_melds(self.state.hand34, len(self.state.melds)) == 0:
                waits_34 = base_engine._tenpai_waits(self.state.hand34, len(self.state.melds))
        elif sum(self.state.hand34) % 3 == 2:
            for i in range(34):
                if self.state.hand34[i] > 0:
                    t = list(self.state.hand34)
                    t[i] -= 1
                    if base_engine._shanten_with_melds(t, len(self.state.melds)) == 0:
                        for w in base_engine._tenpai_waits(t, len(self.state.melds)):
                            if w not in waits_34:
                                waits_34.append(w)

        seats_rel = []
        for offset in range(4):
            s = (actor + offset) % 4
            meld_tiles = []
            if s == actor:
                for m in self.state.melds:
                    meld_tiles.extend(m.get("tiles", []))
            seats_rel.append({
                "discards": list(self.state.rivers[s]),
                "melds": meld_tiles,
                "riichi": (s == actor and self.state.reach_accepted) or (self.riichi_tiles[s] is not None),
                "riichi_tile": self.riichi_tiles[s],
                "score": self.scores[s],
                "kita": 0
            })

        wind_map = {"E": 0, "S": 1, "W": 2, "N": 3}
        obs = encode_observation(
            num_players=4,
            hand_tiles=hand_tiles,
            drawn_tile=self.drawn_tile,
            waits_34=waits_34,
            is_tenpai=len(waits_34) > 0,
            dora_indicators=list(getattr(self.state, "dora_markers", [])),
            seats_rel=seats_rel,
            last_discard=self.last_discard,
            round_wind=wind_map.get(self.state.round_wind_mjai(), 0),
            seat_wind=wind_map.get(self.state.seat_wind_mjai(), 0),
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
            turn_count=self.turn_count,
            is_dealer=(self.state.seat_wind_mjai() == "E"),
            kyoku_index=self.kyoku_index,
            self_riichi=self.state.reach_declared
        )
        tensor_x = torch.tensor([obs], dtype=torch.float32, device=device).view(1, 39, 34)
        with torch.no_grad():
            p_logits, r_logits = model(tensor_x, return_hand_read=True)
            self.last_opp_read_probs = torch.sigmoid(r_logits)[0].cpu().tolist()
            return p_logits[0].cpu().tolist()

    def decide_tsumo(self, last_event):
        actor = self.state.actor_id
        drawn = last_event["pai"]
        drawn_idx = base_engine._to34(drawn)
        hand14 = list(self.state.hand34)

        res = base_engine._hand_value(self.state, hand14, self.state.aka_in_hand, drawn_idx, is_tsumo=True)
        if res.cost is not None:
            return {"type": "hora", "actor": actor, "target": actor, "pai": drawn}

        if self.state.reach_accepted:
            return {"type": "dahai", "actor": actor, "pai": drawn, "tsumogiri": True}

        logits = self.build_logits()
        candidates = []

        if self.state.is_closed() and not self.state.reach_declared and self.state.left_tiles >= 4:
            for idx in range(34):
                if hand14[idx] > 0:
                    t = list(hand14)
                    t[idx] -= 1
                    if base_engine._shanten_with_melds(t, len(self.state.melds)) == 0:
                        candidates.append((logits[37], {"type": "reach", "actor": actor}))
                        break

        for idx in range(34):
            if hand14[idx] == 0:
                continue
            pai = base_engine._from34(idx)
            if idx in (4, 13, 22):
                suit = "mps"[idx // 9]
                if self.state.aka_in_hand.get(suit, 0) > 0 and hand14[idx] - self.state.aka_in_hand.get(suit, 0) == 0:
                    pai = f"5{suit}r"
            candidates.append((logits[idx], {
                "type": "dahai",
                "actor": actor,
                "pai": pai,
                "tsumogiri": pai == drawn
            }))

        if not candidates:
            return base_engine.decide_after_own_tsumo(self.state, last_event)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def decide_response(self, last_event):
        rule_action = base_engine.decide_after_others_dahai(self.state, last_event)
        rtype = rule_action.get("type")
        if rtype == "hora":
            return rule_action
        if rtype in ("pon", "chi", "daiminkan"):
            logits = self.build_logits()
            pass_logit = logits[81]
            if rtype == "pon":
                act_logit = logits[41]
            elif rtype == "daiminkan":
                idx34 = tile34_from_str(last_event["pai"])
                act_logit = logits[42 + idx34] if idx34 >= 0 else -1e9
            else:
                p34 = tile34_from_str(last_event["pai"])
                c34 = sorted([tile34_from_str(c) for c in rule_action.get("consumed", [])])
                if len(c34) == 2 and p34 < c34[0]:
                    act_logit = logits[38]
                elif len(c34) == 2 and c34[0] < p34 < c34[1]:
                    act_logit = logits[39]
                else:
                    act_logit = logits[40]
            if act_logit > pass_logit:
                return rule_action
            return {"type": "none"}
        return {"type": "none"}

def announce_status():
    state_path = os.path.join(BASE_DIR, "auto_learn", "train_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                info = json.load(f).get("4p", {})
            acc = info.get("best_acc", 0.0) * 100
            r_acc = info.get("opp_read_acc", 0.0) * 100
            base_engine.notify("success", "Pure Neural AI (4P) 起動", f"打牌一致率: {acc:.1f}% / 手牌読み精度: {r_acc:.1f}%", id="pure-ai-4p")
            return
        except Exception:
            pass
    base_engine.notify("info", "Pure Neural AI (4P) 起動", f"純正ResNet脳 ({model_source}) 稼働中", id="pure-ai-4p")

def main():
    session = PureNeuralSession4P()
    announce_status()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            events = json.loads(line)
            if not events:
                resp = json.dumps({"type": "none"}, separators=(",", ":"))
            else:
                for ev in events:
                    session.observe(ev)
                last = events[-1]
                t = last.get("type")
                actor = session.state.actor_id
                if t == "tsumo" and last.get("actor") == actor:
                    action = session.decide_tsumo(last)
                elif t == "dahai" and last.get("actor") != actor:
                    action = session.decide_response(last)
                elif t == "kakan" and last.get("actor") != actor:
                    action = base_engine.decide_after_others_kakan(session.state, last)
                else:
                    action = {"type": "none"}
                resp = json.dumps(action, separators=(",", ":"))
        except Exception as e:
            sys.stderr.write(f"bot error: {e}\n")
            sys.stderr.flush()
            resp = json.dumps({"type": "none"}, separators=(",", ":"))
        sys.stdout.write(resp + "\n")
        sys.stdout.flush()
        try:
            if any(ev.get("type") == "end_game" for ev in json.loads(line)):
                break
        except Exception:
            pass

if __name__ == "__main__":
    main()