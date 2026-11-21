import json
import math
import os
import sys

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
    tile_str_from_compact,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, model_source = load_pretrained_resnet(BASE_DIR, "3p", device)
model.eval()


def softmax_probs(scored):
    if not scored:
        return []
    m = max(x[0] for x in scored)
    exps = [math.exp(max(-20.0, min(20.0, x[0] - m))) for x in scored]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def format_overlay_meta(scored, title="Akagi · Local"):
    if not scored:
        return None
    ordered = sorted(scored, key=lambda x: x[0], reverse=True)
    probs = softmax_probs(ordered)
    items = []
    for (_, act), prob in zip(ordered[:3], probs[:3]):
        t = act.get("type", "none")
        pct = f"{round(prob * 100)}%"
        if t == "dahai":
            items.append({
                "label": "Discard",
                "pais": [act["pai"]],
                "prob": round(prob, 6),
                "type": "dahai",
                "value": pct,
            })
        elif t == "reach":
            items.append({
                "label": "Riichi",
                "pais": [act["pai"]] if act.get("pai") else [],
                "prob": round(prob, 6),
                "type": "reach",
                "value": pct,
            })
        elif t == "kita":
            items.append({
                "label": "Kita",
                "pais": ["N"],
                "prob": round(prob, 6),
                "type": "kita",
                "value": pct,
            })
        elif t == "pon":
            items.append({
                "label": "Pon",
                "pais": [act["pai"]] + list(act.get("consumed", [])),
                "prob": round(prob, 6),
                "type": "pon",
                "value": pct,
            })
        elif t in ("daiminkan", "ankan", "kakan"):
            pais = ([act["pai"]] if act.get("pai") else []) + list(act.get("consumed", []))
            items.append({
                "label": "Kan",
                "pais": pais,
                "prob": round(prob, 6),
                "type": t,
                "value": pct,
            })
        elif t == "hora":
            items.append({
                "label": "Hora",
                "prob": round(prob, 6),
                "type": "hora",
                "value": pct,
            })
        elif t == "none":
            items.append({
                "label": "Pass",
                "prob": round(prob, 6),
                "type": "none",
                "value": pct,
            })
    if not items:
        return None
    return {"show": {"title": title, "items": items}}


class Adaptive3PSession:
    def __init__(self):
        self.state = base_engine.State(actor_id=base_engine._initial_actor_id())
        self.last_discard = None
        self.riichi_tiles = [None] * 3
        self.scores = [35000] * 3
        self.kita_counts = [0] * 3
        self.kyoku_index = 0
        self.turn_count = 0
        self.last_opp_read_probs = None

    def observe(self, ev):
        etype = ev.get("type")
        if etype == "start_kyoku":
            self.last_discard = None
            self.riichi_tiles = [None] * 3
            self.kita_counts = [0] * 3
            self.last_opp_read_probs = None
            self.kyoku_index = max(0, ev.get("kyoku", 1) - 1)
            self.turn_count = 0
            if "scores" in ev and len(ev["scores"]) >= 3:
                self.scores = list(ev["scores"][:3])
        elif etype == "tsumo":
            if ev.get("actor") == self.state.actor_id:
                self.turn_count += 1
        elif etype == "dahai":
            actor = ev.get("actor")
            pai = ev.get("pai")
            self.last_discard = pai
            if actor is not None and 0 <= actor < 3:
                if actor in getattr(self.state, "reach_declared_seats", set()) and self.riichi_tiles[actor] is None:
                    self.riichi_tiles[actor] = pai
        elif etype in ("kita", "nukidora"):
            actor = ev.get("actor")
            if actor is not None and 0 <= actor < 3:
                self.kita_counts[actor] += 1
        elif etype == "reach_accepted":
            actor = ev.get("actor")
            if actor is not None and 0 <= actor < 3:
                self.scores[actor] -= 1000
        elif etype in ("hora", "ryukyoku"):
            self._notify_opp_read(ev)
        self.state.consume(ev)

    def _notify_opp_read(self, ev):
        if self.last_opp_read_probs is None:
            return
        my_seat = self.state.actor_id
        winner = ev.get("actor")
        win_pai = ev.get("pai")
        tehais = ev.get("tehais", [])

        actual = set()
        for s in range(min(3, len(tehais))):
            if s != my_seat and len(tehais[s]) > 0 and "?" not in tehais[s]:
                for t_str in tehais[s]:
                    c_idx = compact_idx_from_34(tile34_from_str(t_str), 3)
                    if c_idx is not None:
                        actual.add(c_idx)

        win_str = ""
        if ev.get("type") == "hora" and winner != my_seat and win_pai:
            w_idx = compact_idx_from_34(tile34_from_str(win_pai), 3)
            if w_idx is not None:
                actual.add(w_idx)
                win_str = f"当たり牌[{win_pai}]危険予測: {self.last_opp_read_probs[w_idx] * 100.0:.1f}% / "

        if not actual:
            return

        ranked = sorted(range(27), key=lambda i: self.last_opp_read_probs[i], reverse=True)
        hits = len(set(ranked[:7]) & actual)
        top3 = ",".join(tile_str_from_compact(i, 3) for i in ranked[:3])
        base_engine.notify(
            "info",
            "他家手牌読み 答え合わせ (3P)",
            f"{win_str}上位予測一致: {hits}/7種 ({(hits / 7.0) * 100.0:.0f}%) [予測上位: {top3}]",
            id="opp-read-3p",
        )

    def _build_logits(self):
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
        for offset in range(3):
            s = (actor + offset) % 3
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
                "kita": self.kita_counts[s],
            })

        wind_map = {"E": 0, "S": 1, "W": 2, "N": 3}
        drawn_for_obs = self.state.last_self_tsumo or None
        obs = encode_observation(
            num_players=3,
            hand_tiles=hand_tiles,
            drawn_tile=drawn_for_obs,
            waits_34=waits_34,
            is_tenpai=len(waits_34) > 0,
            dora_indicators=list(getattr(self.state, "dora_indicators", [])),
            seats_rel=seats_rel,
            last_discard=self.last_discard,
            round_wind=wind_map.get(self.state.round_wind_mjai(), 0),
            seat_wind=wind_map.get(self.state.seat_wind_mjai(), 0),
            honba=self.state.honba,
            riichi_sticks=self.state.kyotaku,
            turn_count=self.turn_count,
            is_dealer=(self.state.seat_wind_mjai() == "E"),
            kyoku_index=self.kyoku_index,
            self_riichi=self.state.reach_declared,
        )
        tensor_x = torch.tensor([obs], dtype=torch.float32, device=device).view(1, 37, 27)
        with torch.no_grad():
            p_logits, r_logits = model(tensor_x, return_hand_read=True)
            self.last_opp_read_probs = torch.sigmoid(r_logits)[0].cpu().tolist()
            return p_logits[0].cpu().tolist()

    def _pai_for_index(self, idx, hand14):
        pai = base_engine._from34(idx)
        if idx in (4, 13, 22):
            suit = "mps"[idx // 9]
            aka_cnt = self.state.aka_in_hand.get(suit, 0)
            if aka_cnt > 0 and hand14[idx] - aka_cnt == 0:
                pai = f"5{suit}r"
        return pai

    def _can_kita_safely(self, hand14):
        if hand14[30] <= 0 or self.state.reach_declared or self.state.reach_accepted:
            return False
        # 北を抜いたあとの13枚形（北を1枚減らす）で向聴数を確認
        after_kita13 = list(hand14)
        after_kita13[30] -= 1
        # 打牌14枚->13枚時の最良向聴数と比較
        best_dahai_sh = 99
        for idx in range(34):
            if hand14[idx] > 0:
                t = list(hand14)
                t[idx] -= 1
                sh = base_engine._shanten_with_melds(t, len(self.state.melds))
                if sh < best_dahai_sh:
                    best_dahai_sh = sh
        kita_sh = base_engine._shanten_with_melds(after_kita13, len(self.state.melds))
        # 北を抜いて向聴数が悪化しない場合のみ許可
        return kita_sh <= best_dahai_sh

    def decide_tsumo(self, last_event):
        actor = self.state.actor_id
        drawn = last_event["pai"]
        drawn_idx = base_engine._to34(drawn)
        hand14 = list(self.state.hand34)

        res = base_engine._hand_value(self.state, hand14, self.state.aka_in_hand, drawn_idx, is_tsumo=True)
        if res.cost is not None:
            hora_act = {"type": "hora", "actor": actor, "target": actor, "pai": drawn}
            dahai_act = {"type": "dahai", "actor": actor, "pai": drawn, "tsumogiri": True}
            return hora_act, format_overlay_meta([(10.0, hora_act), (0.0, dahai_act)])

        if self.state.reach_accepted:
            act = {"type": "dahai", "actor": actor, "pai": drawn, "tsumogiri": True}
            return act, format_overlay_meta([(10.0, act)])

        logits = self._build_logits()
        scored = []

        if self._can_kita_safely(hand14):
            scored.append((logits[59], {"type": "kita", "actor": actor, "pai": "N"}))

        if self.state.is_closed() and not self.state.reach_declared and self.state.left_tiles >= 4:
            for idx in range(34):
                if hand14[idx] == 0:
                    continue
                t = list(hand14)
                t[idx] -= 1
                if base_engine._shanten_with_melds(t, len(self.state.melds)) == 0:
                    scored.append((logits[27], {"type": "reach", "actor": actor}))
                    break

        for idx in range(34):
            if hand14[idx] == 0:
                continue
            c_idx = compact_idx_from_34(idx, 3)
            if c_idx is None:
                continue
            pai = self._pai_for_index(idx, hand14)
            scored.append((logits[c_idx], {
                "type": "dahai",
                "actor": actor,
                "pai": pai,
                "tsumogiri": pai == drawn,
            }))

        if not scored:
            act = base_engine.decide_after_own_tsumo(self.state, last_event)
            return act, format_overlay_meta([(1.0, act)])

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[0][1]
        return chosen, format_overlay_meta(scored)

    def decide_response(self, last_event):
        rule_action = base_engine.decide_after_others_dahai(self.state, last_event)
        rtype = rule_action.get("type")
        pass_act = {"type": "none"}

        if rtype == "hora":
            return rule_action, format_overlay_meta([(10.0, rule_action), (0.0, pass_act)])

        if rtype in ("pon", "daiminkan"):
            logits = self._build_logits()
            pass_logit = logits[58]
            if rtype == "pon":
                act_logit = logits[28]
            else:
                c_idx = compact_idx_from_34(tile34_from_str(last_event["pai"]), 3)
                act_logit = logits[29 + c_idx] if c_idx is not None else -1e9
            chosen = rule_action if act_logit > pass_logit else pass_act
            return chosen, format_overlay_meta([(act_logit, rule_action), (pass_logit, pass_act)])

        return pass_act, None


def announce_status():
    state_path = os.path.join(BASE_DIR, "auto_learn", "train_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                info = json.load(f).get("3p", {})
            acc = info.get("best_acc", 0.0) * 100
            r_acc = info.get("opp_read_acc", 0.0) * 100
            base_engine.notify("success", "Pure Neural AI (3P) 起動", f"打牌一致率: {acc:.1f}% / 手牌読み精度: {r_acc:.1f}%", id="pure-ai-3p")
            return
        except Exception:
            pass
    base_engine.notify("info", "Pure Neural AI (3P) 起動", f"純正ResNet脳 ({model_source}) 稼働中", id="pure-ai-3p")


def react(session: Adaptive3PSession, line: str) -> str:
    events = json.loads(line)
    if not events:
        return json.dumps({"type": "none"}, separators=(",", ":"))
    for ev in events:
        session.observe(ev)
    last = events[-1]
    t = last.get("type")
    actor = session.state.actor_id
    meta = None

    if t == "tsumo" and last.get("actor") == actor:
        action, meta = session.decide_tsumo(last)
    elif t == "dahai" and last.get("actor") != actor:
        action, meta = session.decide_response(last)
    elif t == "kakan" and last.get("actor") != actor:
        action = base_engine.decide_after_others_kakan(session.state, last)
        if action.get("type") == "hora":
            meta = format_overlay_meta([(10.0, action), (0.0, {"type": "none"})])
    else:
        action = {"type": "none"}

    if meta is not None:
        action = dict(action)
        action["meta"] = meta
    return json.dumps(action, separators=(",", ":"))


def main() -> None:
    session = Adaptive3PSession()
    announce_status()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            resp = react(session, line)
        except Exception as e:
            sys.stderr.write(f"bot error: {e}\n")
            sys.stderr.flush()
            resp = json.dumps({"type": "none"}, separators=(",", ":"))
        sys.stdout.write(resp + "\n")
        sys.stdout.flush()
        try:
            evs = json.loads(line)
        except Exception:
            continue
        if any(ev.get("type") == "end_game" for ev in evs):
            break


if __name__ == "__main__":
    main()
