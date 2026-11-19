import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUNDLED_SITE_PACKAGES = os.path.join(
    BASE_DIR, "runtime", "python", "aarch64-apple-darwin", "lib", "python3.12", "site-packages"
)
if os.path.exists(BUNDLED_SITE_PACKAGES) and BUNDLED_SITE_PACKAGES not in sys.path:
    sys.path.append(BUNDLED_SITE_PACKAGES)

sys.path.insert(0, os.path.join(BASE_DIR, "auto_learn"))

import base_engine
from model_core import HybridNeuralEvaluator

evaluator = HybridNeuralEvaluator(BASE_DIR, "3p")
original_decide_tsumo = base_engine.decide_after_own_tsumo
original_react = base_engine.react

def hooked_react(state, events_json):
    try:
        events = json.loads(events_json)
        for ev in events:
            evaluator.observe_event(ev, state.actor_id)
    except Exception:
        pass
    return original_react(state, events_json)

def neural_guided_decide_tsumo(state, last_event):
    actor = state.actor_id
    drawn = last_event["pai"]
    drawn_idx = base_engine._to34(drawn)

    hand14 = list(state.hand34)
    res = base_engine._hand_value(state, hand14, state.aka_in_hand, drawn_idx, is_tsumo=True)
    if res.cost is not None:
        return {"type": "hora", "actor": actor, "target": actor, "pai": drawn}

    if state.reach_accepted:
        return {"type": "dahai", "actor": actor, "pai": drawn, "tsumogiri": True}

    if not evaluator.is_loaded:
        return original_decide_tsumo(state, last_event)

    policy_scores, danger_probs = evaluator.evaluate_situation(state)
    opp_riichi = len(getattr(state, "reach_accepted_seats", set())) > 0
    turn_count = len(state.rivers[actor])

    if state.is_closed() and not state.reach_declared and state.left_tiles >= 4:
        best_riichi = None
        for idx in range(34):
            if hand14[idx] == 0:
                continue
            test = list(hand14)
            test[idx] -= 1
            if base_engine._shanten_with_melds(test, len(state.melds)) != 0:
                continue
            waits = base_engine._tenpai_waits(test, len(state.melds))
            if len(waits) >= 2:
                risk_penalty = danger_probs[idx] * (12.0 if opp_riichi else 3.0)
                score = len(waits) * 2.0 + policy_scores[idx] - risk_penalty
                if best_riichi is None or score > best_riichi[0]:
                    best_riichi = (score, idx)
        if best_riichi is not None:
            return {"type": "reach", "actor": actor}

    best_choice = None
    for idx in range(34):
        if hand14[idx] == 0:
            continue
        test = list(hand14)
        test[idx] -= 1
        sh = base_engine._shanten_with_melds(test, len(state.melds))
        uke = base_engine._effective_ukeire(state, test)
        pai = base_engine._from34(idx)
        if idx in (4, 13, 22):
            suit = "mps"[idx // 9]
            if state.aka_in_hand.get(suit, 0) > 0 and hand14[idx] - state.aka_in_hand.get(suit, 0) == 0:
                pai = f"5{suit}r"

        if opp_riichi:
            risk_weight = 6.0 if sh == 0 else 30.0
        elif turn_count >= 12:
            risk_weight = 4.0 if sh == 0 else 15.0
        else:
            risk_weight = 2.0

        utility = policy_scores[idx] * 10.0 + uke - (danger_probs[idx] * risk_weight)
        effective_sh = sh if not (opp_riichi and sh >= 2 and danger_probs[idx] >= 0.45) else sh + 1
        hybrid_rank = (effective_sh, -utility, pai)
        if best_choice is None or hybrid_rank < best_choice:
            best_choice = hybrid_rank

    if best_choice is not None:
        _, _, pai = best_choice
        return {
            "type": "dahai",
            "actor": actor,
            "pai": pai,
            "tsumogiri": pai == drawn,
        }
    return original_decide_tsumo(state, last_event)

def announce_status():
    state_path = os.path.join(BASE_DIR, "auto_learn", "train_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                info = json.load(f).get("3p", {})
            acc = info.get("best_acc", 0.0) * 100
            d_acc = info.get("danger_read_acc", 0.0) * 100
            base_engine.notify("success", "Local Adaptive AI (3P) 起動", f"打牌一致率: {acc:.1f}% / 待ち読み精度: {d_acc:.1f}%", id="local-ai-3p")
            return
        except Exception:
            pass
    base_engine.notify("info", "Local Adaptive AI (3P) 起動", "手牌読み・押し引きモデル稼働中", id="local-ai-3p")

if __name__ == "__main__":
    base_engine.react = hooked_react
    base_engine.decide_after_own_tsumo = neural_guided_decide_tsumo
    announce_status()
    base_engine.main()