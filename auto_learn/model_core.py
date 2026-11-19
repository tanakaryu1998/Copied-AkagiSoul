import os
import torch
import torch.nn as nn

TILE_TYPES = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C"
]
TILE_TO_IDX = {t: i for i, t in enumerate(TILE_TYPES)}

def normalize_tile(tile_str):
    if tile_str.endswith("r"):
        return tile_str[:-1]
    return tile_str

class MahjongMultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(250, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.policy_head = nn.Linear(384, 34)
        self.danger_head = nn.Linear(384, 34)

    def forward(self, x):
        h = self.backbone(x)
        return self.policy_head(h), self.danger_head(h)

def build_feature_vector(
    hand,
    my_discards,
    all_discards,
    opp_tedashi,
    opp_melds,
    dora_vec,
    last_discard_idx,
    bakaze,
    jikaze,
    turn_count,
    riichi_flags,
    kita_counts,
    num_players
):
    hand_vec = [0.0] * 34
    for t in hand:
        base = normalize_tile(t)
        if base in TILE_TO_IDX:
            hand_vec[TILE_TO_IDX[base]] += 0.25
    my_disc_vec = [min(c * 0.25, 1.0) for c in my_discards]
    all_disc_vec = [min(c * 0.25, 1.0) for c in all_discards]
    tedashi_vec = [min(c * 0.25, 1.0) for c in opp_tedashi]
    meld_vec = [min(c * 0.25, 1.0) for c in opp_melds]
    last_disc_vec = [0.0] * 34
    if 0 <= last_discard_idx < 34:
        last_disc_vec[last_discard_idx] = 1.0
    r_padded = (riichi_flags + [0.0] * 4)[:4]
    k_padded = [min(c * 0.25, 1.0) for c in (kita_counts + [0] * 3)[:3]]
    context_vec = [
        1.0 if num_players == 3 else 0.0,
        1.0 if bakaze == 27 else 0.0,
        1.0 if bakaze == 28 else 0.0,
        1.0 if jikaze == 27 else 0.0,
        min(turn_count / 18.0, 1.0)
    ] + r_padded + k_padded
    return (
        hand_vec
        + my_disc_vec
        + all_disc_vec
        + tedashi_vec
        + meld_vec
        + dora_vec
        + last_disc_vec
        + context_vec
    )

def extract_features_from_engine_state(state, num_players, opp_tedashi, opp_melds, last_discard_idx):
    hand_vec = [min(c * 0.25, 1.0) for c in state.hand34]
    my_disc_vec = [0.0] * 34
    all_disc_vec = [0.0] * 34
    for seat_idx, river in enumerate(state.rivers):
        for t in river:
            base = normalize_tile(t)
            if base in TILE_TO_IDX:
                idx = TILE_TO_IDX[base]
                all_disc_vec[idx] = min(all_disc_vec[idx] + 0.25, 1.0)
                if seat_idx == state.actor_id:
                    my_disc_vec[idx] = min(my_disc_vec[idx] + 0.25, 1.0)
    tedashi_vec = [min(c * 0.25, 1.0) for c in opp_tedashi]
    meld_vec = [min(c * 0.25, 1.0) for c in opp_melds]
    dora_vec = [0.0] * 34
    for dm in getattr(state, "dora_markers", []):
        base = normalize_tile(dm)
        if base in TILE_TO_IDX:
            dora_vec[TILE_TO_IDX[base]] = 1.0
    last_disc_vec = [0.0] * 34
    if 0 <= last_discard_idx < 34:
        last_disc_vec[last_discard_idx] = 1.0
    bakaze = TILE_TO_IDX.get(state.round_wind_mjai(), 27)
    jikaze = TILE_TO_IDX.get(state.seat_wind_mjai(), 27)
    turn_count = len(state.rivers[state.actor_id])
    riichi_flags = [0.0] * 4
    for s in getattr(state, "reach_accepted_seats", set()):
        if 0 <= s < 4:
            riichi_flags[s] = 1.0
    kita_counts = [0.0] * 3
    if hasattr(state, "kita"):
        for s in range(min(3, len(state.kita))):
            kita_counts[s] = min(len(state.kita[s]) * 0.25, 1.0)
    context_vec = [
        1.0 if num_players == 3 else 0.0,
        1.0 if bakaze == 27 else 0.0,
        1.0 if bakaze == 28 else 0.0,
        1.0 if jikaze == 27 else 0.0,
        min(turn_count / 18.0, 1.0)
    ] + riichi_flags + kita_counts
    return (
        hand_vec
        + my_disc_vec
        + all_disc_vec
        + tedashi_vec
        + meld_vec
        + dora_vec
        + last_disc_vec
        + context_vec
    )

class HybridNeuralEvaluator:
    def __init__(self, base_dir, mode):
        self.mode = mode
        self.num_players = 3 if mode == "3p" else 4
        self.model_path = os.path.join(base_dir, "auto_learn", "models", f"active_model_{mode}.pt")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = MahjongMultiTaskNet().to(self.device)
        self.is_loaded = False
        self.opp_tedashi = [0] * 34
        self.opp_melds = [0] * 34
        self.last_discard_idx = -1
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.is_loaded = True
            except Exception:
                self.is_loaded = False
        self.model.eval()

    def observe_event(self, event, my_actor_id):
        etype = event.get("type")
        if etype == "start_kyoku":
            self.opp_tedashi = [0] * 34
            self.opp_melds = [0] * 34
            self.last_discard_idx = -1
        elif etype == "dahai":
            actor = event.get("actor")
            pai = event.get("pai")
            if pai:
                idx = TILE_TO_IDX.get(normalize_tile(pai), -1)
                if idx >= 0:
                    self.last_discard_idx = idx
                    if actor != my_actor_id and not event.get("tsumogiri", False):
                        self.opp_tedashi[idx] += 1
        elif etype in ("pon", "chi", "kan", "daiminkan", "kakan", "ankan"):
            actor = event.get("actor")
            if actor != my_actor_id:
                pai = event.get("pai")
                if pai:
                    idx = TILE_TO_IDX.get(normalize_tile(pai), -1)
                    if idx >= 0:
                        self.opp_melds[idx] += 1
                for c in event.get("consumed", []):
                    c_idx = TILE_TO_IDX.get(normalize_tile(c), -1)
                    if c_idx >= 0:
                        self.opp_melds[c_idx] += 1

    def evaluate_situation(self, state):
        if not self.is_loaded:
            return [0.0] * 34, [0.0] * 34
        features = extract_features_from_engine_state(
            state,
            self.num_players,
            self.opp_tedashi,
            self.opp_melds,
            self.last_discard_idx
        )
        tensor_x = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_logits, danger_logits = self.model(tensor_x)
            danger_probs = torch.sigmoid(danger_logits)[0].cpu().tolist()
            policy_scores = policy_logits[0].cpu().tolist()
        return policy_scores, danger_probs