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

class MahjongPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(148, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 34)
        )

    def forward(self, x):
        return self.net(x)

def build_feature_vector(hand, my_discards, all_discards, dora_vec, bakaze, jikaze, turn_count, riichi_flags, kita_counts, num_players):
    hand_vec = [0.0] * 34
    for t in hand:
        base = normalize_tile(t)
        if base in TILE_TO_IDX:
            hand_vec[TILE_TO_IDX[base]] += 0.25
    my_disc_vec = [min(c * 0.25, 1.0) for c in my_discards]
    all_disc_vec = [min(c * 0.25, 1.0) for c in all_discards]
    r_padded = (riichi_flags + [0.0] * 4)[:4]
    k_padded = [min(c * 0.25, 1.0) for c in (kita_counts + [0] * 4)[:4]]
    context_vec = [
        1.0 if num_players == 3 else 0.0,
        1.0 if bakaze == 27 else 0.0,
        1.0 if bakaze == 28 else 0.0,
        1.0 if jikaze == 27 else 0.0,
        min(turn_count / 18.0, 1.0)
    ] + r_padded + k_padded
    return hand_vec + my_disc_vec + all_disc_vec + dora_vec + context_vec

def extract_features_from_engine_state(state, num_players):
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
    dora_vec = [0.0] * 34
    for dm in getattr(state, "dora_markers", []):
        base = normalize_tile(dm)
        if base in TILE_TO_IDX:
            dora_vec[TILE_TO_IDX[base]] = 1.0
    bakaze = TILE_TO_IDX.get(state.round_wind_mjai(), 27)
    jikaze = TILE_TO_IDX.get(state.seat_wind_mjai(), 27)
    turn_count = len(state.rivers[state.actor_id])
    riichi_flags = [0.0] * 4
    for s in getattr(state, "reach_accepted_seats", set()):
        if 0 <= s < 4:
            riichi_flags[s] = 1.0
    kita_counts = [0.0] * 4
    if hasattr(state, "kita"):
        for s in range(min(4, len(state.kita))):
            kita_counts[s] = min(len(state.kita[s]) * 0.25, 1.0)
    context_vec = [
        1.0 if num_players == 3 else 0.0,
        1.0 if bakaze == 27 else 0.0,
        1.0 if bakaze == 28 else 0.0,
        1.0 if jikaze == 27 else 0.0,
        min(turn_count / 18.0, 1.0)
    ] + riichi_flags + kita_counts
    return hand_vec + my_disc_vec + all_disc_vec + dora_vec + context_vec

class HybridNeuralEvaluator:
    def __init__(self, base_dir, mode):
        self.mode = mode
        self.num_players = 3 if mode == "3p" else 4
        self.model_path = os.path.join(base_dir, "auto_learn", "models", f"active_model_{mode}.pt")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = MahjongPolicyNet().to(self.device)
        self.is_loaded = False
        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.is_loaded = True
        self.model.eval()

    def score_tiles(self, state):
        if not self.is_loaded:
            return [0.0] * 34
        features = extract_features_from_engine_state(state, self.num_players)
        tensor_x = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)[0]
        return logits.cpu().tolist()