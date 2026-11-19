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

class AdaptiveMjaiBot:
    def __init__(self, base_dir, mode):
        self.mode = mode
        self.num_players = 3 if mode == "3p" else 4
        model_path = os.path.join(base_dir, "auto_learn", "models", f"active_model_{mode}.pt")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = MahjongPolicyNet().to(self.device)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.eval()
        self.actor_id = 0
        self.hand = []
        self.my_discards = [0] * 34
        self.all_discards = [0] * 34
        self.dora_vec = [0.0] * 34
        self.riichi_flags = [0.0] * 4
        self.kita_counts = [0] * 4
        self.bakaze = 27
        self.jikaze = 27
        self.turn_count = 0

    def choose_discard(self):
        if not self.hand:
            return None
        features = build_feature_vector(
            self.hand,
            self.my_discards,
            self.all_discards,
            self.dora_vec,
            self.bakaze,
            self.jikaze,
            self.turn_count,
            self.riichi_flags,
            self.kita_counts,
            self.num_players
        )
        state_tensor = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(state_tensor)[0]
        best_tile = self.hand[0]
        best_score = -1e9
        for t in self.hand:
            idx = TILE_TO_IDX.get(normalize_tile(t), -1)
            score = logits[idx].item() if idx >= 0 else -1e9
            if score > best_score:
                best_score = score
                best_tile = t
        return best_tile

    def handle_event(self, event):
        etype = event.get("type")
        if etype == "start_game":
            self.actor_id = event.get("id", 0)
        elif etype == "start_kyoku":
            self.my_discards = [0] * 34
            self.all_discards = [0] * 34
            self.dora_vec = [0.0] * 34
            self.riichi_flags = [0.0] * 4
            self.kita_counts = [0] * 4
            self.turn_count = 0
            self.bakaze = TILE_TO_IDX.get(event.get("bakaze", "E"), 27)
            oya = event.get("oya", 0)
            self.jikaze = 27 + ((self.actor_id - oya) % self.num_players)
            dora_marker = event.get("dora_marker")
            if dora_marker:
                base = normalize_tile(dora_marker)
                if base in TILE_TO_IDX:
                    self.dora_vec[TILE_TO_IDX[base]] = 1.0
            tehais = event.get("tehais", [])
            if len(tehais) > self.actor_id:
                self.hand = list(tehais[self.actor_id])
        elif etype == "tsumo":
            if event.get("actor") == self.actor_id:
                self.turn_count += 1
                pai = event.get("pai")
                if pai:
                    self.hand.append(pai)
                chosen = self.choose_discard()
                if chosen:
                    return {"type": "dahai", "actor": self.actor_id, "pai": chosen, "tsumogiri": chosen == pai}
        elif etype == "dahai":
            actor = event.get("actor")
            pai = event.get("pai")
            if pai:
                base = normalize_tile(pai)
                if base in TILE_TO_IDX:
                    idx = TILE_TO_IDX[base]
                    self.all_discards[idx] += 1
                    if actor == self.actor_id:
                        self.my_discards[idx] += 1
                if actor == self.actor_id and pai in self.hand:
                    self.hand.remove(pai)
        elif etype == "reach":
            actor = event.get("actor")
            if actor is not None and 0 <= actor < 4:
                self.riichi_flags[actor] = 1.0
        elif etype == "nukidora":
            actor = event.get("actor")
            if actor is not None and 0 <= actor < 4:
                self.kita_counts[actor] += 1
                if actor == self.actor_id and "N" in self.hand:
                    self.hand.remove("N")
        elif etype in ("pon", "chi", "kan", "daiminkan", "kakan", "ankan"):
            if event.get("actor") == self.actor_id:
                for c in event.get("consumed", []):
                    if c in self.hand:
                        self.hand.remove(c)
                if etype in ("pon", "chi"):
                    chosen = self.choose_discard()
                    if chosen:
                        return {"type": "dahai", "actor": self.actor_id, "pai": chosen, "tsumogiri": False}
        return {"type": "none"}