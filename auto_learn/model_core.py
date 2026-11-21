import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

TILE_TYPES = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C"
]
TILE_TO_IDX = {t: i for i, t in enumerate(TILE_TYPES)}

TILE34_TO_COMPACT = [
    0, None, None, None, None, None, None, None, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26
]

COMPACT_TO_TILE34 = [
    0, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33
]

def normalize_tile(tile_str):
    if tile_str.endswith("r"):
        return tile_str[:-1]
    return tile_str

def tile34_from_str(tile_str):
    return TILE_TO_IDX.get(normalize_tile(tile_str), -1)

def compact_idx_from_34(t34, num_players):
    if t34 < 0 or t34 >= 34:
        return None
    if num_players == 3:
        return TILE34_TO_COMPACT[t34]
    return t34

def tile_str_from_compact(c_idx, num_players):
    if num_players == 3:
        if 0 <= c_idx < len(COMPACT_TO_TILE34):
            return TILE_TYPES[COMPACT_TO_TILE34[c_idx]]
        return "?"
    if 0 <= c_idx < len(TILE_TYPES):
        return TILE_TYPES[c_idx]
    return "?"

def next_dora_34(ind34, num_players):
    if num_players == 3:
        if ind34 == 0:
            return 8
        if ind34 == 8:
            return 0
    if 0 <= ind34 <= 8:
        return 0 if ind34 == 8 else ind34 + 1
    if 9 <= ind34 <= 17:
        return 9 if ind34 == 17 else ind34 + 1
    if 18 <= ind34 <= 26:
        return 18 if ind34 == 26 else ind34 + 1
    if 27 <= ind34 <= 30:
        return 27 if ind34 == 30 else ind34 + 1
    if 31 <= ind34 <= 33:
        return 31 if ind34 == 33 else ind34 + 1
    return ind34

class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, bias=True)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y + x)

class AkagiResNet(nn.Module):
    def __init__(self, num_players, conv=64, blocks=3, fc=256):
        super().__init__()
        self.num_players = num_players
        self.tile_dim = 27 if num_players == 3 else 34
        self.channels = 37 if num_players == 3 else 39
        self.action_space = 60 if num_players == 3 else 82

        self.conv_in = nn.Conv1d(self.channels, conv, 3, padding=1, bias=True)
        self.res = nn.ModuleList([ResBlock1D(conv) for _ in range(blocks)])
        self.fc = nn.Linear(conv * self.tile_dim, fc)
        self.head = nn.Linear(fc, self.action_space)
        self.hand_read_head = nn.Linear(fc, self.tile_dim)

    def forward(self, x, return_hand_read=False):
        x = F.relu(self.conv_in(x))
        for r in self.res:
            x = r(x)
        x = torch.flatten(x, 1)
        feat = F.relu(self.fc(x))
        policy_logits = self.head(feat)
        if return_hand_read:
            read_logits = self.hand_read_head(feat)
            return policy_logits, read_logits
        return policy_logits

def load_pretrained_resnet(base_dir, mode, device):
    num_players = 3 if mode == "3p" else 4
    model = AkagiResNet(num_players).to(device)
    pt_path = os.path.join(base_dir, "auto_learn", "models", f"active_model_{mode}.pt")
    sf_path = os.path.join(base_dir, "auto_learn", "pretrained", f"akagi{num_players}p.safetensors")
    if os.path.exists(pt_path):
        sd = torch.load(pt_path, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        return model, "adaptive"
    if os.path.exists(sf_path):
        sd = load_file(sf_path, device=str(device))
        model.load_state_dict(sd, strict=False)
        return model, "pretrained"
    return model, "random"

def encode_observation(
    num_players,
    hand_tiles,
    drawn_tile,
    waits_34,
    is_tenpai,
    dora_indicators,
    seats_rel,
    last_discard,
    round_wind,
    seat_wind,
    honba,
    riichi_sticks,
    turn_count,
    is_dealer,
    kyoku_index,
    self_riichi
):
    t = 27 if num_players == 3 else 34
    c = 37 if num_players == 3 else 39
    buf = [0.0] * (c * t)
    ch = 0

    hand_counts = [0] * t
    aka_here = [False, False, False]
    for ts in hand_tiles:
        t34 = tile34_from_str(ts)
        idx = compact_idx_from_34(t34, num_players)
        if idx is not None:
            hand_counts[idx] += 1
        if ts == "5mr":
            aka_here[0] = True
        elif ts == "5pr":
            aka_here[1] = True
        elif ts == "5sr":
            aka_here[2] = True

    for thresh in (1, 2, 3, 4):
        for idx in range(t):
            if hand_counts[idx] >= thresh:
                buf[ch * t + idx] = 1.0
        ch += 1

    for k, a34 in enumerate((4, 13, 22)):
        if aka_here[k]:
            idx = compact_idx_from_34(a34, num_players)
            if idx is not None:
                buf[ch * t + idx] = 1.0
    ch += 1

    if drawn_tile:
        idx = compact_idx_from_34(tile34_from_str(drawn_tile), num_players)
        if idx is not None:
            buf[ch * t + idx] = 1.0
    ch += 1

    for w34 in waits_34:
        idx = compact_idx_from_34(w34, num_players)
        if idx is not None:
            buf[ch * t + idx] = 1.0
    ch += 1

    for dm in dora_indicators:
        idx = compact_idx_from_34(tile34_from_str(dm), num_players)
        if idx is not None:
            buf[ch * t + idx] += 0.25
    ch += 1

    for dm in dora_indicators:
        d34 = next_dora_34(tile34_from_str(dm), num_players)
        idx = compact_idx_from_34(d34, num_players)
        if idx is not None:
            buf[ch * t + idx] += 0.25
    ch += 1

    for seat in seats_rel:
        for d in seat["discards"]:
            idx = compact_idx_from_34(tile34_from_str(d), num_players)
            if idx is not None:
                buf[ch * t + idx] += 0.25
        ch += 1

    for seat in seats_rel:
        for m in seat["melds"]:
            idx = compact_idx_from_34(tile34_from_str(m), num_players)
            if idx is not None:
                buf[ch * t + idx] += 0.25
        ch += 1

    for seat in seats_rel:
        if seat["riichi"]:
            for i in range(t):
                buf[ch * t + i] = 1.0
        ch += 1

    for seat in seats_rel:
        rt = seat.get("riichi_tile")
        if rt:
            idx = compact_idx_from_34(tile34_from_str(rt), num_players)
            if idx is not None:
                buf[ch * t + idx] = 1.0
        ch += 1

    for seat in seats_rel:
        v = max(0.0, min(4.0, float(seat["score"]) / 50000.0))
        for i in range(t):
            buf[ch * t + i] = v
        ch += 1

    if last_discard:
        idx = compact_idx_from_34(tile34_from_str(last_discard), num_players)
        if idx is not None:
            buf[ch * t + idx] = 1.0
    ch += 1

    rw_idx = compact_idx_from_34(27 + min(3, max(0, round_wind)), num_players)
    if rw_idx is not None:
        buf[ch * t + rw_idx] = 1.0
    ch += 1

    sw_idx = compact_idx_from_34(27 + min(3, max(0, seat_wind)), num_players)
    if sw_idx is not None:
        buf[ch * t + sw_idx] = 1.0
    ch += 1

    scalars = [
        min(2.0, float(honba) / 4.0),
        min(2.0, float(riichi_sticks) / 4.0),
        min(1.0, float(turn_count) / 70.0),
        1.0 if is_tenpai else 0.0,
        1.0 if is_dealer else 0.0,
        min(1.0, float(kyoku_index) / 8.0),
        1.0 if self_riichi else 0.0
    ]
    for val in scalars:
        for i in range(t):
            buf[ch * t + i] = val
        ch += 1

    if num_players == 3:
        for seat in seats_rel:
            kv = min(1.0, float(seat.get("kita", 0)) / 4.0)
            for i in range(t):
                buf[ch * t + i] = kv
            ch += 1

    return buf