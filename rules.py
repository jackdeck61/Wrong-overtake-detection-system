"""rules.py - turns vehicle tracks into overtaking events + violations.

tracks format (produced by main.run_tracking):
{ "7": [ {"f": 12, "box": [x1,y1,x2,y2], "cls": "car"}, ... ], ... }
"""
import json
from collections import Counter
from itertools import permutations

import numpy as np
from shapely.geometry import LineString

# Pixel values assume ~1280x720 video. Tune these for your clip.
DEFAULT_PARAMS = {
    "min_track_len": 30,   # ignore tracks shorter than this many frames (noise / ID switches)
    "min_overlap": 20,     # two vehicles must be visible together for this many frames
    "margin": 10,          # px of forward gap needed to say "clearly behind / clearly ahead"
    "window": 15,          # frames around the pass moment used to judge the side
    "max_lateral": 250,    # if vehicles are further apart sideways than this, not an overtake
    "side_min": 20,        # if sideways gap is smaller than this, side is ambiguous -> no flag
}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    params = dict(DEFAULT_PARAMS)
    params.update(cfg.get("params", {}))
    cfg["params"] = params
    cfg.setdefault("drive_side", "left")      # India = left-hand traffic
    cfg.setdefault("line_type", "solid")      # "solid" or "broken"
    cfg.setdefault("center_line", [])
    return cfg


def _axes(cfg):
    """forward unit vector f, and 'right-hand' unit vector r (image y points down)."""
    (x1, y1), (x2, y2) = cfg["direction"]
    f = np.array([x2 - x1, y2 - y1], float)
    f /= np.linalg.norm(f)
    r = np.array([-f[1], f[0]])
    return f, r


def _smooth(a, k=5):
    if len(a) < k:
        return a
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    return np.vstack([np.convolve(ap[:, i], np.ones(k) / k, mode="valid")
                      for i in range(a.shape[1])]).T


def _prepare(tracks, f, r, P):
    out = {}
    for tid, pts in tracks.items():
        if len(pts) < P["min_track_len"]:
            continue
        frames = np.array([p["f"] for p in pts])
        # ground position = bottom-centre of the box
        xy = np.array([[(p["box"][0] + p["box"][2]) / 2, p["box"][3]] for p in pts], float)
        xy = _smooth(xy)
        fwd, lat = xy @ f, xy @ r
        if fwd[-1] - fwd[0] <= 0:      # not moving in the travel direction -> skip
            continue
        cls = Counter(p["cls"] for p in pts).most_common(1)[0][0]
        out[tid] = dict(frames=frames, xy=xy, fwd=fwd, lat=lat, cls=cls)
    return out


def detect_events(tracks, cfg):
    """Return a list of overtaking events. 'violations' is empty for legal overtakes."""
    P = cfg["params"]
    f, r = _axes(cfg)
    T = _prepare(tracks, f, r, P)
    line = LineString(cfg["center_line"]) if len(cfg["center_line"]) >= 2 else None
    events = []

    for ida, idb in permutations(T, 2):          # B tries to overtake A
        A, B = T[ida], T[idb]
        common = np.intersect1d(A["frames"], B["frames"])
        if len(common) < P["min_overlap"]:
            continue
        ia = np.searchsorted(A["frames"], common)
        ib = np.searchsorted(B["frames"], common)
        d = B["fwd"][ib] - A["fwd"][ia]           # >0 means B is ahead of A
        l = B["lat"][ib] - A["lat"][ia]           # >0 means B is to the right of A

        behind = np.where(d < -P["margin"])[0]
        ahead = np.where(d > P["margin"])[0]
        if len(behind) == 0 or len(ahead) == 0 or behind[0] > ahead[-1]:
            continue                               # B was never behind-then-ahead

        k = behind[0] + int(np.argmax(d[behind[0]:] >= 0))   # moment of passing
        w = P["window"]
        lo, hi = max(0, k - w), min(len(d), k + w + 1)
        side = float(np.mean(l[lo:hi]))            # sideways offset while passing
        if abs(side) > P["max_lateral"]:
            continue

        violations = []
        if abs(side) >= P["side_min"]:
            wrong = side < 0 if cfg["drive_side"] == "left" else side > 0
            if wrong:
                violations.append("WRONG_SIDE_OVERTAKE")

        if line is not None and cfg["line_type"] == "solid":
            m = (B["frames"] >= common[lo]) & (B["frames"] <= common[hi - 1])
            seg = B["xy"][m]
            if len(seg) >= 2 and LineString(seg).intersects(line):
                violations.append("SOLID_LINE_CROSSING")

        events.append(dict(
            frame=int(common[k]), overtaker=idb, overtaken=ida,
            overtaker_class=B["cls"], side="RIGHT" if side > 0 else "LEFT",
            violations=violations))
    return sorted(events, key=lambda e: e["frame"])
