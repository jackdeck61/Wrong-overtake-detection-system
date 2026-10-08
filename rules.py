"""rules.py - traffic-law engine for wrong-overtaking detection (India, left-hand traffic).

tracks format (produced by main.run_tracking):
{ "7": [ {"f": 12, "box": [x1,y1,x2,y2], "cls": "car"}, ... ], ... }

Legal basis: Rules of the Road Regulations 1989 (made under s.118 Motor Vehicles Act 1988).
Always double-check regulation numbers against the official PDF before quoting them.
"""
import json
from collections import Counter
from itertools import permutations

import numpy as np
from shapely.geometry import LineString, Polygon

# ----------------------------------------------------------------------------------
# 1. THE RULEBOOK  (code -> text shown in CSV / report)
# ----------------------------------------------------------------------------------
RULES = {
    "WRONG_SIDE_OVERTAKE": {
        "description": "Overtook on the LEFT. Vehicles must pass to the right of traffic going the same way.",
        "legal_ref": "Rules of the Road Regs 1989, Reg. 4 (exceptions in Reg. 5)",
    },
    "DIVIDING_LINE_CROSSING": {
        "description": "Crossed a solid / yellow dividing line while overtaking.",
        "legal_ref": "Reg. 17(2) yellow dividing line; IRC:35 road markings; MV Act s.119",
    },
    "OVERTAKE_IN_NO_OVERTAKE_ZONE": {
        "description": "Overtook at a place where overtaking is prohibited (junction, pedestrian crossing, "
                       "bend/curve, bridge, poor visibility).",
        "legal_ref": "Reg. 6 (overtaking prohibited in certain cases)",
    },
    "WRONG_SIDE_OF_ROAD": {
        "description": "Drove into the oncoming carriageway / wrong side of the road while overtaking.",
        "legal_ref": "Keep-left rule, Rules of the Road Regs 1989; MV Act s.184 (dangerous driving)",
    },
    "UNSAFE_CUT_IN": {
        "description": "Cut back in front of the overtaken vehicle before a safe distance was gained.",
        "legal_ref": "General safe-overtaking duty, Reg. 6/7; MV Act s.184",
    },
}

# ----------------------------------------------------------------------------------
# 2. SETTINGS  (pixel values assume ~1280x720 video; tune per camera)
# ----------------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "min_track_len": 30,    # ignore tracks shorter than this many frames
    "min_overlap": 20,      # two vehicles must be visible together this many frames
    "margin": 10,           # px forward gap to be "clearly behind / ahead"
    "window": 15,           # frames either side of the pass used to judge it
    "max_lateral": 250,     # sideways distance beyond which it isn't an overtake
    "side_min": 20,         # sideways gap below this -> side is ambiguous, don't flag
    "safe_gap": 80,         # px: B must be at least this far ahead before moving back in
    "cutin_lat": 15,        # px: B is "back in A's path" if sideways gap < this
    "turn_right_shift": 40, # px: A drifting right by this much = "turning right" (left pass allowed)
}

DEFAULT_CONFIG = {
    "drive_side": "left",                       # "left" (India) or "right"
    "line_type": "solid",                       # "solid" | "double" | "yellow" -> no crossing; "broken" -> allowed
    "lane_traffic": False,                      # True on multi-lane roads: skip the left/right side rule
    "check_classes": ["car", "motorcycle", "bus", "truck"],
    "center_line": [],                          # polyline [[x,y],...]
    "zones": [],                                # [{"name":..,"type":"no_overtake"|"oncoming","points":[[x,y],..]}]
    "enabled_rules": list(RULES),               # remove codes here to switch a rule off
}

NO_CROSS_LINES = {"solid", "double", "yellow"}


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    params = dict(DEFAULT_PARAMS)
    params.update(cfg.get("params", {}))
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    cfg["params"] = params
    return cfg


# ----------------------------------------------------------------------------------
# 3. GEOMETRY HELPERS
# ----------------------------------------------------------------------------------
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
        xy = np.array([[(p["box"][0] + p["box"][2]) / 2, p["box"][3]] for p in pts], float)
        xy = _smooth(xy)                                   # ground point = bottom-centre of box
        fwd, lat = xy @ f, xy @ r
        if fwd[-1] - fwd[0] <= 0:                          # not moving along travel direction
            continue
        cls = Counter(p["cls"] for p in pts).most_common(1)[0][0]
        out[tid] = dict(frames=frames, xy=xy, fwd=fwd, lat=lat, cls=cls)
    return out


def _zones(cfg):
    out = []
    for z in cfg["zones"]:
        if len(z.get("points", [])) >= 3:
            out.append((z.get("name", "zone"), z.get("type", "no_overtake"), Polygon(z["points"])))
    return out


# ----------------------------------------------------------------------------------
# 4. MAIN DETECTOR
# ----------------------------------------------------------------------------------
def detect_events(tracks, cfg):
    """Find every overtake (B passes A). 'violations' = list of RULES codes (empty = legal)."""
    P = cfg["params"]
    f, r = _axes(cfg)
    T = _prepare(tracks, f, r, P)
    line = LineString(cfg["center_line"]) if len(cfg["center_line"]) >= 2 else None
    zones = _zones(cfg)
    on = set(cfg["enabled_rules"])
    events = []

    for ida, idb in permutations(T, 2):                     # B tries to overtake A
        A, B = T[ida], T[idb]
        if B["cls"] not in cfg["check_classes"]:
            continue
        common = np.intersect1d(A["frames"], B["frames"])
        if len(common) < P["min_overlap"]:
            continue
        ia = np.searchsorted(A["frames"], common)
        ib = np.searchsorted(B["frames"], common)
        d = B["fwd"][ib] - A["fwd"][ia]                      # >0: B ahead of A
        l = B["lat"][ib] - A["lat"][ia]                      # >0: B to the right of A

        # --- is this an overtake? (B clearly behind, later clearly ahead)
        behind = np.where(d < -P["margin"])[0]
        ahead = np.where(d > P["margin"])[0]
        if len(behind) == 0 or len(ahead) == 0 or behind[0] > ahead[-1]:
            continue
        k = behind[0] + int(np.argmax(d[behind[0]:] >= 0))   # moment of passing
        w = P["window"]
        lo, hi = max(0, k - w), min(len(d), k + w + 1)
        side = float(np.mean(l[lo:hi]))                      # sideways offset while passing
        if abs(side) > P["max_lateral"]:
            continue

        # B's path around the pass
        m = (B["frames"] >= common[lo]) & (B["frames"] <= common[hi - 1])
        seg = B["xy"][m]
        path = LineString(seg) if len(seg) >= 2 else None

        v, details = [], []

        # RULE 1: pass on the correct side (Reg. 4) - with Reg. 5 exceptions
        if "WRONG_SIDE_OVERTAKE" in on and not cfg["lane_traffic"] and abs(side) >= P["side_min"]:
            wrong = side < 0 if cfg["drive_side"] == "left" else side > 0
            a_lat = A["lat"][ia]
            a_shift = a_lat[min(len(d) - 1, k + w)] - a_lat[max(0, k - w)]
            a_turning_right = (a_shift > P["turn_right_shift"]) if cfg["drive_side"] == "left" \
                else (a_shift < -P["turn_right_shift"])
            if wrong and a_turning_right:
                details.append("left pass allowed: vehicle ahead appears to be turning right")
            elif wrong:
                v.append("WRONG_SIDE_OVERTAKE")

        if path is not None:
            # RULE 2: do not cross a solid / yellow dividing line
            if ("DIVIDING_LINE_CROSSING" in on and line is not None
                    and cfg["line_type"] in NO_CROSS_LINES and path.intersects(line)):
                v.append("DIVIDING_LINE_CROSSING")

            # RULE 3 + 4: zones (no-overtake areas, oncoming carriageway)
            for name, ztype, poly in zones:
                if not path.intersects(poly):
                    continue
                if ztype == "no_overtake" and "OVERTAKE_IN_NO_OVERTAKE_ZONE" in on:
                    v.append("OVERTAKE_IN_NO_OVERTAKE_ZONE")
                    details.append(f"zone: {name}")
                elif ztype == "oncoming" and "WRONG_SIDE_OF_ROAD" in on:
                    v.append("WRONG_SIDE_OF_ROAD")
                    details.append(f"zone: {name}")

        # RULE 5: do not cut back in before a safe gap
        if "UNSAFE_CUT_IN" in on and abs(side) >= P["side_min"]:
            dk, lk = d[k:], l[k:]
            if np.any((dk >= 0) & (dk < P["safe_gap"]) & (np.abs(lk) < P["cutin_lat"])):
                v.append("UNSAFE_CUT_IN")

        events.append(dict(
            frame=int(common[k]), overtaker=idb, overtaken=ida,
            overtaker_class=B["cls"], side="RIGHT" if side > 0 else "LEFT",
            violations=sorted(set(v)), details=details))
    return sorted(events, key=lambda e: e["frame"])
