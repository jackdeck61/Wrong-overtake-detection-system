"""Test rules.py with FAKE trajectories (no video / YOLO needed).
Run:  python test_rules.py
Road goes UP the image (direction (500,700)->(500,300)); left-hand traffic.
"""
from rules import load_config, detect_events
import json, tempfile, os


def make_track(x_of_t, y_of_t, n=100):
    return [{"f": t, "box": [x_of_t(t) - 20, y_of_t(t) - 40, x_of_t(t) + 20, y_of_t(t)], "cls": "car"}
            for t in range(n)]


def cfg(center_line=None):
    c = {"direction": [[500, 700], [500, 300]], "drive_side": "left", "line_type": "solid",
         "center_line": center_line or []}
    p = os.path.join(tempfile.gettempdir(), "cfg_test.json")
    json.dump(c, open(p, "w"))
    return load_config(p)


slow = make_track(lambda t: 500, lambda t: 600 - 3 * t)               # slow car, x=500

# 1) fast car passes on the LEFT (x=440)  -> should be WRONG_SIDE
left = make_track(lambda t: 440, lambda t: 700 - 6 * t)
ev = detect_events({"1": slow, "2": left}, cfg())
print("left pass :", ev)
assert ev and "WRONG_SIDE_OVERTAKE" in ev[0]["violations"]

# 2) fast car passes on the RIGHT (x=560) -> legal
right = make_track(lambda t: 560, lambda t: 700 - 6 * t)
ev = detect_events({"1": slow, "2": right}, cfg())
print("right pass:", ev)
assert ev and ev[0]["violations"] == []

# 3) passes on the right but crosses a solid line at x=530
drift = make_track(lambda t: 600 - 2 * t, lambda t: 700 - 6 * t)
ev = detect_events({"1": slow, "2": drift}, cfg([[530, 0], [530, 800]]))
print("line cross:", ev)
assert ev and "SOLID_LINE_CROSSING" in ev[0]["violations"]
print("ALL TESTS PASSED")
