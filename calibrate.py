"""calibrate.py - click on the first frame to describe the road. Saves config.json.

Usage:  python calibrate.py videos/traffic.mp4 --drive_side left --line_type solid

Click order:
  1st, 2nd click : travel direction (a point near the bottom, then a point further up the road)
  next clicks    : points along the CENTER / DIVIDER LINE (default mode)
Keys:
  z = start a NO-OVERTAKE ZONE polygon (junction, zebra crossing, bend, bridge) - click its corners
  o = start an ONCOMING-LANE polygon (the opposite carriageway) - click its corners
  l = go back to adding points to the center line
  u = undo last click      s = save      q = quit
"""
import argparse
import json
import sys

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default="config.json")
ap.add_argument("--drive_side", default="left", choices=["left", "right"])
ap.add_argument("--line_type", default="solid", choices=["solid", "double", "yellow", "broken"])
ap.add_argument("--lane_traffic", action="store_true", help="multi-lane road: skip left/right side rule")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit("Could not read the video. Check the path.")

H, W = frame.shape[:2]
scale = min(1.0, 1280 / W)
disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()

direction, line, zones = [], [], []
mode = "line"
COL = {"no_overtake": (0, 140, 255), "oncoming": (255, 0, 255)}


def redraw():
    img = disp.copy()
    if len(direction) == 2:
        cv2.arrowedLine(img, direction[0], direction[1], (0, 255, 0), 2, tipLength=0.1)
    for p in direction:
        cv2.circle(img, p, 6, (0, 255, 0), -1)
    for a, b in zip(line, line[1:]):
        cv2.line(img, a, b, (0, 0, 255), 2)
    for p in line:
        cv2.circle(img, p, 5, (0, 0, 255), -1)
    for z in zones:
        pts = np.array(z["points"], np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], len(pts) >= 3, COL[z["type"]], 2)
        for p in z["points"]:
            cv2.circle(img, p, 5, COL[z["type"]], -1)
    if len(direction) < 2:
        msg = "Click 2 points: travel direction (bottom -> up the road)"
    else:
        msg = f"mode: {mode} | z=no-overtake zone o=oncoming l=line u=undo s=save q=quit"
    cv2.putText(img, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imshow("calibrate", img)


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(direction) < 2:
        direction.append((x, y))
    elif mode == "line":
        line.append((x, y))
    else:
        zones[-1]["points"].append((x, y))
    redraw()


def new_zone(ztype):
    global mode
    mode = ztype
    zones.append({"name": f"{ztype}_{len(zones) + 1}", "type": ztype, "points": []})


cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_mouse)
redraw()
while True:
    k = cv2.waitKey(20) & 0xFF
    if k == ord("z"):
        new_zone("no_overtake"); redraw()
    elif k == ord("o"):
        new_zone("oncoming"); redraw()
    elif k == ord("l"):
        mode = "line"; redraw()
    elif k == ord("u"):
        if mode != "line" and zones and zones[-1]["points"]:
            zones[-1]["points"].pop()
        elif mode == "line" and line:
            line.pop()
        elif direction and not line and not zones:
            direction.pop()
        redraw()
    elif k == ord("q"):
        break
    elif k == ord("s"):
        if len(direction) < 2:
            print("Need the 2 direction points first.")
            continue
        up = lambda pts: [[x / scale, y / scale] for x, y in pts]    # back to original resolution
        cfg = {"direction": up(direction), "center_line": up(line),
               "zones": [{"name": z["name"], "type": z["type"], "points": up(z["points"])}
                         for z in zones if len(z["points"]) >= 3],
               "drive_side": args.drive_side, "line_type": args.line_type,
               "lane_traffic": args.lane_traffic, "params": {}}
        json.dump(cfg, open(args.out, "w"), indent=2)
        print("Saved", args.out)
        break
cv2.destroyAllWindows()
