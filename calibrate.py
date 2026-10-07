"""calibrate.py - click on the first frame to tell the system where the road goes.

Usage:  python calibrate.py videos/traffic.mp4 --drive_side left --line_type solid

Click order:
  1st click : a point on the road FAR BEHIND (near the bottom of the image)
  2nd click : a point further ALONG the direction vehicles travel (up the road)
  3rd+ clicks: points along the CENTER / DIVIDER LINE (optional, in order)
Keys: u = undo last click, s = save config.json, q = quit
"""
import argparse
import json
import sys

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default="config.json")
ap.add_argument("--drive_side", default="left", choices=["left", "right"])
ap.add_argument("--line_type", default="solid", choices=["solid", "broken"])
args = ap.parse_args()

cap = cv2.VideoCapture(args.video)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit("Could not read the video. Check the path.")

H, W = frame.shape[:2]
scale = min(1.0, 1280 / W)
disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
pts = []


def redraw():
    img = disp.copy()
    for i, p in enumerate(pts):
        cv2.circle(img, p, 6, (0, 255, 0) if i < 2 else (0, 0, 255), -1)
    if len(pts) >= 2:
        cv2.arrowedLine(img, pts[0], pts[1], (0, 255, 0), 2, tipLength=0.1)
    for a, b in zip(pts[2:], pts[3:]):
        cv2.line(img, a, b, (0, 0, 255), 2)
    if len(pts) < 2:
        msg = "Click 2 points along travel direction (bottom -> up the road)"
    else:
        msg = "Click center line points (optional). s=save u=undo q=quit"
    cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("calibrate", img)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        redraw()


cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_mouse)
redraw()
while True:
    k = cv2.waitKey(20) & 0xFF
    if k == ord("u") and pts:
        pts.pop()
        redraw()
    elif k == ord("q"):
        break
    elif k == ord("s"):
        if len(pts) < 2:
            print("Need at least the 2 direction points first.")
            continue
        real = [[x / scale, y / scale] for x, y in pts]   # back to original resolution
        cfg = {"direction": real[:2], "center_line": real[2:],
               "drive_side": args.drive_side, "line_type": args.line_type, "params": {}}
        json.dump(cfg, open(args.out, "w"), indent=2)
        print("Saved", args.out)
        break
cv2.destroyAllWindows()
