"""main.py - detect + track vehicles, find wrong overtakes, save evidence.

Usage:  python main.py videos/traffic.mp4 --config config.json
Outputs (in ./output): annotated.mp4, violations.csv, evidence/*.jpg, tracks.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import cv2

from rules import detect_events, load_config

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def run_tracking(video, model_name="yolov8n.pt", conf=0.3, imgsz=640):
    from ultralytics import YOLO          # imported here so rules/tests work without it
    model = YOLO(model_name)
    tracks = {}
    results = model.track(source=video, stream=True, tracker="bytetrack.yaml",
                          persist=True, conf=conf, imgsz=imgsz,
                          classes=list(COCO_VEHICLES), verbose=False)
    for frame_idx, r in enumerate(results):
        if r.boxes.id is None:
            continue
        ids = r.boxes.id.int().cpu().tolist()
        boxes = r.boxes.xyxy.cpu().tolist()
        clss = r.boxes.cls.int().cpu().tolist()
        for i, b, c in zip(ids, boxes, clss):
            tracks.setdefault(str(i), []).append(
                {"f": frame_idx, "box": [round(v, 1) for v in b], "cls": COCO_VEHICLES[c]})
    return tracks


def render(video, tracks, events, cfg, out_video, evidence_dir):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    by_frame = defaultdict(list)
    for tid, pts in tracks.items():
        for p in pts:
            by_frame[p["f"]].append((tid, p["box"], p["cls"]))
    bad = [e for e in events if e["violations"]]
    line = [tuple(map(int, p)) for p in cfg["center_line"]]
    n_saved = 0

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for a, b in zip(line, line[1:]):
            cv2.line(frame, a, b, (0, 255, 255), 2)
        banner = None
        for tid, box, cls in by_frame.get(idx, []):
            active = [e for e in bad if e["overtaker"] == tid and e["frame"] - 15 <= idx <= e["frame"] + 45]
            color = (0, 0, 255) if active else (0, 200, 0)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls} #{tid}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if active:
                banner = f"VIOLATION #{tid}: {', '.join(active[0]['violations'])}"
        if banner:
            cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 255), -1)
            cv2.putText(frame, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for e in bad:
            if e["frame"] == idx:
                path = os.path.join(evidence_dir, f"event_{n_saved}_vehicle{e['overtaker']}.jpg")
                cv2.imwrite(path, frame)
                e["snapshot"] = path
                n_saved += 1
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    return fps


def process(video, config_path, out_dir="output", model_name="yolov8n.pt"):
    cfg = load_config(config_path)
    os.makedirs(os.path.join(out_dir, "evidence"), exist_ok=True)

    print("1/3 Detecting + tracking vehicles ...")
    tracks = run_tracking(video, model_name)
    json.dump(tracks, open(os.path.join(out_dir, "tracks.json"), "w"))
    print(f"    {len(tracks)} tracks found")

    print("2/3 Checking overtaking rules ...")
    events = detect_events(tracks, cfg)
    print(f"    {len(events)} overtakes, {sum(1 for e in events if e['violations'])} violations")

    print("3/3 Writing annotated video + evidence ...")
    out_video = os.path.join(out_dir, "annotated.mp4")
    fps = render(video, tracks, events, cfg, out_video, os.path.join(out_dir, "evidence"))

    csv_path = os.path.join(out_dir, "violations.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "frame", "overtaker_id", "vehicle", "overtaken_id",
                    "side", "status", "violations", "snapshot"])
        for e in events:
            w.writerow([round(e["frame"] / fps, 2), e["frame"], e["overtaker"],
                        e["overtaker_class"], e["overtaken"], e["side"],
                        "VIOLATION" if e["violations"] else "LEGAL",
                        "|".join(e["violations"]), e.get("snapshot", "")])
    print("Done. See the", out_dir, "folder.")
    return events, out_video, csv_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="yolov8n.pt")
    a = ap.parse_args()
    process(a.video, a.config, a.out, a.model)
