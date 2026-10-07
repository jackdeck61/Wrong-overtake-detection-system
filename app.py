"""app.py - Streamlit demo.  Run:  streamlit run app.py
Needs config.json (made by calibrate.py for the SAME camera view)."""
import os
import subprocess
import tempfile

import pandas as pd
import streamlit as st

from main import process

st.set_page_config(page_title="Wrong Overtaking Detection", layout="wide")
st.title("🚦 Wrong Overtaking Detection")
st.caption("Upload CCTV footage from the calibrated camera. Rules: overtake on the correct side, never across a solid line.")

cfg_path = st.text_input("Calibration file", "config.json")
up = st.file_uploader("CCTV video", type=["mp4", "avi", "mov"])


def to_h264(src):
    """OpenCV writes mp4v which browsers can't play; convert if ffmpeg exists."""
    dst = src.replace(".mp4", "_web.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst],
                       check=True, capture_output=True)
        return dst
    except Exception:
        return src


if up and st.button("Analyse video", type="primary"):
    if not os.path.exists(cfg_path):
        st.error("config.json not found. Run calibrate.py first.")
        st.stop()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(up.read())
    tmp.close()
    with st.spinner("Detecting, tracking and checking rules..."):
        events, video_path, csv_path = process(tmp.name, cfg_path, "output")
    st.success(f"{len(events)} overtakes found, {sum(1 for e in events if e['violations'])} violations")

    left, right = st.columns([3, 2])
    with left:
        st.video(to_h264(video_path))
    with right:
        df = pd.read_csv(csv_path)
        st.dataframe(df.drop(columns=["snapshot"]), use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "violations.csv")

    snaps = [p for p in df["snapshot"].dropna().tolist() if os.path.exists(p)]
    if snaps:
        st.subheader("Evidence")
        cols = st.columns(min(3, len(snaps)))
        for i, p in enumerate(snaps):
            cols[i % len(cols)].image(p, caption=os.path.basename(p))
