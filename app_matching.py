"""Standalone read-only top-three viewer for completed two-camera runs."""

from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Two-Camera Match Review", page_icon="◉", layout="wide")
st.markdown("""
<style>
.stApp { background:#0b1119; color:#e8eef6; }
[data-testid="stSidebar"] { background:#101923; }
.block-container { max-width:1320px; padding-top:2rem; }
div[data-testid="stMetric"] { background:#121e2a; border:1px solid #283747; border-radius:12px; padding:12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_matches(csv_path: str, modified: float) -> list[dict]:
    del modified
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


st.title("Two-Camera Person Match Review")
st.caption("Review 3 · OSNet appearance candidates · Similarity scores rank visual matches; they are not probabilities or identity proof.")
run_root = st.text_input("Completed run folder", str(ROOT / "outputs"))
root = Path(run_root).expanduser()
if not root.is_dir():
    st.info("Choose a folder containing a completed run with matches.csv.")
    st.stop()
match_files = sorted(root.rglob("matches.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
if not match_files:
    st.info("No match table found yet. Run the two-camera processing first.")
    st.stop()

run_label = st.selectbox("Select a completed run", match_files,
                         format_func=lambda path: str(path.parent.relative_to(root)))
rows = load_matches(str(run_label), run_label.stat().st_mtime)
if not rows:
    st.warning("This run has no Camera 1-to-Camera 2 candidate rows.")
    st.stop()
queries = sorted({row["query_track_key"] for row in rows})
query = st.selectbox("Camera 1 person", queries)
query_rows = [row for row in rows if row["query_track_key"] == query]
selected = sorted((row for row in query_rows if int(row["rank"]) > 0),
                  key=lambda row: int(row["rank"]))
query_crop = next((row["query_best_crop"] for row in selected if row.get("query_best_crop")), "")
columns = st.columns([1, 3])
with columns[0]:
    st.subheader(query)
    if query_crop and Path(query_crop).is_file():
        st.image(query_crop, caption="Best retained Camera 1 crop", use_container_width=True)
    else:
        st.caption("No retained source crop is available.")
with columns[1]:
    st.subheader("Top Camera 2 candidates")
    if not selected:
        st.info("No candidate fingerprint could be ranked for Camera 2.")
    else:
        cards = st.columns(len(selected))
        for column, row in zip(cards, selected):
            with column:
                st.metric(f"Rank {row['rank']} · {row['candidate_track_key']}", f"{float(row['score']):.3f}")
                crop = row.get("candidate_best_crop", "")
                if crop and Path(crop).is_file():
                    st.image(crop, caption="Best retained candidate crop", use_container_width=True)
                else:
                    st.caption("No retained candidate crop is available.")
run_dir = run_label.parent
videos = [run_dir / camera / "processed_video.mp4" for camera in ("camera1", "camera2")]
available = [path for path in videos if path.is_file()]
if available:
    with st.expander("Processed video context"):
        video_columns = st.columns(len(available))
        for column, path in zip(video_columns, available):
            with column:
                st.caption(path.parent.name)
                st.video(str(path))
st.download_button("Download match table (.csv)", run_label.read_bytes(),
                   file_name="matches.csv", mime="text/csv")
