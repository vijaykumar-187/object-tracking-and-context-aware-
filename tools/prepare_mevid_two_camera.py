"""Create a two-camera MEVID manifest and separate ground-truth pair table.

This reads the official MEVID tracklet metadata. Its IDs are dataset ground truth,
not predictions and must never be fed to the detector or matcher as features.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path


VIDEO_URLS = "https://raw.githubusercontent.com/Kitware/MEVID/main/mevid-v1-video-URLS.txt"


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def resolve_file(root: Path, filename: str) -> Path:
    candidates = [root / filename, root / f"{filename.removesuffix('.txt')}.txt"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Required MEVID annotation file not found in {root}: {filename}")


def read_tracklets(root: Path, split: str) -> list[dict]:
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("Split must be train or test.")
    info_path = resolve_file(root, f"track_{split}_info.txt")
    names_path = resolve_file(root, f"{split}_name.txt")
    names = read_lines(names_path)
    tracklets = []
    for index, line in enumerate(read_lines(info_path)):
        fields = [int(part) for part in re.split(r"[\s,]+", line) if part]
        if len(fields) != 5:
            raise ValueError(f"Expected five integer columns at {info_path}:{index + 1}.")
        start, end, person_id, outfit_id, camera_id = fields
        if start < 0 or end < start or end >= len(names):
            raise ValueError(f"Invalid crop-name offsets for tracklet {index}.")
        first_name, last_name = names[start], names[end]
        tracklets.append({
            "split": split,
            "dataset_tracklet_id": index,
            "global_person_id": person_id,
            "outfit_id": outfit_id,
            "camera_id": camera_id,
            "start_name_offset": start,
            "end_name_offset": end,
            "frame_count": end - start + 1,
            "first_crop_name": first_name,
            "last_crop_name": last_name,
        })
    return tracklets


def build_two_camera_tables(tracklets: list[dict], camera1: int, camera2: int,
                            output_dir: Path, split: str,
                            video_urls: list[str] | None = None) -> tuple[Path, Path, dict]:
    if camera1 == camera2:
        raise ValueError("Camera 1 and Camera 2 must be different camera IDs.")
    cam1 = [row for row in tracklets if row["camera_id"] == camera1]
    cam2 = [row for row in tracklets if row["camera_id"] == camera2]
    if not cam1 or not cam2:
        available = sorted({row["camera_id"] for row in tracklets})
        raise ValueError(f"One selected camera has no tracklets. Available IDs: {available[:30]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"mevid_{split}_cam{camera1}_cam{camera2}"
    tracklet_path = output_dir / f"{prefix}_tracklets.csv"
    pairs_path = output_dir / f"{prefix}_ground_truth_pairs.csv"
    fields = ["split", "dataset_tracklet_id", "global_person_id", "outfit_id",
              "camera_id", "frame_count", "start_name_offset", "end_name_offset",
              "first_crop_name", "last_crop_name"]
    with tracklet_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cam1 + cam2)

    pairs_fields = ["camera1_tracklet_id", "camera2_tracklet_id", "is_same_person",
                    "global_person_id", "camera1_outfit_id", "camera2_outfit_id"]
    with pairs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=pairs_fields)
        writer.writeheader()
        for left in cam1:
            for right in cam2:
                same_person = left["global_person_id"] == right["global_person_id"]
                writer.writerow({
                    "camera1_tracklet_id": left["dataset_tracklet_id"],
                    "camera2_tracklet_id": right["dataset_tracklet_id"],
                    "is_same_person": int(same_person),
                    "global_person_id": left["global_person_id"] if same_person else "",
                    "camera1_outfit_id": left["outfit_id"],
                    "camera2_outfit_id": right["outfit_id"],
                })

    summary = {
        "dataset": "MEVID v1 based on MEVA recorded camera video",
        "split": split,
        "selected_camera_ids": [camera1, camera2],
        "camera1_tracklets": len(cam1),
        "camera2_tracklets": len(cam2),
        "camera1_global_identities": len({row["global_person_id"] for row in cam1}),
        "camera2_global_identities": len({row["global_person_id"] for row in cam2}),
        "positive_pairs": sum(a["global_person_id"] == b["global_person_id"]
                              for a in cam1 for b in cam2),
        "all_candidate_pairs": len(cam1) * len(cam2),
        "ground_truth_separate_from_predictions": True,
        "note": (
            "MEVID ground truth is tracklet/crop metadata. The source video URLs are "
            "listed separately by the official project; this script does not assert "
            "a direct frame-level mapping between crop tracklets and video clips."
        ),
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if video_urls is not None:
        selected_urls = []
        for camera in (camera1, camera2):
            marker = f".G{camera}."
            selected_urls.extend(
                {"camera_id": camera, "url": url}
                for url in video_urls if marker in url
            )
        video_list_path = output_dir / f"{prefix}_public_video_urls.csv"
        with video_list_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["camera_id", "url"])
            writer.writeheader()
            writer.writerows(selected_urls)
        summary["matching_video_urls"] = len(selected_urls)
        summary["video_url_manifest"] = str(video_list_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return tracklet_path, pairs_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing track_{train,test}_info.txt and name lists")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--camera1", type=int, required=True, help="MEVID numeric camera ID")
    parser.add_argument("--camera2", type=int, required=True, help="A different MEVID numeric camera ID")
    parser.add_argument("--output", type=Path, default=Path("outputs/mevid_two_camera"))
    parser.add_argument("--include-video-urls", action="store_true",
                        help="Fetch the official 976-video URL list and write links for these cameras")
    args = parser.parse_args()

    tracklets = read_tracklets(args.root, args.split)
    video_urls = None
    if args.include_video_urls:
        with urllib.request.urlopen(VIDEO_URLS, timeout=30) as response:
            video_urls = response.read().decode("utf-8").splitlines()
        video_urls = [url.strip() for url in video_urls if url.strip().startswith("https://")]
    tracklet_path, pair_path, summary = build_two_camera_tables(
        tracklets, args.camera1, args.camera2, args.output, args.split, video_urls,
    )
    print(json.dumps(summary, indent=2))
    print(f"Tracklet inventory (ground truth only): {tracklet_path}")
    print(f"Cross-camera labeled pair table: {pair_path}")
    print(f"Official source-video list: {VIDEO_URLS}")


if __name__ == "__main__":
    main()
