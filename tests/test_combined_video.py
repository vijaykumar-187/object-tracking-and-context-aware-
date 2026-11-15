"""Phase D coverage for one combined multi-camera video per global identity.

The renderer is additive: it reads the registry, the saved track CSVs and the
source videos, and never creates, merges or rejects an identity. Every test uses
small synthetic clips so the module stays fast and dependency-free.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.reid.global_registry import (
    associate_confirmed_track, get_or_create_identity,
)
from src.reid.video_results import (
    COMBINED_OUTPUT_SIZE, COMBINED_PANEL, COMBINED_SYNC_NOTE,
    COMBINED_TILE_SIZE, combined_video_status, compose_empty_tile,
    compose_end_card, compose_grid, compose_tile, end_card_lines, grid_shape,
    load_track_index, plan_frame_boxes, render_combined_global_person_video,
    tile_caption,
)

FRAME_INTERVAL = 2
MAX_LOST_FRAMES = 8
PREDICTION_WINDOW = 4


def _steady_rows(track_id, frame_count, box, every=2):
    """Alternate one observed row and one Kalman-predicted row."""
    return [{
        "frame_number": frame,
        "track_id": track_id,
        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
        "observed": 1 if frame % every == 0 else 0,
        "time_since_update": frame % every,
    } for frame in range(frame_count)]


def _write_camera(root, camera_id, camera_name, width, height, frame_count,
                  fps, rows, frame_interval=FRAME_INTERVAL,
                  max_lost_frames=MAX_LOST_FRAMES):
    folder = root / camera_id
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / f"{camera_id}.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height),
    )
    if not writer.isOpened():
        raise AssertionError("Synthetic source video could not be created.")
    for index in range(frame_count):
        frame = np.full((height, width, 3), 40 + (index % 5), dtype=np.uint8)
        frame[: max(1, height // 3)] = (70, 60, 90)
        writer.write(frame)
    writer.release()

    tracks = folder / "tracks.csv"
    fields = ["camera_id", "frame_number", "timestamp_seconds", "track_id",
              "x1", "y1", "x2", "y2", "confidence", "observed", "recovered",
              "time_since_update"]
    with tracks.open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=fields)
        output.writeheader()
        for row in rows:
            output.writerow({
                "camera_id": camera_id,
                "timestamp_seconds": round(row["frame_number"] / fps, 3),
                "confidence": 0.9,
                "recovered": 0,
                **row,
            })
    summary = folder / "tracks.json"
    summary.write_text(json.dumps({
        "max_lost_frames": max_lost_frames,
        "detector_settings": {"frame_interval": frame_interval},
        "tracks": [],
    }), encoding="utf-8")
    return {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "source_name": source.name,
        "source_path": str(source),
        "tracks_file": str(tracks),
        "track_summary_file": str(summary),
    }


def _three_cameras(root):
    """Camera 1 holds two tracks, camera 2 one, camera 3 no member."""
    return [
        _write_camera(root, "camera1", "Camera 1", 160, 120, 20, 10.0,
                      _steady_rows(3, 20, (5, 10, 45, 100)) +
                      _steady_rows(7, 20, (110, 10, 150, 100))),
        _write_camera(root, "camera2", "Camera 2", 120, 160, 24, 10.0,
                      _steady_rows(4, 24, (20, 20, 90, 140))),
        _write_camera(root, "camera3", "Camera 3", 200, 100, 16, 10.0,
                      _steady_rows(9, 16, (40, 10, 120, 90))),
    ]


def _read_frames(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError("The combined video could not be opened.")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _green(image):
    """Count observed-track green, excluding the amber prediction/status colour."""
    blue, green, red = (image[:, :, 0].astype(int), image[:, :, 1].astype(int),
                        image[:, :, 2].astype(int))
    return int(((green > 190) & (red > 110) & (red < 190) & (blue < 110)).sum())


def _amber(image):
    blue, green, red = (image[:, :, 0].astype(int), image[:, :, 1].astype(int),
                        image[:, :, 2].astype(int))
    return int(((green > 140) & (red > 215) & (blue < 90)).sum())


class TestCombinedTilePlanning(unittest.TestCase):
    """Pure planning checks: what may be drawn at one source frame."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tracks = self.root / "tracks.csv"
        with self.tracks.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "frame_number", "track_id", "x1", "y1", "x2", "y2",
                "observed", "time_since_update",
            ])
            writer.writeheader()
            for frame in range(0, 4):
                writer.writerow({
                    "frame_number": frame, "track_id": 3,
                    "x1": 5, "y1": 10, "x2": 45, "y2": 100,
                    "observed": 1, "time_since_update": 0,
                })
            for frame in range(0, 4):
                writer.writerow({
                    "frame_number": frame, "track_id": 7,
                    "x1": 50, "y1": 10, "x2": 90, "y2": 100,
                    "observed": 1 if frame == 0 else 0,
                    "time_since_update": frame,
                })
            for frame in (5, 6, 7, 8, 9):
                writer.writerow({
                    "frame_number": frame, "track_id": 5,
                    "x1": 10, "y1": 20, "x2": 60, "y2": 110,
                    "observed": 1, "time_since_update": 0,
                })

    def tearDown(self):
        self.temporary.cleanup()

    def test_index_keeps_every_member_and_reports_frame_spans(self):
        index = load_track_index(self.tracks, [3, 7, 5])
        self.assertEqual(set(index["rows"]), {3, 5, 7})
        self.assertEqual(index["spans"][3], (0, 3))
        self.assertEqual(index["spans"][5], (5, 9))
        self.assertEqual(index["spans"][7], (0, 3))
        self.assertTrue(index["rows"][7][0]["observed"])
        self.assertEqual(index["rows"][7][2]["age"], 2)

    def test_index_ignores_tracks_that_are_not_members(self):
        index = load_track_index(self.tracks, [3])
        self.assertEqual(set(index["rows"]), {3})

    def test_plan_boxes_every_member_of_one_camera(self):
        index = load_track_index(self.tracks, [3, 7])
        members = [{"track_id": 7}, {"track_id": 3}]
        boxes = plan_frame_boxes(members, index, 0, PREDICTION_WINDOW)
        self.assertEqual([box["track_id"] for box in boxes], [3, 7])
        self.assertIsNotNone(boxes[0]["rect"])
        self.assertIsNotNone(boxes[1]["rect"])

    def test_plan_draws_prediction_only_inside_the_window(self):
        index = load_track_index(self.tracks, [7])
        members = [{"track_id": 7}]
        inside = plan_frame_boxes(members, index, 2, 2)[0]
        self.assertEqual(inside["state"], "prediction (2f)")
        self.assertFalse(inside["observed"])
        self.assertIsNotNone(inside["rect"])
        outside = plan_frame_boxes(members, index, 3, 2)[0]
        self.assertEqual(outside["state"], "track lost (3f)")
        self.assertIsNone(outside["rect"])

    def test_plan_reports_absent_frames_without_inventing_tracking(self):
        index = load_track_index(self.tracks, [5])
        members = [{"track_id": 5}]
        self.assertEqual(plan_frame_boxes(members, index, 2, PREDICTION_WINDOW)[0]["state"],
                         "not yet in view")
        self.assertEqual(plan_frame_boxes(members, index, 12, PREDICTION_WINDOW)[0]["state"],
                         "left the scene")
        unknown = plan_frame_boxes([{"track_id": 99}], index, 6, PREDICTION_WINDOW)[0]
        self.assertEqual(unknown["state"], "member track missing from this run")
        self.assertIsNone(unknown["rect"])

    def test_caption_names_camera_ids_elapsed_time_and_global_id(self):
        caption = tile_caption("Camera 1", "GLOBAL-001", [3, 7], 4.233, 18.067)
        self.assertIn("Camera 1", caption["header"])
        self.assertIn("GLOBAL-001", caption["header"])
        self.assertIn("Local ID 3, 7", caption["timestamp"])
        self.assertIn("t = 4.233 s", caption["timestamp"])
        self.assertIn("elapsed", caption["timestamp"])
        self.assertIn("not synchronised", caption["footer"])

    def test_end_of_footage_text_is_explicit(self):
        lines = end_card_lines("Camera 2", "GLOBAL-001", [4], 18.067)
        self.assertEqual(lines[0], "END OF FOOTAGE")
        self.assertIn("Camera 2", lines[1])
        self.assertIn("ended at 18.067 s", lines[1])
        self.assertIn("GLOBAL-001", lines[2])
        self.assertIn("Local ID 4", lines[2])


class TestCombinedTileComposition(unittest.TestCase):
    """Geometry checks: tiling, letterboxing and the neutral placeholder."""

    def test_tall_frame_is_letterboxed_without_stretching(self):
        frame = np.full((160, 40, 3), 90, dtype=np.uint8)
        tile = compose_tile(frame, [], "Camera 1", "GLOBAL-001", [3], 0.0, 2.0)
        self.assertEqual(tile.shape, (COMBINED_TILE_SIZE[1], COMBINED_TILE_SIZE[0], 3))
        self.assertTrue(np.array_equal(tile[200, 100], np.array(COMBINED_PANEL)),
                        "The pillar box beside a tall frame must stay neutral.")
        self.assertFalse(np.array_equal(tile[70, 320], np.array(COMBINED_PANEL)),
                         "The frame content must still be drawn in the middle.")
        self.assertLessEqual(abs(int(tile[70, 320, 0]) - 90), 2)

    def test_wide_frame_is_letterboxed_without_stretching(self):
        frame = np.full((40, 160, 3), 90, dtype=np.uint8)
        tile = compose_tile(frame, [], "Camera 1", "GLOBAL-001", [3], 0.0, 2.0)
        self.assertTrue(np.array_equal(tile[100, 320], np.array(COMBINED_PANEL)),
                        "The letterbox bar above a wide frame must stay neutral.")
        self.assertFalse(np.array_equal(tile[130, 320], np.array(COMBINED_PANEL)))

    def test_observed_and_predicted_boxes_use_distinct_colours(self):
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        tile = compose_tile(
            frame,
            [{"track_id": 3, "rect": (10, 10, 60, 100), "observed": True,
              "state": "", "age": 0},
             {"track_id": 7, "rect": (90, 10, 140, 100), "observed": False,
              "state": "prediction (2f)", "age": 2}],
            "Camera 1", "GLOBAL-001", [3, 7], 0.0, 2.0,
        )
        self.assertGreater(_green(tile[:, :320]), 100)
        self.assertGreater(_amber(tile[:, 320:]), 100)

    def test_missing_member_draws_status_text_instead_of_a_box(self):
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        tile = compose_tile(
            frame,
            [{"track_id": 3, "rect": None, "observed": False,
              "state": "not yet in view", "age": None}],
            "Camera 1", "GLOBAL-001", [3], 0.0, 2.0,
        )
        self.assertEqual(_green(tile), 0)
        self.assertGreater(_amber(tile), 50)

    def test_grid_uses_two_by_two_for_three_cameras_with_one_placeholder(self):
        self.assertEqual(grid_shape(3), (2, 2))
        self.assertEqual(grid_shape(4), (2, 2))
        self.assertEqual(grid_shape(2), (1, 2))
        self.assertEqual(grid_shape(1), (1, 1))
        with self.assertRaises(ValueError):
            grid_shape(5)

        tiles = [np.full((COMBINED_TILE_SIZE[1], COMBINED_TILE_SIZE[0], 3),
                         colour, dtype=np.uint8)
                 for colour in ((20, 200, 20), (20, 20, 200), (200, 20, 20))]
        canvas = compose_grid(tiles)
        self.assertEqual(canvas.shape, (COMBINED_OUTPUT_SIZE[1],
                                        COMBINED_OUTPUT_SIZE[0], 3))
        cell_height, cell_width = COMBINED_TILE_SIZE[1], COMBINED_TILE_SIZE[0]
        for position, blue in enumerate((20, 20, 200)):
            row, column = divmod(position, 2)
            centre = canvas[row * cell_height + cell_height // 2,
                            column * cell_width + cell_width // 2]
            self.assertEqual(int(centre[0]), blue)
        placeholder = compose_empty_tile("No camera in this run",
                                         COMBINED_TILE_SIZE)
        self.assertTrue(np.array_equal(
            canvas[cell_height:cell_height * 2, cell_width:cell_width * 2],
            placeholder,
        ), "The unused fourth cell must be the neutral placeholder.")

    def test_end_card_does_not_repeat_the_last_source_frame(self):
        frame = np.full((120, 160, 3), 90, dtype=np.uint8)
        live = compose_tile(frame, [], "Camera 1", "GLOBAL-001", [3], 2.0, 2.0)
        ended = compose_end_card("Camera 1", "GLOBAL-001", [3], 2.0)
        self.assertFalse(np.array_equal(live, ended))
        self.assertGreater(_amber(ended), 200)
        self.assertEqual(_green(ended), 0)


class TestCombinedVideoRender(unittest.TestCase):
    """End-to-end coverage for one combined MP4 per global identity."""

    def test_one_combined_video_contains_every_confirmed_member(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = _three_cameras(root)
            registry = root / "global_identity_registry.json"
            global_id = get_or_create_identity(registry, "camera1", 3, "Camera 1")
            associate_confirmed_track(registry, global_id, "camera1", 7,
                                      "Camera 1", 0.82, "OSNet")
            associate_confirmed_track(registry, global_id, "camera2", 4,
                                      "Camera 2", 0.71, "OSNet")
            before = registry.read_bytes()

            manifest = render_combined_global_person_video(results, root, global_id)

            self.assertEqual(registry.read_bytes(), before,
                             "Combined rendering must not modify the registry.")
            output = root / "global_person_videos" / global_id
            self.assertEqual(sorted(path.name for path in output.glob("*.mp4")),
                             [f"{global_id}_combined_multi_camera.mp4"])
            self.assertEqual(list(output.glob("*.partial.mp4")), [])
            self.assertFalse((output / "video_manifest.json").exists(),
                             "The per-camera manifest must not be overwritten.")
            self.assertFalse((output / "global_person_assignments.csv").exists())
            self.assertTrue((output / "combined_video_manifest.json").is_file())
            self.assertTrue((output / "combined_video_assignments.csv").is_file())
            self.assertTrue((output / "manifest_signature.json").is_file())

            self.assertEqual(manifest["global_person_id"], global_id)
            self.assertEqual(manifest["width"], 1280)
            self.assertEqual(manifest["height"], 720)
            self.assertEqual(manifest["fps"], 10.0)
            self.assertEqual(manifest["frame_count"], 24)
            self.assertEqual(manifest["duration_seconds"], 2.4)
            self.assertEqual(manifest["grid"], {"rows": 2, "columns": 2})
            self.assertEqual(manifest["synchronisation"], COMBINED_SYNC_NOTE)
            by_camera = {row["camera_id"]: row for row in manifest["cameras"]}
            self.assertEqual(by_camera["camera1"]["local_track_ids"], "3, 7")
            self.assertEqual(by_camera["camera1"]["member_count"], 2)
            self.assertEqual(by_camera["camera2"]["local_track_ids"], "4")
            self.assertEqual(by_camera["camera3"]["local_track_ids"], "")
            self.assertEqual(by_camera["camera3"]["member_count"], 0)
            self.assertEqual(
                [(row["tile_row"], row["tile_column"])
                 for row in manifest["cameras"]],
                [(0, 0), (0, 1), (1, 0)],
            )

            frames = _read_frames(Path(manifest["video_path"]))
            self.assertEqual(len(frames), 24)
            capture = cv2.VideoCapture(manifest["video_path"])
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1280)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 720)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FPS)), 10)
            finally:
                capture.release()

            first = frames[0]
            tile_one = first[0:360, 0:640]
            self.assertGreater(_green(tile_one[:, :320]), 200)
            self.assertGreater(_green(tile_one[:, 320:]), 200,
                               "Both same-camera members must be boxed on one tile.")
            self.assertGreater(_green(first[0:360, 640:1280]), 200)
            self.assertLess(_green(first[360:720, 0:640]), 20,
                            "A camera without a confirmed member must stay unboxed.")
            self.assertLess(_green(first[360:720, 640:1280]), 20,
                            "The unused grid slot must stay unboxed.")

            self.assertGreater(_amber(frames[1][0:360, 0:640]), 200,
                               "A short prediction must be amber, not green.")
            self.assertEqual(_green(frames[1][0:360, 0:640]), 0)

            chained = frames[22]
            self.assertGreater(_amber(chained[0:360, 0:640]), 200,
                               "Camera 1 ended before the combined timeline.")
            self.assertGreater(_amber(chained[360:720, 0:640]), 200,
                               "Camera 3 ended before the combined timeline.")
            self.assertGreater(_green(chained[0:360, 640:1280]), 200,
                               "Camera 2 is still running at that frame.")

    def test_status_requires_a_matching_signature_and_registry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = _three_cameras(root)
            registry = root / "global_identity_registry.json"
            global_id = get_or_create_identity(registry, "camera1", 3, "Camera 1")
            render_combined_global_person_video(results, root, global_id)

            self.assertIsNotNone(combined_video_status(results, root, global_id))

            signature = (root / "global_person_videos" / global_id /
                         "manifest_signature.json")
            signature.unlink()
            self.assertIsNone(combined_video_status(results, root, global_id),
                              "A missing signature must invalidate the video.")

            render_combined_global_person_video(results, root, global_id)
            self.assertIsNotNone(combined_video_status(results, root, global_id))

            associate_confirmed_track(registry, global_id, "camera3", 9,
                                      "Camera 3", 0.5, "OSNet")
            self.assertIsNone(combined_video_status(results, root, global_id),
                              "A new member must invalidate the combined video.")

    def test_rerender_replaces_the_single_output_without_orphans(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = _three_cameras(root)
            registry = root / "global_identity_registry.json"
            global_id = get_or_create_identity(registry, "camera1", 3, "Camera 1")

            first = render_combined_global_person_video(results, root, global_id)
            second = render_combined_global_person_video(results, root, global_id)

            self.assertEqual(first["video_path"], second["video_path"])
            self.assertEqual(first["frame_count"], second["frame_count"])
            output = root / "global_person_videos" / global_id
            self.assertEqual(len(list(output.glob("*.mp4"))), 1)
            self.assertEqual(list(output.glob("*.partial.mp4")), [])

    def test_progress_callback_reports_every_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = [
                _write_camera(root, "camera1", "Camera 1", 160, 120, 12, 10.0,
                              _steady_rows(3, 12, (5, 10, 45, 100))),
            ]
            registry = root / "global_identity_registry.json"
            global_id = get_or_create_identity(registry, "camera1", 3, "Camera 1")
            seen = []
            manifest = render_combined_global_person_video(
                results, root, global_id,
                progress_callback=lambda current, total: seen.append((current, total)),
            )
            self.assertEqual(len(seen), manifest["frame_count"])
            self.assertEqual(seen[-1], (manifest["frame_count"], manifest["frame_count"]))

    def test_unknown_or_blank_global_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = _three_cameras(root)
            registry = root / "global_identity_registry.json"
            get_or_create_identity(registry, "camera1", 3, "Camera 1")
            with self.assertRaisesRegex(ValueError, "Choose a global person ID"):
                render_combined_global_person_video(results, root, "   ")
            with self.assertRaisesRegex(ValueError, "not a confirmed global identity"):
                render_combined_global_person_video(results, root, "GLOBAL-999")

    def test_missing_source_video_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = _three_cameras(root)
            registry = root / "global_identity_registry.json"
            global_id = get_or_create_identity(registry, "camera1", 3, "Camera 1")
            Path(results[2]["source_path"]).unlink()
            with self.assertRaisesRegex(ValueError, "Camera 3"):
                render_combined_global_person_video(results, root, global_id)


if __name__ == "__main__":
    unittest.main()
