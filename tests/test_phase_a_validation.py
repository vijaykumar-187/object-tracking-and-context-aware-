"""Pure tests for the Phase A validation harness helpers.

These tests deliberately avoid OpenCV and Ultralytics. The harness keeps heavy
imports inside ``main()`` so the retention/settings/metrics logic can be tested
in a minimal Python environment.
"""

import json
import tempfile
import unittest
from pathlib import Path

import phaseA_validation as pv


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def legacy_tracks_payload(**overrides) -> dict:
    payload = {
        "camera_id": "camera1",
        "max_lost_frames": 30,
        "tracker_settings": {
            "max_age_frames": 30,
            "max_center_distance": 2.5,
            "iou_threshold": 0.05,
        },
        "detector_settings": {
            "image_size": 640,
            "confidence_threshold": 0.4,
            "frame_interval": 10,
            "nms_iou_threshold": 0.7,
        },
        "tracks": [],
    }
    payload.update(overrides)
    return payload


class TestRetentionSemantics(unittest.TestCase):
    def test_equivalent_retention_uses_integer_division(self):
        self.assertEqual(pv.equivalent_retention_opportunities(30, 10), 3)
        self.assertEqual(pv.equivalent_retention_opportunities(5, 10), 1)
        self.assertEqual(pv.equivalent_retention_opportunities(1, 10), 1)
        self.assertEqual(pv.equivalent_retention_opportunities(30, 1), 30)

    def test_invalid_retention_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            pv.equivalent_retention_opportunities(0, 10)
        with self.assertRaises(ValueError):
            pv.equivalent_retention_opportunities(30, 0)

    def test_baseline_context_reads_legacy_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "camera1" / "tracks.json", legacy_tracks_payload())

            context = pv.baseline_tracker_context(root)

            self.assertEqual(context["legacy_max_lost_frames"], 30)
            self.assertEqual(context["frame_interval"], 10)
            self.assertEqual(context["equivalent_retention_opportunities"], 3)
            self.assertEqual(context["equivalent_retention_max_age_frames"], 30)
            self.assertFalse(context["baseline_records_min_hits"])
            self.assertIn("raw video frames", context["semantics"])

    def test_baseline_context_rejects_missing_retention_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = legacy_tracks_payload()
            del payload["max_lost_frames"]
            del payload["tracker_settings"]["max_age_frames"]
            write_json(root / "camera1" / "tracks.json", payload)

            with self.assertRaisesRegex(ValueError, "legacy raw-frame retention"):
                pv.baseline_tracker_context(root)

    def test_build_run_configurations_distinguishes_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "camera1" / "tracks.json", legacy_tracks_payload())
            context = pv.baseline_tracker_context(root)

            configs = pv.build_run_configurations(context)

            self.assertEqual(
                [config["key"] for config in configs],
                ["equivalent_retention", "new_default"],
            )
            self.assertEqual(configs[0]["settings"]["tracker_max_age"], 3)
            self.assertEqual(configs[0]["max_age_frames"], 30)
            self.assertEqual(configs[1]["settings"]["tracker_max_age"], 30)
            self.assertEqual(configs[1]["max_age_frames"], 300)
            for config in configs:
                self.assertEqual(config["settings"]["frame_interval"], 10)
                self.assertEqual(config["settings"]["tracker_min_hits"], 2)
                self.assertEqual(config["settings"]["confidence_threshold"], 0.4)
                self.assertEqual(config["settings"]["nms_iou_threshold"], 0.7)


class TestMetrics(unittest.TestCase):
    def test_metrics_from_run_computes_fragmentation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "camera1"
            write_json(camera / "tracks.json", {
                "tracks": [
                    {"track_id": 1, "observed_frames": 1},
                    {"track_id": 2, "observed_frames": 2},
                    {"track_id": 3, "observed_frames": 5},
                    {"track_id": 4, "observed_frames": 8},
                ],
            })
            write_json(camera / "statistics.json", {
                "crops_saved": 12,
                "total_detections": 20,
                "frames_read": 100,
                "frames_processed": 10,
                "processing_seconds": 3.5,
                "track_count": 99,  # tracks.json is authoritative
            })
            write_json(camera / "performance.json", {
                "detector_inference_seconds": 1.25,
            })

            metrics = pv.metrics_from_run(root, "camera1")

            self.assertEqual(metrics["track_count"], 4)
            self.assertEqual(metrics["tracks_1_2_obs"], 2)
            self.assertEqual(metrics["short_share_pct"], 50.0)
            self.assertEqual(metrics["median_lifespan_obs"], 3.5)
            self.assertEqual(metrics["mean_lifespan_obs"], 4.0)
            self.assertEqual(metrics["max_lifespan_obs"], 8)
            self.assertEqual(metrics["crops_saved"], 12)
            self.assertEqual(metrics["total_detections"], 20)
            self.assertEqual(metrics["frames_read"], 100)
            self.assertEqual(metrics["frames_processed"], 10)
            self.assertEqual(metrics["detector_seconds"], 1.25)
            self.assertEqual(metrics["processing_seconds"], 3.5)

    def test_metrics_from_run_handles_no_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "camera1"
            write_json(camera / "tracks.json", {"tracks": []})
            write_json(camera / "statistics.json", {"processing_seconds": 1.0})

            metrics = pv.metrics_from_run(root, "camera1")

            self.assertEqual(metrics["track_count"], 0)
            self.assertEqual(metrics["short_share_pct"], 0.0)
            self.assertEqual(metrics["median_lifespan_obs"], 0.0)
            self.assertEqual(metrics["mean_lifespan_obs"], 0.0)
            self.assertEqual(metrics["max_lifespan_obs"], 0)
            self.assertEqual(metrics["detector_seconds"], 0.0)

    def test_metrics_from_run_fails_loudly_when_tracks_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                pv.metrics_from_run(Path(directory), "camera1")

    def test_comparison_rows_include_deltas_and_missing_values(self):
        baseline = {"track_count": 10, "short_share_pct": 20.0}
        new = {"track_count": 7, "short_share_pct": 10.0}

        rows = pv.comparison_rows(baseline, new)
        by_metric = {row["metric"]: row for row in rows}

        self.assertEqual(by_metric["track_count"]["delta"], -3)
        self.assertEqual(by_metric["short_share_pct"]["delta"], -10.0)
        self.assertIsNone(by_metric["crops_saved"]["baseline"])
        self.assertIsNone(by_metric["crops_saved"]["delta"])

    def test_markdown_report_states_both_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "camera1" / "tracks.json", legacy_tracks_payload())
            context = pv.baseline_tracker_context(root)
            configs = pv.build_run_configurations(context)
            comparison = {
                config["key"]: {
                    camera: {
                        "baseline": {},
                        "new": {},
                        "rows": pv.comparison_rows({}, {}),
                    }
                    for camera in pv.CAMERAS
                }
                for config in configs
            }

            report = pv.render_markdown_report(context, configs, comparison)

            self.assertIn("raw-frame", report)
            self.assertIn("Equivalent retention", report)
            self.assertIn("New default retention", report)
            self.assertIn("No improvement is claimed", report)


if __name__ == "__main__":
    unittest.main()
