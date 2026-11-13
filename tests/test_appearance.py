"""Tests for embedding ranking and manually labelled evaluation."""

import unittest

import numpy as np

from src.reid.appearance import (
    aggregate_track_embeddings, evaluate_pairs, rank_across_cameras, rank_candidates,
    validate_ground_truth_pairs,
)


class TestAppearanceMatching(unittest.TestCase):
    def setUp(self):
        self.fingerprints = {
            "camera1:7": np.array([1.0, 0.0, 0.0]),
            "camera2:2": np.array([0.95, 0.05, 0.0]),
            "camera2:5": np.array([0.0, 1.0, 0.0]),
            "camera2:9": np.array([0.8, 0.2, 0.0]),
        }

    def test_candidates_are_ranked_by_cosine_similarity(self):
        ranked = rank_candidates(
            self.fingerprints["camera1:7"],
            {key: value for key, value in self.fingerprints.items() if key.startswith("camera2:")},
        )
        self.assertEqual([item["key"] for item in ranked], ["camera2:2", "camera2:9", "camera2:5"])

    def test_dynamic_search_ranks_every_other_camera_and_keeps_empty_camera(self):
        vectors = {
            "camera1:7": np.array([1.0, 0.0]),
            "camera2:4": np.array([0.9, 0.1]),
            "camera3:2": np.array([0.0, 1.0]),
        }
        ranked = rank_across_cameras("camera1:7", vectors, ["camera1", "camera2", "camera3", "camera4"])
        self.assertEqual(list(ranked), ["camera2", "camera3", "camera4"])
        self.assertEqual(ranked["camera2"][0]["key"], "camera2:4")
        self.assertEqual(ranked["camera4"], [])

    def test_dynamic_ground_truth_evaluation_is_scoped_per_destination_camera(self):
        vectors = {
            "camera1:7": np.array([1.0, 0.0]),
            "camera2:4": np.array([0.9, 0.1]),
            "camera2:8": np.array([0.0, 1.0]),
            "camera3:5": np.array([0.95, 0.05]),
        }
        rows = [
            {"query_camera_id": "camera1", "query_track_id": "7",
             "candidate_camera_id": "camera2", "candidate_track_id": "4"},
            {"query_camera_id": "camera1", "query_track_id": "7",
             "candidate_camera_id": "camera3", "candidate_track_id": "5"},
        ]
        ids = {"camera1": {7}, "camera2": {4, 8}, "camera3": {5}}
        pairs = validate_ground_truth_pairs(rows, ids)
        metrics = evaluate_pairs(pairs, vectors)
        self.assertEqual(metrics["evaluated"], 2)
        self.assertEqual(metrics["top1_correct"], 2)
        self.assertEqual(metrics["top1_accuracy"], 1.0)

    def test_dynamic_ground_truth_rejects_same_camera_pair(self):
        with self.assertRaisesRegex(ValueError, "positive integers"):
            validate_ground_truth_pairs([{
                "query_camera_id": "camera1", "query_track_id": "7",
                "candidate_camera_id": "camera1", "candidate_track_id": "8",
            }], {"camera1": {7, 8}})

    def test_manual_pair_evaluation_counts_real_input_pairs(self):
        metrics = evaluate_pairs(
            [{"camera1_track_id": 7, "camera2_track_id": 9}], self.fingerprints
        )
        self.assertEqual(metrics["evaluated"], 1)
        self.assertEqual(metrics["top1_correct"], 0)
        self.assertEqual(metrics["top3_correct"], 1)
        self.assertEqual(metrics["top3_accuracy"], 1.0)
        self.assertEqual(metrics["top5_correct"], 1)
        self.assertAlmostEqual(metrics["mean_average_precision"], 0.5)
        self.assertAlmostEqual(metrics["mrr"], 1 / 2)

    def test_mean_quality_weighted_mean_and_medoid_are_normalized(self):
        vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        weighted = aggregate_track_embeddings(vectors, [0.9, 0.1], "quality_weighted_mean")
        mean = aggregate_track_embeddings(vectors, [0.9, 0.1], "mean")
        medoid = aggregate_track_embeddings(vectors, [0.9, 0.1], "medoid")
        self.assertGreater(weighted[0], weighted[1])
        np.testing.assert_allclose(np.linalg.norm(weighted), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(mean), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(medoid), 1.0, atol=1e-6)
        np.testing.assert_array_equal(medoid, vectors[0])

    def test_multiple_true_camera2_tracklets_count_once_per_identity(self):
        vectors = {
            "camera1:1": np.array([1.0, 0.0]),
            "camera2:2": np.array([0.8, 0.2]),
            "camera2:3": np.array([1.0, 0.0]),
            "camera2:4": np.array([0.0, 1.0]),
        }
        result = evaluate_pairs([
            {"camera1_track_id": 1, "camera2_track_id": 2},
            {"camera1_track_id": 1, "camera2_track_id": 3},
        ], vectors)
        self.assertEqual(result["evaluated"], 1)
        self.assertEqual(result["top1_correct"], 1)

    def test_missing_query_fingerprint_is_not_counted(self):
        metrics = evaluate_pairs(
            [{"camera1_track_id": 99, "camera2_track_id": 9}], self.fingerprints
        )
        self.assertEqual(metrics["evaluated"], 0)
        self.assertIsNone(metrics["top1_accuracy"])

    def test_missing_ground_truth_candidate_fingerprint_is_unmatchable(self):
        metrics = evaluate_pairs(
            [{"camera1_track_id": 7, "camera2_track_id": 77}], self.fingerprints
        )
        self.assertEqual(metrics["evaluated"], 0)
        self.assertEqual(metrics["missing_fingerprints"], 1)

    def test_explicit_no_match_labels_measure_rejection_and_false_matches(self):
        vectors = {
            "camera1:7": np.array([1.0, 0.0, 0.0]),
            "camera1:8": np.array([0.0, 0.0, 1.0]),
            "camera2:2": np.array([1.0, 0.0, 0.0]),
            "camera2:3": np.array([0.0, 1.0, 0.0]),
        }
        labels = [
            {"query_camera_id": "camera1", "query_track_id": "7",
             "candidate_camera_id": "camera2", "candidate_track_id": "2"},
            {"query_camera_id": "camera1", "query_track_id": "8",
             "candidate_camera_id": "camera2", "candidate_track_id": "NOT_IN_CAMERA2"},
        ]
        metrics = evaluate_pairs(labels, vectors, threshold=0.5)
        self.assertEqual(metrics["top1_correct"], 1)
        self.assertEqual(metrics["false_matches"], 0)
        self.assertEqual(metrics["missed_matches"], 0)
        self.assertEqual(metrics["no_match_correct"], 1)
        self.assertEqual(metrics["no_match_accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        rejected = evaluate_pairs(labels, vectors, threshold=0.0)
        self.assertEqual(rejected["false_matches"], 1)
        self.assertEqual(rejected["no_match_correct"], 0)

    def test_ground_truth_csv_rejects_duplicates_and_missing_ids(self):
        pair = {"camera1_track_id": "7", "camera2_track_id": "2"}
        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            validate_ground_truth_pairs([pair, pair], {7}, {2})
        with self.assertRaisesRegex(ValueError, "does not exist"):
            validate_ground_truth_pairs([pair], {7}, {9})

    def test_ground_truth_csv_rejects_blank_or_non_numeric_ids(self):
        with self.assertRaisesRegex(ValueError, "integer track IDs"):
            validate_ground_truth_pairs(
                [{"camera1_track_id": "seven", "camera2_track_id": "2"}], {7}, {2}
            )
        with self.assertRaisesRegex(ValueError, "missing a camera track ID"):
            validate_ground_truth_pairs(
                [{"camera1_track_id": "", "camera2_track_id": "2"}], {7}, {2}
            )

    def test_ground_truth_csv_accepts_explicit_camera_two_no_match(self):
        result = validate_ground_truth_pairs([{
            "query_camera_id": "camera1", "query_track_id": "7",
            "candidate_camera_id": "camera2", "candidate_track_id": "NOT_IN_CAMERA2",
        }], {"camera1": {7}, "camera2": {2}})
        self.assertEqual(result[0]["candidate_track_id"], "NOT_IN_CAMERA2")

    def test_fingerprint_generation_caps_to_eight_and_normalizes(self):
        import tempfile
        from pathlib import Path

        import cv2
        from src.reid.appearance import make_track_fingerprints

        class EmbeddingModel:
            def embed(self, source, **kwargs):
                return np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (len(source), 1))

        with tempfile.TemporaryDirectory() as folder:
            rows = []
            for index in range(12):
                image_path = Path(folder) / f"{index}.jpg"
                cv2.imwrite(str(image_path), np.full((10, 10, 3), index, dtype=np.uint8))
                rows.append({"camera_id": "camera1", "track_id": 9,
                             "frame_number": index, "quality_score": 1.0 - index / 100,
                             "path": str(image_path)})
            result = make_track_fingerprints(EmbeddingModel(), rows)
        np.testing.assert_allclose(result["camera1:9"], np.array([0.6, 0.8]), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
