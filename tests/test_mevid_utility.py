"""Tests for preserving MEVID labels separately from predicted track IDs."""

import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_mevid_two_camera import build_two_camera_tables, read_tracklets


class TestMEVIDPreparation(unittest.TestCase):
    def test_two_camera_pair_table_preserves_ground_truth_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "data"
            out = Path(folder) / "out"
            root.mkdir()
            (root / "train_name.txt").write_text(
                "0100O001C011T000F00001.jpg\n0100O001C011T000F00002.jpg\n"
                "0100O002C021T000F00001.jpg\n0100O002C021T000F00002.jpg\n"
                "0200O001C011T001F00001.jpg\n0201O001C021T002F00001.jpg\n",
                encoding="utf-8",
            )
            (root / "track_train_info.txt").write_text(
                "0 1 100 1 11\n2 3 100 2 21\n4 4 200 1 11\n5 5 201 1 21\n",
                encoding="utf-8",
            )
            tracklets = read_tracklets(root, "train")
            _, pairs_path, summary = build_two_camera_tables(
                tracklets, 11, 21, out, "train",
                ["https://example.test/clip.G11.avi", "https://example.test/clip.G21.avi",
                 "https://example.test/clip.G31.avi"],
            )
            with pairs_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            url_manifest_exists = Path(summary["video_url_manifest"]).is_file()
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(int(row["is_same_person"]) for row in rows), 1)
        self.assertEqual(summary["positive_pairs"], 1)
        self.assertTrue(summary["ground_truth_separate_from_predictions"])
        self.assertEqual(summary["matching_video_urls"], 2)
        self.assertTrue(url_manifest_exists)


if __name__ == "__main__":
    unittest.main()
