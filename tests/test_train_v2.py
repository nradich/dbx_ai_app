"""Torch-free tests for training/metrics_utils.py and the v2 workload files.

Run: python -m unittest discover -s tests -p test_train_v2.py -v
"""
import math
import random
import unittest
from pathlib import Path

from training.metrics_utils import (
    brier_score, calibration_bins, expected_calibration_error, jittered_context_box,
    per_class_metrics, sample_context_box,
)

ROOT = Path(__file__).resolve().parents[1]


class ContextBoxTests(unittest.TestCase):
    def test_scale_one_no_shift_returns_tail_box(self):
        self.assertEqual(jittered_context_box((0.3, 0.3, 0.7, 0.7), (100, 200), 1.0, 0, 0), (30, 60, 70, 140))

    def test_scale_about_centre(self):
        # 40x40 box centred at (50, 50) scaled 2x -> 80x80.
        self.assertEqual(jittered_context_box((0.3, 0.3, 0.7, 0.7), (100, 100), 2.0, 0, 0), (10, 10, 90, 90))

    def test_shift_is_ten_percent_of_tail_box(self):
        self.assertEqual(jittered_context_box((0.3, 0.3, 0.7, 0.7), (100, 100), 1.0, 1, -1), (34, 26, 74, 66))

    def test_clamped_to_image(self):
        self.assertEqual(jittered_context_box((0.0, 0.5, 0.4, 1.0), (100, 100), 2.0, -1, 1), (0, 30, 56, 100))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            jittered_context_box((0.5, 0.2, 0.4, 0.8), (10, 10), 1.0, 0, 0)
        with self.assertRaises(ValueError):
            jittered_context_box((0.1, 0.1, 0.4, 0.8), (10, 10), 1.0, 2, 0)
        with self.assertRaises(ValueError):
            sample_context_box((0.1, 0.1, 0.4, 0.8), (10, 10), random.Random(0), (2.0, 1.0))

    def test_sampled_boxes_contain_shifted_tail_and_stay_in_bounds(self):
        rng = random.Random(42)
        bbox, size = (0.3, 0.3, 0.7, 0.7), (250, 250)  # Tail box expanded 2.5x as in dataset v2.
        for _ in range(500):
            x0, y0, x1, y1 = sample_context_box(bbox, size, rng, (1.0, 2.0))
            self.assertTrue(0 <= x0 < x1 <= 250 and 0 <= y0 < y1 <= 250)
            # Scale >= 1 and shift <= 10% of the box: the central 80% of the tail stays inside.
            self.assertTrue(x0 <= 75 + 10 and x1 >= 175 - 10 and y0 <= 85 and y1 >= 165)
            self.assertLessEqual(x1 - x0, math.ceil(100 * 2.0) + 2)

    def test_sampling_is_deterministic_for_a_seed(self):
        a = [sample_context_box((0.3, 0.3, 0.7, 0.7), (250, 250), r, (1.0, 2.0))
             for r in [random.Random(7)] for _ in range(20)]
        b = [sample_context_box((0.3, 0.3, 0.7, 0.7), (250, 250), r, (1.0, 2.0))
             for r in [random.Random(7)] for _ in range(20)]
        self.assertEqual(a, b)


class CalibrationTests(unittest.TestCase):
    def test_perfectly_confident_correct(self):
        probs, labels = [[1.0, 0.0], [0.0, 1.0]], [0, 1]
        self.assertEqual(expected_calibration_error(probs, labels), 0.0)
        self.assertEqual(brier_score(probs, labels), 0.0)

    def test_ece_known_value(self):
        # Bin [0.6, 0.8): confidences 0.7, 0.7 with one correct -> |0.5 - 0.7| = 0.2, weight 2/3.
        # Bin [0.8, 1.0]: confidence 0.9, correct -> |1 - 0.9| = 0.1, weight 1/3.
        probs = [[0.7, 0.3], [0.7, 0.3], [0.1, 0.9]]
        labels = [0, 1, 1]
        self.assertAlmostEqual(expected_calibration_error(probs, labels), 2 / 3 * 0.2 + 1 / 3 * 0.1)
        bins = calibration_bins(probs, labels)
        self.assertEqual([b["count"] for b in bins], [0, 0, 0, 2, 1])
        self.assertAlmostEqual(bins[3]["accuracy"], 0.5)
        self.assertIsNone(bins[0]["accuracy"])

    def test_confidence_one_goes_to_last_bin(self):
        self.assertEqual(calibration_bins([[1.0, 0.0]], [0])[-1]["count"], 1)

    def test_brier_known_value(self):
        # (0.7-1)^2 + 0.3^2 = 0.18 ; (0.7-0)^2 + (0.3-1)^2 = 0.98 -> mean 0.58
        self.assertAlmostEqual(brier_score([[0.7, 0.3], [0.7, 0.3]], [0, 1]), 0.58)

    def test_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([], [])
        with self.assertRaises(ValueError):
            brier_score([], [])


class PerClassTests(unittest.TestCase):
    def test_precision_recall_f1(self):
        rows = per_class_metrics([0, 0, 1, 2], [0, 1, 1, 1], ["a", "b", "c"])
        by = {r["class"]: r for r in rows}
        self.assertEqual((by["a"]["precision"], by["a"]["recall"], by["a"]["support"]), (1.0, 0.5, 2))
        self.assertAlmostEqual(by["a"]["f1"], 2 / 3)
        self.assertAlmostEqual(by["b"]["precision"], 1 / 3)
        self.assertEqual(by["b"]["recall"], 1.0)
        self.assertEqual((by["c"]["precision"], by["c"]["recall"], by["c"]["f1"]), (0.0, 0.0, 0.0))

    def test_matches_sklearn_when_available(self):
        try:
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = random.Random(3)
        labels = [rng.randrange(4) for _ in range(60)]
        predicted = [rng.randrange(4) for _ in range(60)]
        p, r, f, s = precision_recall_fscore_support(labels, predicted, labels=range(4), zero_division=0)
        for k, row in enumerate(per_class_metrics(labels, predicted, list("wxyz"))):
            self.assertAlmostEqual(row["precision"], p[k])
            self.assertAlmostEqual(row["recall"], r[k])
            self.assertAlmostEqual(row["f1"], f[k])
            self.assertEqual(row["support"], s[k])


class WorkloadTests(unittest.TestCase):
    def test_workloads_ship_metrics_utils(self):
        for name in ("workload.yaml", "workload_v2.yaml", "workload_v2_improved.yaml"):
            text = (ROOT / "training" / name).read_text(encoding="utf-8")
            self.assertIn("- metrics_utils.py", text, name)

    def test_v2_workloads_parse(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        base = yaml.safe_load((ROOT / "training/workload_v2.yaml").read_text(encoding="utf-8"))
        improved = yaml.safe_load((ROOT / "training/workload_v2_improved.yaml").read_text(encoding="utf-8"))
        for workload in (base, improved):
            p = workload["parameters"]
            self.assertTrue(p["dataset_root"].endswith("/dataset_v2"))
            self.assertEqual((p["alias"], p["latest_pointer"], workload["timeout_minutes"]), ("", "", 30))
            self.assertEqual((p["batch_size"], p["learning_rate"], p["seed"]), (16, 0.002, 42))
        self.assertEqual(base["parameters"]["recipe"], "v1")
        self.assertEqual(base["parameters"]["epochs"], 30)
        self.assertFalse(base["parameters"]["context_jitter"])
        p = improved["parameters"]
        self.assertEqual((p["recipe"], p["epochs"], p["context_jitter"], p["lr_schedule"]),
                         ("improved", 40, True, "cosine"))
        self.assertEqual((p["label_smoothing"], p["fine_tune_layer4_lr"], p["random_erasing_p"]), (0.1, 0.0001, 0.25))
        self.assertEqual(p["color_jitter"], [0.3, 0.3, 0.2, 0.02])
        self.assertEqual((p["crop_scale"], p["crop_ratio"], p["rotation_degrees"]), ([0.7, 1.0], [0.8, 1.25], 10))


if __name__ == "__main__":
    unittest.main()
