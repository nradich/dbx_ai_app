"""Run: python -m unittest discover -s tests -p test_prepare_data.py -v"""

from collections import Counter
import contextlib
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image, ImageOps

from training.prepare_data import (
    ANNOTATIONS, CLASSES, assign_splits, crop_tail, duplicate_pairs,
    fingerprints, merge_duplicate_groups, prepare, sha256, validate_leakage,
)

ROOT = Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    def test_reviewed_groups_exact_deterministic_even_when_reordered(self):
        entries = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["records"]
        records = [{"id": e["source_path"], "airline": e["source_path"].split("/")[0],
                    "group": e["group"]} for e in entries]
        self.assertTrue(assign_splits(records))
        shuffled = copy.deepcopy(records)
        random.Random(83).shuffle(shuffled)
        self.assertTrue(assign_splits(shuffled))
        self.assertEqual({r["id"]: r["split"] for r in records},
                         {r["id"]: r["split"] for r in shuffled})
        for airline in CLASSES:
            self.assertEqual(Counter(r["split"] for r in records if r["airline"] == airline),
                             {"train": 6, "val": 2, "test": 2})
        for group in {r["group"] for r in records}:
            self.assertEqual(len({r["split"] for r in records if r["group"] == group}), 1)

    def test_infeasible_counts_do_not_break_group(self):
        records = [{"airline": "a", "group": "all"} for _ in range(10)]
        self.assertFalse(assign_splits(records, classes=("a",)))
        self.assertEqual(len({r["split"] for r in records}), 1)


class CropTests(unittest.TestCase):
    def test_crop_pixels_round_outward_and_strip_metadata(self):
        image = Image.new("RGB", (20, 10))
        image.putpixel((2, 1), (10, 20, 30))
        image.info["exif"] = b"source metadata"
        crop = crop_tail(image, [0.125, 0.15, 0.81, 0.91])
        self.assertEqual(crop.size, (15, 9))
        self.assertEqual(crop.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(crop.mode, "RGB")
        self.assertFalse(crop.info)

    def test_invalid_boxes(self):
        image = Image.new("RGB", (20, 10))
        for bbox in ([0, 0, 0, 1], [-0.1, 0, 1, 1], [0, 0, 1.1, 1],
                     [0, 0, 1, float("nan")], [0, 0, 1], [False, 0, 1, 1],
                     [0, 0, 0.01, 0.01], [0.8, 0, 0.2, 1]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                crop_tail(image, bbox)


class DuplicateTests(unittest.TestCase):
    def record(self, image, name):
        fp = fingerprints(image)
        return {"id": name, "airline": "alaska", "group": name,
                "source_sha256": "source-" + name, "sha256": "crop-" + name,
                "_fingerprints": {"source": fp, "crop": fp}}

    def test_perceptual_reencoded_duplicates_merge_transitively(self):
        with Image.open(ROOT / "airline_tails/alaska/alaska_tail_5.jpg") as original:
            image = crop_tail(original, [0.40, 0.12, 0.89, 0.63])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        with Image.open(io.BytesIO(buffer.getvalue())) as jpeg:
            records = [self.record(image, "a"), self.record(jpeg.convert("RGB"), "b")]
        findings = duplicate_pairs(records)
        self.assertEqual(len(findings), 1)
        self.assertTrue(any("perceptual" in reason for reason in findings[0]["reasons"]))
        third = self.record(image, "c")
        records.append(third)
        merge_duplicate_groups(records, findings + [{"ids": ["b", "c"], "reasons": []}])
        self.assertEqual(len({r["group"] for r in records}), 1)

    def test_exact_hash_and_decoded_pixel_duplicates_detected(self):
        image = Image.new("RGB", (30, 30), "blue")
        a, b = self.record(image, "a"), self.record(image, "b")
        a["source_sha256"] = b["source_sha256"]
        reasons = duplicate_pairs([a, b])[0]["reasons"]
        self.assertIn("source_bytes_exact", reasons)
        self.assertIn("source_pixels_exact", reasons)
        self.assertIn("crop_pixels_exact", reasons)

    def test_exact_leakage_rejected_without_findings(self):
        for key in ("source_sha256", "sha256", "group"):
            a = {"id": "a", "group": "a", "sha256": "a", "source_sha256": "a", "split": "train"}
            b = {"id": "b", "group": "b", "sha256": "b", "source_sha256": "b", "split": "test"}
            b[key] = a[key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cross-split"):
                validate_leakage([a, b], [])

    def test_perceptual_leakage_and_conflicting_labels_rejected(self):
        image = Image.new("RGB", (30, 30), "blue")
        a, b = self.record(image, "a"), self.record(image, "b")
        a["split"], b["split"] = "train", "test"
        findings = duplicate_pairs([a, b])
        with self.assertRaisesRegex(ValueError, "cross-split duplicate"):
            validate_leakage([a, b], findings)
        b["airline"] = "delta"
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            merge_duplicate_groups([a, b], findings)


class PreparationTests(unittest.TestCase):
    def test_full_dataset_metadata_pixels_attribution_and_immutability(self):
        source = ROOT / "airline_tails"
        before = {p.relative_to(source): sha256(p.read_bytes())
                  for p in source.rglob("*") if p.is_file()}
        local = ROOT / ".local"
        local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            output = Path(directory) / "prepared"
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare(source, output)
                self.assertEqual(prepare(source, output), manifest)
            self.assertEqual(len(manifest["records"]), 50)
            self.assertTrue(manifest["split_policy"]["exact_targets_met"])
            for r in manifest["records"]:
                self.assertFalse(Path(r["path"]).is_absolute())
                self.assertNotIn("\\", r["path"])
                path = output / r["path"]
                self.assertEqual(sha256(path.read_bytes()), r["sha256"])
                with Image.open(source / r["source_path"]) as original:
                    expected = crop_tail(ImageOps.exif_transpose(original), r["bbox"])
                with Image.open(path) as crop:
                    self.assertEqual(crop.mode, "RGB")
                    self.assertFalse(crop.getexif())
                    self.assertFalse(crop.info)
                    self.assertEqual(crop.size, expected.size)
                    self.assertEqual(crop.tobytes(), expected.tobytes())
            for c in CLASSES:
                self.assertEqual((source / c / "SOURCES.md").read_bytes(),
                                 (output / "sources" / c / "SOURCES.md").read_bytes())
                self.assertTrue((output / "review" / c / "review.jpg").is_file())
        self.assertEqual(before, {p.relative_to(source): sha256(p.read_bytes())
                                  for p in source.rglob("*") if p.is_file()})

    def test_wrong_counts_and_output_overlap_fail_before_writing(self):
        local = ROOT / ".local"
        local.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            source, output = Path(directory) / "source", Path(directory) / "output"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "exactly 50"):
                prepare(source, output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare(source, source / "output")


if __name__ == "__main__":
    unittest.main()
