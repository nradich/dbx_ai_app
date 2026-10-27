"""Run: python -m unittest discover -s tests -p "test_prepare_data*.py" -v"""

from collections import Counter
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageOps

from training.prepare_data import (
    CLASSES, CONTEXT_SCALE, SPLITS, context_geometry, crop_tail, prepare, prepare_v2, sha256,
)

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
V2_ANNOTATIONS = ROOT / "dataset/tail_annotations_v2.json"
V1_FIELDS = ("id", "airline", "label", "source_path", "bbox", "group", "sha256",
             "source_sha256", "split", "path")
V2_FIELDS = ("source_root", "kind", "box_source", "context_path", "context_bbox",
             "context_sha256", "original_path", "original_sha256")


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


def tree(directory):
    return {p.relative_to(directory).as_posix(): sha256(p.read_bytes())
            for p in sorted(directory.rglob("*")) if p.is_file()}


def oriented(path):
    with Image.open(path) as original:
        return ImageOps.exif_transpose(original).convert("RGB")


def expected_context_size(size, bbox):
    """Independent restatement of the documented 2.5x-about-centre, clamped context box."""
    w, h = size
    left, top, right, bottom = bbox
    cx, cy = (left + right) / 2, (top + bottom) / 2
    hw, hh = (right - left) * CONTEXT_SCALE / 2, (bottom - top) * CONTEXT_SCALE / 2
    box = (math.floor(max(0.0, cx - hw) * w), math.floor(max(0.0, cy - hh) * h),
           math.ceil(min(1.0, cx + hw) * w), math.ceil(min(1.0, cy + hh) * h))
    return box[2] - box[0], box[3] - box[1]


class RealDatasetV2Tests(unittest.TestCase):
    """Builds the real dataset v2 (twice, for determinism) and dataset v1 once."""

    @classmethod
    def setUpClass(cls):
        LOCAL.mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=LOCAL)
        base = Path(cls.tmp.name)
        cls.spec = json.loads(V2_ANNOTATIONS.read_text(encoding="utf-8"))
        cls.roots = [ROOT / r for r in cls.spec["source_roots"]]
        cls.sources_before = {str(r): tree(r) for r in cls.roots}
        cls.output = base / "v2"
        cls.manifest = quiet(prepare_v2, V2_ANNOTATIONS, ROOT, cls.output)
        cls.records = cls.manifest["records"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_counts_match_annotations(self):
        self.assertEqual(len(self.records), 83)
        self.assertEqual(Counter(r["split"] for r in self.records),
                         {"train": 54, "val": 13, "test": 16})
        expected = Counter((e["airline"], e["split"]) for e in self.spec["records"])
        actual = Counter((r["airline"], r["split"]) for r in self.records)
        self.assertEqual(actual, expected)
        self.assertEqual(self.manifest["counts"],
                         {c: {s: actual[(c, s)] for s in SPLITS} for c in CLASSES})
        self.assertEqual(self.manifest["counts"]["delta"], {"train": 9, "val": 2, "test": 4})
        by_path = {e["source_path"]: e for e in self.spec["records"]}
        for r in self.records:
            e = by_path[r["source_path"]]
            for field in ("airline", "bbox", "group", "split", "kind", "box_source"):
                self.assertEqual(r[field], e[field], (r["id"], field))

    def test_manifest_fields(self):
        m = self.manifest
        self.assertEqual((m["version"], m["dataset_version"]), (2, "v2"))
        self.assertEqual(m["classes"], list(CLASSES))
        self.assertEqual(m["source_roots"], ["airline_tails", "additional_tail_photos"])
        self.assertEqual(m["parent_manifest_sha256"], self.spec["parent"]["manifest_sha256"])
        self.assertEqual(m["split_policy"]["method"], "explicit")
        self.assertEqual(m["context_scale"], 2.5)
        self.assertEqual(m["screen_checks"], {"against": ["full_plane_tests/manifest.json"],
                                              "images_checked": 10, "findings": []})
        self.assertEqual(m["duplicate_checks"]["records_checked"], 83)
        self.assertEqual(len({r["id"] for r in self.records}), 83)
        self.assertEqual([(r["airline"], r["id"]) for r in self.records],
                         sorted((r["airline"], r["id"]) for r in self.records))
        for r in self.records:
            self.assertEqual(tuple(r), V1_FIELDS + V2_FIELDS)
            self.assertEqual(r["label"], CLASSES.index(r["airline"]))
            self.assertEqual(r["source_root"], r["source_path"].split("/")[0])
            self.assertEqual(r["id"], Path(r["source_path"]).stem)
            self.assertEqual(r["path"], f"{r['split']}/{r['airline']}/{r['id']}.png")
            self.assertEqual(r["context_path"], "context/" + r["path"])
            full = r["kind"] == "full-plane"
            self.assertEqual(r["original_path"], "originals/" + r["path"] if full else None)
            self.assertEqual(r["original_sha256"] is None, not full)
        for pair in m["duplicate_checks"]["findings"]:
            self.assertEqual(len({r["split"] for r in self.records if r["id"] in pair["ids"]}), 1)

    def test_files_hashes_and_no_extra_outputs(self):
        files = tree(self.output)
        expected = {"manifest.json"}
        for r in self.records:
            for key in ("path", "context_path", "original_path"):
                if r[key]:
                    expected.add(r[key])
                    self.assertEqual(files[r[key]], r[key.replace("path", "sha256")], r[key])
            self.assertEqual(sha256((ROOT / r["source_path"]).read_bytes()), r["source_sha256"])
        expected |= {a for a in self.manifest["attribution"]}
        expected |= {f"review/{c}/review.jpg" for c in CLASSES}
        self.assertEqual(set(files), expected)
        self.assertEqual(sum(r["original_path"] is not None for r in self.records), 11)
        for root in self.spec["source_roots"]:
            for c in CLASSES:
                self.assertEqual((ROOT / root / c / "SOURCES.md").read_bytes(),
                                 (self.output / "sources" / root / c / "SOURCES.md").read_bytes())
        for c in CLASSES:
            n = sum(r["airline"] == c for r in self.records)
            with Image.open(self.output / "review" / c / "review.jpg") as sheet:
                self.assertEqual(sheet.size, (1000, math.ceil(n / 3) * 270))

    def test_pixels_context_originals_and_metadata(self):
        for r in self.records:
            with self.subTest(id=r["id"]):
                source = oriented(ROOT / r["source_path"])
                expected = crop_tail(source, r["bbox"])
                images = {}
                for key in ("path", "context_path", "original_path"):
                    if r[key]:
                        with Image.open(self.output / r[key]) as image:
                            self.assertEqual(image.format, "PNG")
                            self.assertEqual(image.mode, "RGB")
                            self.assertFalse(image.info)
                            self.assertFalse(image.getexif())
                            images[key] = image.copy()
                self.assertEqual(images["path"].tobytes(), expected.tobytes())
                context = images["context_path"]
                self.assertEqual(context.size, expected_context_size(source.size, r["bbox"]))
                self.assertTrue(all(round(v, 6) == v for v in r["context_bbox"]))
                self.assertEqual(crop_tail(context, r["context_bbox"]).tobytes(), expected.tobytes())
                if r["bbox"] == [0.0, 0.0, 1.0, 1.0]:
                    self.assertEqual(r["context_bbox"], [0, 0, 1, 1])
                    self.assertEqual(context.tobytes(), expected.tobytes())
                if r["original_path"]:
                    self.assertEqual(images["original_path"].tobytes(), source.tobytes())

    def test_v1_splits_preserved_parent_hash_and_determinism(self):
        base = Path(self.tmp.name)
        v1 = quiet(prepare, ROOT / "airline_tails", base / "v1")
        self.assertEqual(sha256((base / "v1" / "manifest.json").read_bytes()),
                         self.spec["parent"]["manifest_sha256"])
        v2_by_id = {r["id"]: r for r in self.records}
        self.assertEqual(len(v1["records"]), 50)
        for old in v1["records"]:
            new = v2_by_id[old["id"]]
            self.assertEqual((new["split"], new["bbox"], new["source_path"], new["sha256"]),
                             (old["split"], old["bbox"], "airline_tails/" + old["source_path"],
                              old["sha256"]))
        quiet(prepare_v2, V2_ANNOTATIONS, ROOT, base / "v2b")
        self.assertEqual((base / "v2b" / "manifest.json").read_bytes(),
                         (self.output / "manifest.json").read_bytes())
        self.assertEqual(tree(base / "v2b"), tree(self.output))
        self.assertEqual({str(r): tree(r) for r in self.roots}, self.sources_before)


def noise_image(seed, size=(48, 40)):
    data = random.Random(seed).randbytes(size[0] * size[1] * 3)
    return Image.frombytes("RGB", size, data)


def encode(image, fmt):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return buffer.getvalue()


class ContextGeometryTests(unittest.TestCase):
    def test_context_bbox_reproduces_tail_pixels_with_floor_ceil(self):
        rng = random.Random(7)
        cases = [((49, 49), [1 / 49, 1 / 49, 48 / 49, 48 / 49]), ((10, 10), [0.0, 0.0, 1.0, 1.0])]
        for _ in range(3000):
            size = (rng.randint(4, 3000), rng.randint(4, 3000))
            a, b = sorted(rng.random() for _ in range(2))
            c, d = sorted(rng.random() for _ in range(2))
            cases.append((size, [round(a, rng.randint(2, 6)), round(c, 3), b, d]))
        checked = 0
        for size, bbox in cases:
            w, h = size
            tail = (math.floor(bbox[0] * w), math.floor(bbox[1] * h),
                    math.ceil(bbox[2] * w), math.ceil(bbox[3] * h))
            if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]) or min(tail[2] - tail[0], tail[3] - tail[1]) < 2:
                continue
            got_tail, context, norm = context_geometry(size, bbox)
            self.assertEqual(got_tail, tail)
            cw, ch = context[2] - context[0], context[3] - context[1]
            self.assertEqual((cw, ch), expected_context_size(size, bbox))
            self.assertEqual((math.floor(norm[0] * cw), math.floor(norm[1] * ch),
                              math.ceil(norm[2] * cw), math.ceil(norm[3] * ch)),
                             (tail[0] - context[0], tail[1] - context[1],
                              tail[2] - context[0], tail[3] - context[1]), (size, bbox))
            self.assertTrue(all(round(v, 6) == v and 0 <= v <= 1 for v in norm))
            checked += 1
        self.assertGreater(checked, 1000)
        self.assertEqual(context_geometry((10, 10), [0.0, 0.0, 1.0, 1.0])[2], [0, 0, 1, 1])


class SyntheticV2Tests(unittest.TestCase):
    """Tiny fake repository: two roots x five classes plus a screening manifest."""

    def setUp(self):
        LOCAL.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=LOCAL)
        self.repo = Path(self.tmp.name) / "repo"
        self.output = Path(self.tmp.name) / "out"
        records, v1, seed = [], [], 0
        for c in CLASSES:
            for root, name, split, kind, bbox, ext in (
                    ("airline_tails", f"{c}_a", "train", "tail-photo", [0.1, 0.1, 0.9, 0.9], "png"),
                    ("airline_tails", f"{c}_b", "test", "tail-photo", [0.0, 0.0, 1.0, 1.0], "png"),
                    ("extra", f"{c}_x", "val", "full-plane", [0.4, 0.35, 0.62, 0.6], "JPG")):
                seed += 1
                self.write_image(f"{root}/{c}/{name}.{ext}", noise_image(seed))
                records.append({"source_path": f"{root}/{c}/{name}.{ext}", "airline": c,
                                "bbox": bbox, "group": f"group-{name}", "split": split,
                                "kind": kind, "box_source": "test"})
                if root == "airline_tails":
                    v1.append({"source_path": f"{c}/{name}.{ext}", "bbox": bbox, "group": name})
            for root in ("airline_tails", "extra"):
                (self.repo / root / c / "SOURCES.md").write_text(f"{root} {c}\n", encoding="utf-8")
        self.write_image("screen/images/alaska/screen_1.png", noise_image(1000))
        self.write_json("screen/manifest.json", {"version": 1, "records": [
            {"id": "screen_1", "airline": "alaska", "path": "images/alaska/screen_1.png",
             "bbox": [0.2, 0.2, 0.8, 0.8]}]})
        self.write_json("dataset/v1.json", {"version": 1, "records": v1})
        self.spec = {"version": 2, "source_roots": ["airline_tails", "extra"],
                     "parent": {"annotations": "dataset/v1.json", "manifest_sha256": "x" * 64},
                     "split_policy": "test policy", "screen_against": ["screen/manifest.json"],
                     "records": records}

    def tearDown(self):
        self.tmp.cleanup()

    def write_image(self, relative, image):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode(image, "JPEG" if path.suffix.lower() == ".jpg" else "PNG"))

    def write_json(self, relative, data):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def build(self, spec=None, output=None):
        self.write_json("dataset/v2.json", spec or self.spec)
        return quiet(prepare_v2, self.repo / "dataset/v2.json", self.repo, output or self.output)

    def assert_fails(self, pattern, spec=None):
        with self.assertRaisesRegex(ValueError, pattern):
            self.build(spec)
        self.assertFalse(self.output.exists())

    def record(self, spec, name):
        return next(r for r in spec["records"] if Path(r["source_path"]).stem == name)

    def test_valid_fixture_builds_idempotently_and_refuses_changes(self):
        manifest = self.build()
        self.assertEqual(len(manifest["records"]), 15)
        self.assertEqual(manifest["counts"], {c: {"train": 1, "val": 1, "test": 1} for c in CLASSES})
        self.assertEqual(manifest["duplicate_checks"]["findings"], [])
        self.assertEqual(manifest["screen_checks"],
                         {"against": ["screen/manifest.json"], "images_checked": 1, "findings": []})
        self.assertEqual(manifest["split_policy"], {"method": "explicit", "note": "test policy"})
        before = tree(self.output)
        self.assertEqual(self.build(), manifest)
        self.assertEqual(tree(self.output), before)
        for r in manifest["records"]:
            source = oriented(self.repo / r["source_path"])
            with Image.open(self.output / r["context_path"]) as context:
                self.assertEqual(context.size, expected_context_size(source.size, r["bbox"]))
                self.assertEqual(crop_tail(context, r["context_bbox"]).tobytes(),
                                 crop_tail(source, r["bbox"]).tobytes())
            self.assertEqual(r["original_path"] is not None, r["kind"] == "full-plane")
        changed = copy.deepcopy(self.spec)
        self.record(changed, "delta_a")["box_source"] = "changed"
        with self.assertRaisesRegex(ValueError, "nonempty output differs"):
            self.build(changed)
        with self.assertRaisesRegex(ValueError, "separate"):
            self.build(output=self.repo / "extra" / "out")
        with self.assertRaisesRegex(ValueError, "separate"):
            self.build(output=self.repo)

    def test_cli_dispatches_version_2(self):
        self.write_json("dataset/v2.json", self.spec)
        result = subprocess.run(
            [sys.executable, str(ROOT / "training/prepare_data.py"), "--annotations",
             str(self.repo / "dataset/v2.json"), "--repo-root", str(self.repo),
             "--output", str(self.output)], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["version"], 2)

    def test_unannotated_file(self):
        self.write_image("extra/united/united_new.png", noise_image(500))
        self.assert_fails(r"unannotated=\['extra/united/united_new.png'\]")

    def test_missing_annotated_file(self):
        (self.repo / "extra/united/united_x.JPG").unlink()
        self.assert_fails(r"missing=\['extra/united/united_x.JPG'\]")

    def test_unknown_airline_and_folder_mismatch(self):
        spec = copy.deepcopy(self.spec)
        self.record(spec, "alaska_a")["airline"] = "jetblue"
        self.assert_fails("unknown airline 'jetblue'", spec)
        self.record(spec, "alaska_a")["airline"] = "delta"
        self.assert_fails("airline 'delta' does not match folder 'alaska'", spec)

    def test_invalid_fields(self):
        for field, value, pattern in (("split", "holdout", "split must be one of"),
                                      ("kind", "nose", "kind must be one of"),
                                      ("group", " ", "group must be a nonempty string"),
                                      ("bbox", [0.5, 0, 0.4, 1], "bad normalized bbox"),
                                      ("bbox", [0, 0, 0.01, 1], "degenerate"),
                                      ("source_path", "other/alaska/alaska_x.JPG", "not under a source root"),
                                      ("source_path", "extra/../x.png", "portable"),
                                      ("source_path", "extra/alaska/sub/alaska_x.JPG", "expected <root>/<airline>/<file>")):
            with self.subTest(field=field, value=value):
                spec = copy.deepcopy(self.spec)
                self.record(spec, "alaska_x")[field] = value  # not a v1 (parent) record
                self.assert_fails(pattern, spec)
        self.assert_fails("version must be 2", dict(self.spec, version=1))

    def test_group_spanning_splits(self):
        spec = copy.deepcopy(self.spec)
        self.record(spec, "alaska_b")["group"] = "group-alaska_a"
        self.assert_fails(r"groups span splits: \{'group-alaska_a': \['test', 'train'\]\}", spec)

    def test_duplicate_ids_across_roots(self):
        self.write_image("extra/alaska/alaska_a.png", noise_image(600))
        spec = copy.deepcopy(self.spec)
        spec["records"].append({"source_path": "extra/alaska/alaska_a.png", "airline": "alaska",
                                "bbox": [0, 0, 1, 1], "group": "g", "split": "train",
                                "kind": "tail-crop", "box_source": "test"})
        self.assert_fails("duplicate image ID 'alaska_a'", spec)

    def add_copy(self, split):
        shutil.copyfile(self.repo / "airline_tails/alaska/alaska_a.png",
                        self.repo / "extra/alaska/alaska_copy.png")
        spec = copy.deepcopy(self.spec)
        spec["records"].append({"source_path": "extra/alaska/alaska_copy.png", "airline": "alaska",
                                "bbox": [0.1, 0.1, 0.9, 0.9], "group": "copy", "split": split,
                                "kind": "tail-crop", "box_source": "test"})
        return spec

    def test_duplicate_across_splits_rejected(self):
        self.assert_fails(r"cross-split duplicate \(train vs test\).*alaska_copy", self.add_copy("test"))

    def test_duplicate_within_split_recorded(self):
        manifest = self.build(self.add_copy("train"))
        findings = manifest["duplicate_checks"]["findings"]
        self.assertEqual([f["ids"] for f in findings], [["alaska_a", "alaska_copy"]])
        self.assertIn("source_bytes_exact", findings[0]["reasons"])
        self.assertIn("crop_pixels_exact", findings[0]["reasons"])

    def test_screen_duplicate_rejected(self):
        # A re-encoded copy of a dataset source in the screening set must be caught.
        image = oriented(self.repo / "extra/delta/delta_x.JPG")
        (self.repo / "screen/images/alaska/screen_1.png").write_bytes(encode(image, "PNG"))
        self.assert_fails(r"duplicate screened evaluation images.*delta_x.*screen_1.*source_pixels_exact")

    def test_parent_bbox_and_membership(self):
        v1 = json.loads((self.repo / "dataset/v1.json").read_text())
        v1["records"][0]["bbox"] = [0.2, 0.2, 0.8, 0.8]
        self.write_json("dataset/v1.json", v1)
        self.assert_fails("v1 bbox changed for airline_tails/alaska/alaska_a.png")
        v1["records"][0] = {"source_path": "alaska/gone.png", "bbox": [0, 0, 1, 1], "group": "g"}
        self.write_json("dataset/v1.json", v1)
        self.assert_fails("v1 record missing from v2: airline_tails/alaska/gone.png")


if __name__ == "__main__":
    unittest.main()
