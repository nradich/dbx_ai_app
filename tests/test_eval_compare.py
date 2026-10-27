"""Tests for evaluation/compare.py, evaluation/eval_metrics.py and evaluation/publish.py.

No torch, MLflow, pandas or network: compare.py imports those lazily inside main(), so the
image preparation, eval-set building, scoring (with a fake predictor), row building and
registry fallbacks (with a fake MLflow client) are exercised directly. publish.py runs
against a fake WorkspaceClient.
"""
import base64
import hashlib
import io
import json
import math
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

import compare  # noqa: E402
import eval_metrics as em  # noqa: E402
import publish  # noqa: E402

EVAL_ID = "0123456789abcdef0123456789abcdef"


def sql_tables():
    """{table: [(column, type)]} parsed from app_setup/create_eval_tables.sql."""
    text = (ROOT / "app_setup" / "create_eval_tables.sql").read_text(encoding="utf-8")
    tables = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS [\w.]*?\.(\w+) \((.*?)\n\) USING DELTA", text, re.S):
        tables[name] = re.findall(r"^\s+(\w+) (STRING|TIMESTAMP|DOUBLE|BOOLEAN|INT)\b", body, re.M)
    return tables


def png(color, size=(40, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class MetricsTest(unittest.TestCase):
    classes = ["a", "b", "c"]
    rows = [
        {"actual": "a", "predicted": "a", "confidence": 0.6, "probabilities": {"a": 0.6, "b": 0.3, "c": 0.1}},
        {"actual": "a", "predicted": "b", "confidence": 0.5, "probabilities": {"a": 0.2, "b": 0.5, "c": 0.3}},
        {"actual": "b", "predicted": "b", "confidence": 1.0, "probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}},
    ]

    def test_hand_computed_metrics(self):
        m, confusion = em.compute_metrics(self.rows, self.classes)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        # a: P=1, R=.5, F1=2/3; b: P=.5, R=1, F1=2/3; c absent everywhere -> 0 (zero_division=0)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 2 / 3 + 0) / 3)
        self.assertAlmostEqual(m["nll"], -(math.log(0.6) + math.log(0.2) + math.log(1.0)) / 3)
        self.assertAlmostEqual(m["brier"], (0.26 + 0.98 + 0.0) / 3)
        self.assertAlmostEqual(m["ece"], (abs(1 - 0.6) + abs(0 - 0.5) + abs(1 - 1.0)) / 3)
        self.assertAlmostEqual(m["mean_confidence"], 0.7)
        self.assertAlmostEqual(m["mean_true_prob"], 0.6)
        self.assertAlmostEqual(m["share_pred_a"], 1 / 3)
        self.assertAlmostEqual(m["share_pred_b"], 2 / 3)
        self.assertEqual(m["share_pred_c"], 0.0)
        self.assertEqual((m["precision_a"], m["recall_a"]), (1.0, 0.5))
        self.assertEqual((m["precision_b"], m["recall_b"]), (0.5, 1.0))
        self.assertEqual((m["f1_c"], m["precision_c"], m["recall_c"], m["support_c"]), (0, 0, 0, 0))
        self.assertEqual((m["support_a"], m["support_b"]), (2.0, 1.0))
        self.assertEqual(confusion, [[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertTrue(all(isinstance(v, float) for v in m.values()))
        expected = (["accuracy", "macro_f1", "nll", "brier", "ece", "mean_confidence", "mean_true_prob"]
                    + [f"share_pred_{c}" for c in self.classes]
                    + [f"{k}_{c}" for c in self.classes for k in ("f1", "precision", "recall", "support")])
        self.assertEqual(list(m), expected)

    def test_macro_f1_counts_absent_classes_as_zero(self):
        rows = [{"actual": "a", "predicted": "a", "confidence": 0.9,
                 "probabilities": {"a": 0.9, "b": 0.05, "c": 0.05}}]
        m, _ = em.compute_metrics(rows, self.classes)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1 / 3)

    def test_nll_floor(self):
        rows = [{"actual": "c", "predicted": "b", "confidence": 1.0, "probabilities": {"a": 0.0, "b": 1.0, "c": 0.0}}]
        m, _ = em.compute_metrics(rows, self.classes)
        self.assertAlmostEqual(m["nll"], -math.log(1e-12))
        self.assertAlmostEqual(m["brier"], 2.0)

    def test_ece_bins_and_edges(self):
        self.assertEqual([em.ece_bin(c) for c in (0.0, 0.1999, 0.2, 0.39, 0.4, 0.6, 0.8, 0.99, 1.0)],
                         [0, 0, 1, 1, 2, 3, 4, 4, 4])
        with self.assertRaises(ValueError):
            em.ece_bin(1.01)
        # 0.2 and 0.3 share bin 1 (one correct): |0.5 - 0.25| * 2/3; 1.0 alone in bin 4 correct: 0.
        self.assertAlmostEqual(em.expected_calibration_error([0.2, 0.3, 1.0], [True, False, True]),
                               (2 / 3) * 0.25)

    def test_rejects_empty_and_unknown(self):
        with self.assertRaises(ValueError):
            em.compute_metrics([], self.classes)
        with self.assertRaises(ValueError):
            em.compute_metrics([{**self.rows[0], "predicted": "z"}], self.classes)


class GeometryTest(unittest.TestCase):
    def test_scale_about_centre(self):
        self.assertEqual(em.scaled_box((0.4, 0.4, 0.6, 0.5), 2.0), (0.3, 0.35, 0.7, 0.55))
        self.assertEqual(em.scaled_box((0.4, 0.4, 0.6, 0.5), 1.5), (0.35, 0.375, 0.65, 0.525))

    def test_clamped_to_image(self):
        self.assertEqual(em.scaled_box((0.0, 0.1, 0.2, 0.3), 3.0), (0.0, 0.0, 0.4, 0.5))
        self.assertEqual(em.scaled_box((0.7, 0.6, 1.0, 1.0), 3.0), (0.4, 0.2, 1.0, 1.0))

    def test_scale_one_is_identity_and_rounding_outward(self):
        box = (0.075, 0.3, 0.245, 0.565)
        self.assertEqual(em.scaled_box(box, 1.0), box)
        self.assertEqual(em.crop_pixels((1920, 1440), box), (144, 432, 471, 814))
        self.assertEqual(em.crop_pixels((10, 10), (0.15, 0.15, 0.85, 0.85)), (1, 1, 9, 9))
        self.assertEqual(em.crop_pixels((10, 10), (0, 0, 1, 1)), (0, 0, 10, 10))

    def test_invalid(self):
        for bad in ((0.5, 0.5, 0.5, 0.6), (0, 0, 1.1, 1), (0, 0, 1), (True, 0, 1, 1), (0, 0, float("nan"), 1)):
            with self.assertRaises(ValueError):
                em.validate_box(bad)
        with self.assertRaises(ValueError):
            em.crop_pixels((10, 10), (0.5, 0.5, 0.55, 0.9))  # < 2 px wide
        for scale in (0, -1, float("inf"), True):
            with self.assertRaises(ValueError):
                em.scaled_box((0.4, 0.4, 0.6, 0.6), scale)

    def test_variant_names(self):
        self.assertEqual([em.context_variant(s) for s in (1.0, 1.5, 2, 3.0, 1.25)],
                         ["context_1.0", "context_1.5", "context_2.0", "context_3.0", "context_1.25"])

    def test_crop_png_matches_app_semantics(self):
        image = Image.new("RGB", (100, 50), "white")
        image.putpixel((10, 5), (255, 0, 0))
        crop = Image.open(io.BytesIO(compare.crop_png(image, (0.1, 0.1, 0.505, 0.5))))
        self.assertEqual(crop.size, (41, 20))   # floor(10)..ceil(50.5)=51, floor(5)..ceil(25)
        self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(crop.info, {})
        whole = Image.open(io.BytesIO(compare.crop_png(image, None)))
        self.assertEqual(whole.size, (100, 50))
        limited = Image.open(io.BytesIO(compare.crop_png(Image.new("RGB", (400, 100)), None, max_side=200)))
        self.assertEqual(limited.size, (200, 50))


class SelectionTest(unittest.TestCase):
    def test_validation_rule(self):
        entries = [{"model_version": "1", "dataset_version": "v1", "accuracy": 1.0, "nll": 0.1},
                   {"model_version": "2", "dataset_version": "v2", "accuracy": 0.9, "nll": 0.5},
                   {"model_version": "3", "dataset_version": "v2", "accuracy": 0.9, "nll": 0.4}]
        selected, rationale = em.select_candidate(entries)
        self.assertEqual(selected, "3")
        self.assertEqual(rationale["excluded"], ["1"])
        entries[2]["nll"] = 0.5
        self.assertEqual(em.select_candidate(entries)[0], "2")      # exact tie -> lowest version
        entries[2]["accuracy"] = 0.95
        self.assertEqual(em.select_candidate(entries)[0], "3")      # accuracy first
        self.assertIsNone(em.select_candidate(entries[:1])[0])

    def test_checkpoint_fallback(self):
        acc = [(1, 0.5), (2, 0.8), (3, 0.8), (4, 0.7)]
        loss = [(1, 1.0), (2, 0.6), (3, 0.5), (4, 0.4)]
        self.assertEqual(em.selected_checkpoint(acc, loss), (0.8, 0.5))
        self.assertEqual(em.selected_checkpoint([(1, 0.8), (2, 0.8)], [(1, 0.5), (2, 0.5)]), (0.8, 0.5))
        self.assertIsNone(em.selected_checkpoint([], []))

    def test_aliases(self):
        alias = types.SimpleNamespace
        self.assertEqual(em.version_aliases("1", ["demo"], {"demo": "1", "champion": "2", "old": 1}),
                         ["demo", "old"])
        self.assertEqual(em.version_aliases("2", [], [alias(alias="champion", version="2"),
                                                      alias(alias="demo", version="1")]), ["champion"])
        self.assertEqual(em.version_aliases("3"), [])

    def test_model_uri(self):
        self.assertEqual(em.parse_model_uri("models:/c.s.m/12"), ("c.s.m", "12"))
        for bad in ("models:/c.s.m@demo", "models:/m/1", "models:/c.s.m/0", "runs:/x/y", None):
            with self.assertRaises(ValueError):
                em.parse_model_uri(bad)


class FakeMlflowClient:
    def __init__(self):
        metric = types.SimpleNamespace
        self.versions = {
            "1": types.SimpleNamespace(tags={}, run_id="r1", aliases=["demo"]),
            "2": types.SimpleNamespace(tags={"recipe": "improved", "dataset_version": "v2", "val_accuracy": "0.92",
                                             "val_loss": "0.31", "training_run_id": "r2"}, run_id="r2", aliases=[]),
        }
        self.history = {"val_accuracy": [metric(step=1, value=0.9), metric(step=2, value=1.0),
                                         metric(step=3, value=1.0)],
                        "val_loss": [metric(step=1, value=0.5), metric(step=2, value=0.4),
                                     metric(step=3, value=0.29)]}

    def get_model_version(self, name, version):
        return self.versions[version]

    def get_run(self, run_id):
        return types.SimpleNamespace(data=types.SimpleNamespace(metrics={"val_accuracy": 1.0, "val_loss": 0.29}))

    def get_metric_history(self, run_id, key):
        return self.history[key]

    def get_registered_model(self, name):
        return types.SimpleNamespace(aliases={"demo": "1", "champion": "2"})


class RegistryTest(unittest.TestCase):
    def test_v1_fallback_and_tagged_version(self):
        client = FakeMlflowClient()
        v1 = compare.model_version_info(client, "models:/c.s.m/1", "1")
        self.assertEqual((v1["recipe"], v1["dataset_version"], v1["training_run_id"]), ("v1", "v1", "r1"))
        self.assertEqual((v1["val_accuracy"], v1["val_loss"]), (1.0, 0.29))
        self.assertEqual((v1["aliases"], v1["serving"]), (["demo"], True))
        self.assertTrue(v1["metadata_fallbacks"])
        v2 = compare.model_version_info(client, "models:/c.s.m/2", "1")
        self.assertEqual((v2["recipe"], v2["dataset_version"], v2["val_accuracy"], v2["val_loss"]),
                         ("improved", "v2", 0.92, 0.31))
        self.assertEqual((v2["aliases"], v2["serving"], v2["metadata_fallbacks"]), (["champion"], False, []))


class Fixture:
    """Tiny v2 dataset + full-plane set + one app submission in a temporary directory."""

    def __init__(self, base):
        self.base = Path(base)
        self.dataset, self.full = self.base / "dataset_v2", self.base / "full_plane"
        self.classes = ["alaska", "delta"]
        records = []

        def add(image_id, airline, split, color, original=None):
            data = png(color)
            path = f"{split}/{airline}/{image_id}.png"
            (self.dataset / path).parent.mkdir(parents=True, exist_ok=True)
            (self.dataset / path).write_bytes(data)
            record = {"id": image_id, "airline": airline, "label": self.classes.index(airline), "split": split,
                      "path": path, "sha256": sha(data), "bbox": [0.25, 0.2, 0.75, 0.8],
                      "original_path": None, "original_sha256": None}
            if original is not None:
                opath = f"originals/{split}/{airline}/{image_id}.png"
                (self.dataset / opath).parent.mkdir(parents=True, exist_ok=True)
                original_png = compare.png_bytes(original)
                (self.dataset / opath).write_bytes(original_png)
                crop = compare.crop_png(original, record["bbox"])
                (self.dataset / path).write_bytes(crop)
                record.update(original_path=opath, original_sha256=sha(original_png), sha256=sha(crop))
            records.append(record)

        add("alaska_tail_1", "alaska", "train", "red")
        add("alaska_tail_v", "alaska", "val", "red")
        add("delta_tail_v", "delta", "val", "blue")
        add("alaska_tail_5", "alaska", "test", "red")
        add("delta_tail_7", "delta", "test", "blue")
        add("delta_full_new", "delta", "test", None, original=Image.new("RGB", (80, 40), "blue"))
        manifest = {"version": 2, "dataset_version": "v2", "classes": self.classes, "records": records}
        (self.dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        full_records = []
        for image_id, airline, data in (("alaska_full_01", "alaska", png("red", (60, 40))),
                                        ("delta_full_01", "delta", self.rotated_jpeg())):
            suffix = "jpg" if data[:2] == b"\xff\xd8" else "png"
            path = f"images/{airline}/{image_id}.{suffix}"
            (self.full / path).parent.mkdir(parents=True, exist_ok=True)
            (self.full / path).write_bytes(data)
            full_records.append({"id": image_id, "airline": airline, "path": path, "sha256": sha(data),
                                 "bbox": [0.1, 0.1, 0.5, 0.5]})
        (self.full / "manifest.json").write_text(json.dumps({"version": 1, "classes": self.classes,
                                                             "records": full_records}), encoding="utf-8")
        app_original = Image.new("RGB", (50, 50), "red")
        (self.base / "app").mkdir()
        (self.base / "app" / "original.png").write_bytes(compare.png_bytes(app_original))
        (self.base / "app" / "tail_crop.png").write_bytes(compare.crop_png(app_original, (0.2, 0.2, 0.6, 0.6)))
        self.params = compare.load_params({
            "models": ["models:/c.s.m/1", "models:/c.s.m/2"], "dataset_root": str(self.dataset),
            "full_plane_root": str(self.full), "output_root": str(self.base / "out"),
            "v1_test_ids": ["alaska_tail_5", "delta_tail_7"], "context_scales": [1.0, 2.0],
            "serving_endpoint_version": "1",
            "app_reviewed": [{"id": "sub-1", "label": "alaska", "original_path": str(self.base / "app" / "original.png"),
                              "crop_path": str(self.base / "app" / "tail_crop.png"), "crop_box": [0.2, 0.2, 0.6, 0.6]}]})

    @staticmethod
    def rotated_jpeg():
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 270 -> upright is 20 x 40
        buffer = io.BytesIO()
        Image.new("RGB", (40, 20), "blue").save(buffer, format="JPEG", exif=exif.tobytes())
        return buffer.getvalue()


def fake_predictor(calls):
    """Predicts by mean colour: reddish -> alaska (0.8), else delta (0.7)."""
    def predict(values):
        calls.append(len(values))
        out = []
        for value in values:
            image = Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
            r, _, b = image.resize((1, 1)).getpixel((0, 0))
            probs = {"alaska": 0.8, "delta": 0.2} if r > b else {"alaska": 0.3, "delta": 0.7}
            winner = max(probs, key=probs.get)
            out.append({"airline": winner, "confidence": probs[winner], "probabilities_json": json.dumps(probs)})
        return out
    return predict


class EvalSetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_sets_variants_and_counts(self):
        sets, context = compare.build_eval_sets(self.fixture.params)
        counts = {key: len(items) for key, items in sets.items()}
        self.assertEqual(counts, {
            ("v2_val", "tail_crop"): 2, ("v1_test", "tail_crop"): 2, ("new_holdout", "tail_crop"): 1,
            ("v2_test", "tail_crop"): 3, ("full_plane_stress", "whole_image"): 2,
            ("full_plane_stress", "tail_crop"): 2, ("context_sensitivity", "context_1.0"): 4,
            ("context_sensitivity", "context_2.0"): 4, ("context_sensitivity", "context_full"): 4,
            ("app_reviewed", "tail_crop"): 1})
        self.assertEqual([i["item_id"] for i in sets[("context_sensitivity", "context_full")]],
                         ["alaska_full_01", "delta_full_01", "delta_full_new", "sub-1"])
        self.assertEqual(context["crop_parity"], {"checked": 2, "mismatched_ids": []})
        self.assertEqual(context["dataset_version"], "v2")
        # Stored dataset crop is sent byte-for-byte.
        stored = (self.fixture.dataset / "val/alaska/alaska_tail_v.png").read_bytes()
        self.assertEqual(base64.b64decode(sets[("v2_val", "tail_crop")][0]["image_base64"]), stored)
        # EXIF orientation 6 applied: 40x20 JPEG is scored upright as 20x40.
        whole = sets[("full_plane_stress", "whole_image")][1]
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(whole["image_base64"]))).size, (20, 40))
        self.assertTrue(whole["source"].endswith("images/delta/delta_full_01.jpg"))
        crop = sets[("full_plane_stress", "tail_crop")][0]
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(crop["image_base64"]))).size, (24, 16))

    def test_hash_and_split_checks(self):
        (self.fixture.dataset / "test/delta/delta_tail_7.png").write_bytes(png("green"))
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            compare.build_eval_sets(self.fixture.params)
        params = {**self.fixture.params, "v1_test_ids": ["alaska_tail_v"]}
        with self.assertRaisesRegex(ValueError, "v1 test id"):
            compare.build_eval_sets(params)

    def test_param_validation(self):
        base = dict(self.fixture.params)
        for change in ({"models": ["models:/c.s.m/1", "models:/c.s.other/2"]},
                       {"models": ["models:/c.s.m/1", "models:/c.s.m/1"]},
                       {"context_scales": [1.0, 1]}, {"serving_endpoint_version": "latest"},
                       {"app_reviewed": [{"id": "x"}]}):
            with self.assertRaises(ValueError):
                compare.load_params({**base, **change})
        self.assertEqual(compare.load_params({**base, "serving_endpoint_version": 1})["serving_endpoint_version"], "1")

    def test_scoring_rows_match_tables(self):
        sets, context = compare.build_eval_sets(self.fixture.params)
        classes = context["classes"]
        calls = []
        evaluation = compare.evaluate_model(fake_predictor(calls), sets, classes, batch_size=4)
        unique = {i["png_sha256"] for items in sets.values() for i in items}
        self.assertEqual(sum(calls), len(unique))       # identical images scored once
        self.assertTrue(all(c <= 4 for c in calls))
        self.assertEqual(evaluation[("v2_val", "tail_crop")]["metrics"]["accuracy"], 1.0)
        models = []
        for version, dataset_version in (("1", "v1"), ("2", "v2")):
            models.append({"model_name": "c.s.m", "model_version": version, "model_uri": f"models:/c.s.m/{version}",
                           "training_run_id": f"r{version}", "recipe": "v1", "dataset_version": dataset_version,
                           "val_accuracy": 1.0, "val_loss": None, "aliases": ["demo"] if version == "1" else [],
                           "serving": version == "1", "selected": False, "child_run_id": f"c{version}",
                           "evaluation": evaluation})
        selected, rationale = compare.apply_selection(models)
        self.assertEqual(selected, "2")
        created = em.utc_timestamp()
        self.assertRegex(created, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$")
        runs, metrics, predictions = compare.build_rows(EVAL_ID, created, models, sets, classes)
        tables = sql_tables()
        for rows, columns, table in ((runs, em.RUN_COLUMNS, "model_eval_runs"),
                                     (metrics, em.METRIC_COLUMNS, "model_eval_metrics"),
                                     (predictions, em.PREDICTION_COLUMNS, "model_eval_predictions")):
            sql_columns = [name for name, _ in tables[table]]
            self.assertEqual(list(columns), sql_columns)
            parsed = [json.loads(line) for line in em.jsonl(rows, columns).splitlines()]
            self.assertTrue(parsed and all(list(r) == sql_columns for r in parsed))
        self.assertEqual(json.loads(runs[0]["aliases"]), ["demo"])
        self.assertEqual([(r["serving"], r["selected"]) for r in runs], [(True, False), (False, True)])
        self.assertIsNone(runs[0]["val_loss"])
        n_variants = len(sets)
        self.assertEqual(len(metrics), 2 * n_variants * (7 + 2 + 4 * 2))
        self.assertEqual(len(predictions), 2 * sum(len(items) for items in sets.values()))
        row = next(r for r in predictions if r["eval_set"] == "app_reviewed")
        self.assertEqual((row["item_id"], row["actual"], row["predicted"], row["correct"]),
                         ("sub-1", "alaska", "alaska", True))
        self.assertEqual(row["true_prob"], 0.8)
        self.assertEqual(json.loads(row["probabilities_json"]), {"alaska": 0.8, "delta": 0.2})
        flat = compare.flat_metrics(evaluation)
        self.assertIn("context_sensitivity/context_2.0/support_delta", flat)
        document = compare.result_document(EVAL_ID, created, self.fixture.params, context, sets, models,
                                           selected, rationale)
        json.dumps(document, allow_nan=False)
        self.assertIn("confusion_matrix", document["models"][0]["results"]["v2_val"]["tail_crop"])

    def test_rejects_inconsistent_predictions(self):
        bad = {"airline": "alaska", "confidence": 0.9, "probabilities_json": json.dumps({"alaska": 0.8, "delta": 0.2})}
        with self.assertRaises(ValueError):
            compare.parse_prediction(bad, ["alaska", "delta"])
        with self.assertRaises(ValueError):
            compare.parse_prediction({**bad, "confidence": 0.8, "probabilities_json": '{"alaska": 0.8}'},
                                     ["alaska", "delta"])
        with self.assertRaises(ValueError):
            em.jsonl([{"eval_run_id": "x"}], em.METRIC_COLUMNS)


class ImportTest(unittest.TestCase):
    def test_compare_imports_without_heavy_dependencies(self):
        for name in ("mlflow", "torch", "pandas"):
            self.assertNotIn(name, sys.modules)

    def test_workload_parameters_validate(self):
        import yaml
        workload = yaml.safe_load((ROOT / "evaluation" / "workload_compare.yaml").read_text(encoding="utf-8"))
        self.assertEqual(workload["experiment_name"], "airplane-tail-evaluation")
        self.assertEqual(workload["environment"]["version"], "databricks_ai_v6")
        self.assertEqual(set(workload["code_source"]["snapshot"]["include_paths"]), {"compare.py", "eval_metrics.py"})
        params = compare.load_params(workload["parameters"])
        self.assertEqual(len(params["v1_test_ids"]), 10)
        self.assertEqual(params["serving_endpoint_version"], "1")


# ---------------------------------------------------------------- publish
class FakeStatements:
    def __init__(self, counts, fail_on=None, pending_polls=1):
        self.counts, self.fail_on, self.pending_polls = counts, fail_on, pending_polls
        self.executed, self.responses = [], {}

    def _response(self, statement_id, state, rows=None, error=None):
        status = types.SimpleNamespace(state=types.SimpleNamespace(value=state),
                                       error=types.SimpleNamespace(message=error) if error else None)
        return types.SimpleNamespace(statement_id=statement_id, status=status,
                                     result=types.SimpleNamespace(data_array=rows) if rows is not None else None)

    def execute_statement(self, statement, warehouse_id, parameters, **options):
        self.executed.append((statement, warehouse_id, [(p.name, p.value) for p in parameters]))
        statement_id = f"s{len(self.executed)}"
        table = next(t for t in publish.TABLES if f".{t} " in statement or f"{t[11:]}.jsonl" in statement)
        rows = None
        if statement.startswith("SELECT count(*) AS n, count_if"):
            rows = [[str(self.counts[table]), "0"]]
        elif statement.startswith("SELECT count(*) AS n FROM"):
            rows = [[str(self.counts[table])]]
        failed = self.fail_on and self.fail_on in statement
        self.responses[statement_id] = (self._response(statement_id, "FAILED", error="boom") if failed
                                        else self._response(statement_id, "SUCCEEDED", rows))
        return self._response(statement_id, "PENDING") if self.pending_polls else self.responses[statement_id]

    def get_statement(self, statement_id):
        return self.responses[statement_id]

    def cancel_execution(self, statement_id):
        pass


class FakeWorkspace:
    def __init__(self, files, **options):
        self.statement_execution = FakeStatements({t: len(files[f].splitlines())
                                                   for t, (f, _) in publish.TABLES.items()}, **options)
        self.downloaded = []
        self._files = files

        def download(path):
            self.downloaded.append(path)
            return types.SimpleNamespace(contents=io.BytesIO(self._files[path.rsplit("/", 1)[1]].encode()))
        self.files = types.SimpleNamespace(download=download)


def jsonl_files(eval_run_id=EVAL_ID):
    run = {c: None for c in em.RUN_COLUMNS}
    metric = {c: None for c in em.METRIC_COLUMNS}
    prediction = {c: None for c in em.PREDICTION_COLUMNS}
    for row in (run, metric, prediction):
        row["eval_run_id"] = eval_run_id
    return {"runs.jsonl": json.dumps(run) + "\n",
            "metrics.jsonl": (json.dumps(metric) + "\n") * 3,
            "predictions.jsonl": (json.dumps(prediction) + "\n") * 2}


class PublishTest(unittest.TestCase):
    root = publish.DEFAULT_VOLUME_ROOT

    def test_schema_matches_sql(self):
        tables = sql_tables()
        self.assertEqual(set(tables), set(publish.TABLES))
        for table, (_, columns) in publish.TABLES.items():
            self.assertEqual(columns, tables[table])

    def test_statement_order_and_binding(self):
        plan = publish.statements(EVAL_ID, self.root, "cat.sch")
        self.assertEqual([k for k, _, _ in plan][:3], ["source_count"] * 3)
        steps = [(k, t) for k, t, _ in plan if k in ("delete", "insert")]
        self.assertEqual(steps, list(publish.STEPS))
        for table in publish.TABLES:
            kinds = [k for k, t in steps if t == table]
            self.assertEqual(kinds, ["delete", "insert"])
        self.assertEqual(steps[-1], ("insert", "model_eval_runs"))
        for kind, table, sql in plan:
            self.assertIn(":eval_run_id", sql)
            self.assertNotIn(";", sql)
        insert = next(sql for k, t, sql in plan if k == "insert" and t == "model_eval_runs")
        self.assertTrue(insert.startswith("INSERT INTO cat.sch.model_eval_runs (eval_run_id, created_at, "))
        self.assertIn("CAST(created_at AS TIMESTAMP) AS created_at", insert)
        self.assertIn(f"read_files('{self.root}/{EVAL_ID}/runs.jsonl', format => 'json', "
                      "schema => 'eval_run_id STRING, created_at STRING,", insert)
        self.assertIn("CAST(n AS INT) AS n", next(sql for k, t, sql in plan if k == "insert" and t == "model_eval_metrics"))

    def test_rejects_unsafe_inputs(self):
        for bad_id in ("x' OR '1'='1", EVAL_ID.upper(), EVAL_ID[:-1], EVAL_ID + "0", ""):
            with self.assertRaises(publish.PublishError):
                publish.statements(bad_id, self.root, "cat.sch")
        for bad_root in ("/Volumes/a/b", "/Volumes/a/b/c/../d", "/Volumes/a/b/c/it's", "/tmp/a/b/c",
                         "/Volumes/a/b/c/"):
            with self.assertRaises(publish.PublishError):
                publish.statements(EVAL_ID, bad_root, "cat.sch")
        for bad_schema in ("cat", "cat.sch.tbl", "cat.sch; DROP TABLE x", "`cat`.sch"):
            with self.assertRaises(publish.PublishError):
                publish.statements(EVAL_ID, self.root, bad_schema)
        with self.assertRaises(publish.PublishError):
            publish.validate(EVAL_ID, self.root, "cat.sch", "abc-def")

    def test_publish_with_fake_client(self):
        client = FakeWorkspace(jsonl_files())
        logged = []
        summary = publish.publish(client, EVAL_ID, "wh123", log=logged.append, sleep=lambda s: None)
        self.assertEqual(summary["tables"], {"model_eval_runs": 1, "model_eval_metrics": 3,
                                             "model_eval_predictions": 2})
        executed = client.statement_execution.executed
        self.assertEqual(len(executed), 12)
        self.assertTrue(all(w == "wh123" and p == [("eval_run_id", EVAL_ID)] for _, w, p in executed))
        first_delete = next(i for i, (sql, _, _) in enumerate(executed) if sql.startswith("DELETE"))
        self.assertTrue(executed[first_delete][0].startswith("DELETE FROM dbx_ai_app.airplane_tail_id.model_eval_runs"))
        self.assertTrue(executed[8][0].startswith("INSERT INTO dbx_ai_app.airplane_tail_id.model_eval_runs"))
        self.assertEqual(client.downloaded, [f"{self.root}/{EVAL_ID}/{f}"
                                             for f in ("runs.jsonl", "metrics.jsonl", "predictions.jsonl")])

    def test_failures_stop_publish(self):
        client = FakeWorkspace(jsonl_files(), fail_on="INSERT INTO dbx_ai_app.airplane_tail_id.model_eval_metrics")
        with self.assertRaisesRegex(publish.PublishError, "boom"):
            publish.publish(client, EVAL_ID, "wh123", log=lambda m: None, sleep=lambda s: None)
        self.assertFalse(any(sql.startswith("INSERT INTO dbx_ai_app.airplane_tail_id.model_eval_runs")
                             for sql, _, _ in client.statement_execution.executed))
        foreign = FakeWorkspace({**jsonl_files(), "runs.jsonl": jsonl_files("f" * 32)["runs.jsonl"]})
        with self.assertRaisesRegex(publish.PublishError, "eval_run_id differs"):
            publish.publish(foreign, EVAL_ID, "wh123", log=lambda m: None)
        self.assertEqual(foreign.statement_execution.executed, [])
        mismatch = FakeWorkspace(jsonl_files())
        mismatch.statement_execution.counts["model_eval_metrics"] = 2
        with self.assertRaisesRegex(publish.PublishError, "read_files sees 2 rows"):
            publish.publish(mismatch, EVAL_ID, "wh123", log=lambda m: None, sleep=lambda s: None)
        self.assertFalse(any(sql.startswith("DELETE") for sql, _, _ in mismatch.statement_execution.executed))


if __name__ == "__main__":
    unittest.main()
