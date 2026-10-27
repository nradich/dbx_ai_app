"""Compare registered model versions on shared evaluation sets (AIR job, MLflow-logged).

Runs inside the AIR-created MLflow run (MLFLOW_RUN_ID, experiment airplane-tail-evaluation)
with parameters from HYPERPARAMETERS_PATH. One nested child run per model version holds
metrics keyed `<eval_set>/<variant>/<metric>`. Writes result.json, runs.jsonl, metrics.jsonl
and predictions.jsonl (rows of app_setup/create_eval_tables.sql) to
`<output_root>/<parent_run_id>/`; evaluation/publish.py loads them into Unity Catalog.

Images are prepared like the app (app/serving_client.py): EXIF-orientation-upright RGB,
normalized box rounded outward (floor/ceil), longest side limited to the app's
CROP_MAX_SIDE, PNG without metadata. Stored dataset/app tail crops are sent byte-for-byte.
Every dataset and full-plane file is checked against its manifest sha256 before use.

mlflow, pandas and yaml are imported inside main() so the image/set/row logic can be
unit-tested without them.
"""
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics as em  # noqa: E402

EXPERIMENT_NAME = "airplane-tail-evaluation"
CROP_MAX_SIDE = 1536          # app/pipeline.py CROP_MAX_SIDE
DEFAULT_SCALES = (1.0, 1.5, 2.0, 3.0)
DEFAULT_BATCH_SIZE = 32
METRIC_BATCH = 500
ITEM_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
APP_REVIEWED_KEYS = {"id", "label", "original_path", "crop_path", "crop_box"}

# ---------------------------------------------------------------- app-parity image preparation
EXIF_ORIENTATION_TAG = 0x0112
_TRANSPOSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
              4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
              6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
              8: Image.Transpose.ROTATE_90}
_SIXTEEN_BIT_GRAY = frozenset({"I", "I;16", "I;16L", "I;16B", "I;16N"})


def exif_orientation(image):
    raw = image.info.get("exif")
    if not raw:
        return 1
    exif = Image.Exif()
    try:
        exif.load(raw)
        value = exif.get(EXIF_ORIENTATION_TAG, 1)
    except Exception:
        return 1
    return value if value in _TRANSPOSE else 1


def oriented_rgb(image_bytes):
    """Same as app/serving_client.oriented_rgb: EXIF-only orientation, metadata-free RGB."""
    with Image.open(io.BytesIO(image_bytes)) as original:
        original.load()
        method = _TRANSPOSE.get(exif_orientation(original))
        image = original if method is None else original.transpose(method)
        if image.mode in _SIXTEEN_BIT_GRAY:
            image = image.convert("I").point(lambda value: value / 257).convert("L")
        rgb = image.convert("RGB")
    rgb.info = {}
    return rgb


def png_bytes(image):
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    return buffer.getvalue()


def crop_png(rgb, box, max_side=CROP_MAX_SIDE):
    """app/serving_client.tail_crop + png_bytes; box None means the whole image."""
    if box is not None:
        rgb = rgb.crop(em.crop_pixels(rgb.size, box))
    if max_side is not None and max(rgb.size) > max_side:
        scale = max_side / max(rgb.size)
        rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                         Image.Resampling.LANCZOS)
    return png_bytes(rgb)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected):
    if not isinstance(expected, str) or not SHA256.match(expected):
        raise ValueError(f"missing or malformed sha256 for {path}")
    data = Path(path).read_bytes()
    if sha256(data) != expected:
        raise ValueError(f"sha256 mismatch: {path}")
    return data


def _item(item_id, actual, source, png):
    return {"item_id": item_id, "actual": actual, "source": str(source), "png_sha256": sha256(png),
            "image_base64": base64.b64encode(png).decode("ascii")}


def _join(root, relative):
    """Join a manifest-relative POSIX path under root, refusing escapes."""
    parts = Path(relative).parts if isinstance(relative, str) else ()
    if not parts or Path(relative).is_absolute() or ".." in parts or "\\" in relative:
        raise ValueError(f"manifest path must be relative: {relative!r}")
    return Path(root).joinpath(*relative.split("/"))


def _source_label(root, relative):
    return f"{str(root).rstrip('/')}/{relative}"


# ---------------------------------------------------------------- parameters
def load_params(raw):
    if not isinstance(raw, dict):
        raise ValueError("parameters must be a mapping")
    params = dict(raw)
    models = params.get("models")
    if not isinstance(models, list) or not models or len(set(models)) != len(models):
        raise ValueError("models must be a nonempty list of distinct model URIs")
    parsed = [em.parse_model_uri(uri) for uri in models]
    if len({name for name, _ in parsed}) != 1:
        raise ValueError("all models must be versions of one registered model")
    for key in ("dataset_root", "full_plane_root", "output_root"):
        if not isinstance(params.get(key), str) or not params[key]:
            raise ValueError(f"{key} is required")
    ids = params.get("v1_test_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) \
            or not all(isinstance(i, str) and ITEM_ID.match(i) for i in ids):
        raise ValueError("v1_test_ids must be a nonempty list of distinct IDs")
    scales = params.get("context_scales", list(DEFAULT_SCALES))
    if not isinstance(scales, list) or not scales or len(set(map(float, scales))) != len(scales):
        raise ValueError("context_scales must be a nonempty list of distinct numbers")
    for scale in scales:
        em.scaled_box((0.4, 0.4, 0.6, 0.6), scale)  # validates the scale
    params["context_scales"] = [float(s) for s in scales]
    reviewed = params.get("app_reviewed") or []
    if not isinstance(reviewed, list):
        raise ValueError("app_reviewed must be a list")
    for entry in reviewed:
        if not isinstance(entry, dict) or set(entry) != APP_REVIEWED_KEYS:
            raise ValueError(f"app_reviewed entries need exactly {sorted(APP_REVIEWED_KEYS)}")
        if not isinstance(entry["id"], str) or not ITEM_ID.match(entry["id"]):
            raise ValueError(f"invalid app_reviewed id: {entry['id']!r}")
        for key in ("label", "original_path", "crop_path"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError(f"app_reviewed {entry['id']}: {key} is required")
        em.validate_box(entry["crop_box"])
    if len({e["id"] for e in reviewed}) != len(reviewed):
        raise ValueError("app_reviewed ids must be distinct")
    params["app_reviewed"] = reviewed
    serving = params.get("serving_endpoint_version")
    if isinstance(serving, bool) or not isinstance(serving, (str, int)) or not str(serving).isdigit():
        raise ValueError("serving_endpoint_version must be a version number string")
    params["serving_endpoint_version"] = str(serving)
    max_side = params.get("crop_max_side", CROP_MAX_SIDE)
    if max_side is not None and (isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 2):
        raise ValueError("crop_max_side must be an integer >= 2 or null")
    params["crop_max_side"] = max_side
    batch = params.get("batch_size", DEFAULT_BATCH_SIZE)
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch_size must be a positive integer")
    params["batch_size"] = batch
    return params


# ---------------------------------------------------------------- evaluation sets
def build_eval_sets(params):
    """Read, verify and prepare every input once.

    Returns (sets, context): sets maps (eval_set, variant) -> [item] in a fixed order, each
    item {item_id, actual, source, png_sha256, image_base64}; empty sets are omitted.
    """
    scales, max_side = params["context_scales"], params["crop_max_side"]
    dataset_root, full_root = Path(params["dataset_root"]), Path(params["full_plane_root"])
    manifest_bytes = (dataset_root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    full_bytes = (full_root / "manifest.json").read_bytes()
    full_manifest = json.loads(full_bytes)
    classes = list(manifest["classes"])
    if manifest.get("dataset_version") != "v2":
        raise ValueError("dataset_root must hold the v2 dataset manifest (dataset_version 'v2')")
    if list(full_manifest["classes"]) != classes:
        raise ValueError("full-plane and dataset manifests list different classes")
    records = manifest["records"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("dataset record IDs must be unique")
    for r in records:
        if r["airline"] not in classes or r["split"] not in ("train", "val", "test"):
            raise ValueError(f"invalid dataset record: {r['id']}")
    by_id = {r["id"]: r for r in records}
    v1_ids = list(params["v1_test_ids"])
    for image_id in v1_ids:
        if image_id not in by_id or by_id[image_id]["split"] != "test":
            raise ValueError(f"v1 test id is not a v2 test record: {image_id}")
    v1_set = set(v1_ids)
    sets = {}
    crop_cache = {}

    def dataset_crop(record):
        if record["id"] not in crop_cache:
            png = read_verified(_join(dataset_root, record["path"]), record["sha256"])
            crop_cache[record["id"]] = _item(record["id"], record["airline"],
                                             _source_label(dataset_root, record["path"]), png)
        return crop_cache[record["id"]]

    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]
    sets[("v2_val", "tail_crop")] = [dataset_crop(r) for r in val]
    sets[("v1_test", "tail_crop")] = [dataset_crop(by_id[i]) for i in v1_ids]
    sets[("new_holdout", "tail_crop")] = [dataset_crop(r) for r in test if r["id"] not in v1_set]
    sets[("v2_test", "tail_crop")] = [dataset_crop(r) for r in test]

    # Sources with a full image and a tail box: (item_id, actual, source label, rgb, box, stored crop sha)
    context_sources = []
    stress_whole, stress_crop = [], []
    for record in full_manifest["records"]:
        if record["airline"] not in classes:
            raise ValueError(f"invalid full-plane record: {record['id']}")
        data = read_verified(_join(full_root, record["path"]), record["sha256"])
        rgb, box = oriented_rgb(data), em.validate_box(record["bbox"])
        label = _source_label(full_root, record["path"])
        stress_whole.append(_item(record["id"], record["airline"], label, crop_png(rgb, None, max_side)))
        stress_crop.append(_item(record["id"], record["airline"], label, crop_png(rgb, box, max_side)))
        context_sources.append((record["id"], record["airline"], label, rgb, box, None))
    sets[("full_plane_stress", "whole_image")] = stress_whole
    sets[("full_plane_stress", "tail_crop")] = stress_crop
    for record in test:
        if record.get("original_path"):
            data = read_verified(_join(dataset_root, record["original_path"]), record.get("original_sha256"))
            context_sources.append((record["id"], record["airline"],
                                    _source_label(dataset_root, record["original_path"]),
                                    oriented_rgb(data), em.validate_box(record["bbox"]), record["sha256"]))
    reviewed_crops = []
    for entry in params["app_reviewed"]:
        if entry["label"] not in classes:
            raise ValueError(f"app_reviewed {entry['id']}: unknown label {entry['label']!r}")
        crop = Path(entry["crop_path"]).read_bytes()  # no stored hash to verify against
        reviewed_crops.append(_item(entry["id"], entry["label"], entry["crop_path"], crop))
        original = oriented_rgb(Path(entry["original_path"]).read_bytes())
        context_sources.append((entry["id"], entry["label"], entry["original_path"], original,
                                em.validate_box(entry["crop_box"]), sha256(crop)))
    parity = {"checked": 0, "mismatched_ids": []}
    for scale in scales:
        variant = em.context_variant(scale)
        sets[("context_sensitivity", variant)] = []
        for item_id, actual, label, rgb, box, stored_sha in context_sources:
            item = _item(item_id, actual, label, crop_png(rgb, em.scaled_box(box, scale), max_side))
            sets[("context_sensitivity", variant)].append(item)
            if scale == 1 and stored_sha is not None:
                parity["checked"] += 1
                if item["png_sha256"] != stored_sha:
                    parity["mismatched_ids"].append(item_id)
    sets[("context_sensitivity", "context_full")] = [
        _item(item_id, actual, label, crop_png(rgb, None, max_side))
        for item_id, actual, label, rgb, _, _ in context_sources]
    sets[("app_reviewed", "tail_crop")] = reviewed_crops
    sets = {key: items for key, items in sets.items() if items}
    for key, items in sets.items():
        if len({i["item_id"] for i in items}) != len(items):
            raise ValueError(f"duplicate item IDs in {key}")
    context = {
        "classes": classes,
        "dataset_version": manifest.get("dataset_version"),
        "dataset_manifest_sha256": sha256(manifest_bytes),
        "full_plane_manifest_sha256": sha256(full_bytes),
        "context_sources": len(context_sources),
        # context_1.0 regenerated from originals vs the stored crop, byte-for-byte (informational)
        "crop_parity": parity,
        "manifest_bytes": manifest_bytes, "full_manifest_bytes": full_bytes,
    }
    return sets, context


# ---------------------------------------------------------------- scoring
def parse_prediction(record, classes):
    probabilities = json.loads(record["probabilities_json"])
    if set(probabilities) != set(classes):
        raise ValueError("prediction classes differ from evaluation classes")
    values = {c: float(probabilities[c]) for c in classes}
    confidence = float(record["confidence"])
    if not all(math.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 for v in [confidence, *values.values()]):
        raise ValueError("prediction probabilities must be finite and within [0, 1]")
    predicted = record["airline"]
    if predicted not in classes or abs(values[predicted] - confidence) > 1e-6:
        raise ValueError("predicted airline/confidence inconsistent with probabilities")
    values = {c: min(max(v, 0.0), 1.0) for c, v in values.items()}
    return {"predicted": predicted, "confidence": min(max(confidence, 0.0), 1.0), "probabilities": values}


def evaluate_model(predict_fn, sets, classes, batch_size=DEFAULT_BATCH_SIZE):
    """predict_fn(list of base64 PNG) -> list of {airline, confidence, probabilities_json}.

    Identical images (same PNG sha256) are scored once per model. Returns
    {(eval_set, variant): {"n", "metrics", "confusion_matrix", "predictions"}}.
    """
    cache, pending, queued = {}, [], set()
    for items in sets.values():
        for item in items:
            if item["png_sha256"] not in queued:
                queued.add(item["png_sha256"])
                pending.append((item["png_sha256"], item["image_base64"]))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        outputs = list(predict_fn([value for _, value in batch]))
        if len(outputs) != len(batch):
            raise ValueError("model returned a different number of predictions")
        for (digest, _), output in zip(batch, outputs):
            cache[digest] = parse_prediction(output, classes)
    results = {}
    for key, items in sets.items():
        predictions = [cache[item["png_sha256"]] for item in items]
        metrics, confusion = em.compute_metrics(
            [{"actual": i["actual"], **p} for i, p in zip(items, predictions)], classes)
        results[key] = {"n": len(items), "metrics": metrics, "confusion_matrix": confusion,
                        "predictions": predictions}
    return results


def flat_metrics(evaluation):
    return {f"{s}/{v}/{m}": value for (s, v), r in evaluation.items() for m, value in r["metrics"].items()}


def build_rows(eval_run_id, created_at, models, sets, classes):
    """models: [{...run_row fields..., "evaluation": evaluate_model(...)}] -> (runs, metrics, predictions)."""
    runs, metrics, predictions = [], [], []
    for model in models:
        version = model["model_version"]
        runs.append(em.run_row(eval_run_id, created_at, model))
        for (eval_set, variant), result in model["evaluation"].items():
            metrics += em.metric_rows(eval_run_id, version, eval_set, variant, result["metrics"], result["n"])
            for item, prediction in zip(sets[(eval_set, variant)], result["predictions"]):
                predictions.append(em.prediction_row(eval_run_id, version, eval_set, variant, item,
                                                     prediction, classes))
    return runs, metrics, predictions


# ---------------------------------------------------------------- registry metadata
def _float_or_none(value):
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def model_version_info(client, uri, serving_version):
    """Registry/training metadata for one version; v1 (no tags) falls back to its training run."""
    name, version = em.parse_model_uri(uri)
    mv = client.get_model_version(name, version)
    tags = dict(getattr(mv, "tags", None) or {})
    fallbacks = []
    recipe, dataset_version = tags.get("recipe"), tags.get("dataset_version")
    if not recipe:
        recipe = "v1"
        fallbacks.append("recipe=v1 (no version tag)")
    if not dataset_version:
        dataset_version = "v1"
        fallbacks.append("dataset_version=v1 (no version tag)")
    training_run_id = tags.get("training_run_id") or getattr(mv, "run_id", None)
    val_accuracy, val_loss = _float_or_none(tags.get("val_accuracy")), _float_or_none(tags.get("val_loss"))
    if (val_accuracy is None or val_loss is None) and training_run_id:
        run_metrics = dict(client.get_run(training_run_id).data.metrics or {})
        if "best_val_accuracy" in run_metrics and "best_val_loss" in run_metrics:
            val_accuracy, val_loss = run_metrics["best_val_accuracy"], run_metrics["best_val_loss"]
            fallbacks.append("val metrics from training run best_val_accuracy/best_val_loss")
        else:
            history = em.selected_checkpoint(
                [(m.step, m.value) for m in client.get_metric_history(training_run_id, "val_accuracy")],
                [(m.step, m.value) for m in client.get_metric_history(training_run_id, "val_loss")])
            if history:
                val_accuracy, val_loss = history
                fallbacks.append("val metrics from training run history at the selected epoch")
    registered = client.get_registered_model(name)
    aliases = em.version_aliases(version, getattr(mv, "aliases", None), getattr(registered, "aliases", None))
    return {"model_name": name, "model_version": version, "model_uri": uri,
            "training_run_id": training_run_id, "recipe": recipe, "dataset_version": dataset_version,
            "val_accuracy": _float_or_none(val_accuracy), "val_loss": _float_or_none(val_loss),
            "aliases": aliases, "serving": version == serving_version, "selected": False,
            "metadata_fallbacks": fallbacks}


def apply_selection(models):
    entries = [{"model_version": m["model_version"], "dataset_version": m["dataset_version"],
                "accuracy": m["evaluation"][("v2_val", "tail_crop")]["metrics"]["accuracy"],
                "nll": m["evaluation"][("v2_val", "tail_crop")]["metrics"]["nll"]} for m in models]
    selected, rationale = em.select_candidate(entries)
    for model in models:
        model["selected"] = model["model_version"] == selected
    return selected, rationale


def result_document(eval_run_id, created_at, params, context, sets, models, selection, rationale):
    def model_detail(m):
        return {**{k: v for k, v in m.items() if k != "evaluation"},
                "results": {s: {v: {k: r[k] for k in ("n", "metrics", "confusion_matrix")}
                                for (s2, v), r in m["evaluation"].items() if s2 == s}
                            for s in em.EVAL_SETS if any(k[0] == s for k in m["evaluation"])}}
    return {
        "eval_run_id": eval_run_id, "created_at": created_at, "experiment": EXPERIMENT_NAME,
        "classes": context["classes"], "confusion_matrix_axes": "rows=actual, columns=predicted, classes order",
        "params": {k: v for k, v in params.items()},
        "dataset_version": context["dataset_version"],
        "dataset_manifest_sha256": context["dataset_manifest_sha256"],
        "full_plane_manifest_sha256": context["full_plane_manifest_sha256"],
        "crop_parity": context["crop_parity"],
        "eval_sets": [{"eval_set": s, "variant": v, "n": len(items), "item_ids": [i["item_id"] for i in items]}
                      for (s, v), items in sets.items()],
        "metric_definitions": em.__doc__.strip(),
        "image_preparation": ("EXIF-orientation upright RGB; normalized box rounded outward (floor/ceil); "
                              f"longest side limited to {params['crop_max_side']} (app CROP_MAX_SIDE); "
                              "metadata-free PNG; stored dataset/app tail crops sent byte-for-byte"),
        "models": [model_detail(m) for m in models],
        "selection": {**rationale, "selected": selection},
        "predictions_file": "predictions.jsonl",
    }


# ---------------------------------------------------------------- AIR entry point
def main():
    import mlflow
    import pandas as pd
    import yaml

    params = load_params(yaml.safe_load(Path(os.environ["HYPERPARAMETERS_PATH"]).read_text()))
    run_id = os.environ.pop("MLFLOW_RUN_ID")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    sets, context = build_eval_sets(params)  # all inputs verified before any model is loaded
    classes = context["classes"]
    output = Path(params["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    client = mlflow.MlflowClient()
    with mlflow.start_run(run_id=run_id) as parent, tempfile.TemporaryDirectory() as temp:
        local = Path(temp)
        counts = {f"n/{s}/{v}": len(items) for (s, v), items in sets.items()}
        mlflow.log_params({
            "models": json.dumps(params["models"]), "dataset_root": params["dataset_root"],
            "full_plane_root": params["full_plane_root"], "output_root": params["output_root"],
            "v1_test_ids": ",".join(params["v1_test_ids"]),
            "context_scales": json.dumps(params["context_scales"]),
            "app_reviewed_count": len(params["app_reviewed"]),
            "serving_endpoint_version": params["serving_endpoint_version"],
            "crop_max_side": params["crop_max_side"], "batch_size": params["batch_size"],
            "eval_sets": json.dumps(sorted({s for s, _ in sets})), **counts})
        mlflow.set_tags({"task": "model_comparison",
                         "dataset_version": context["dataset_version"] or "unknown",
                         "dataset_manifest_sha256": context["dataset_manifest_sha256"],
                         "full_plane_manifest_sha256": context["full_plane_manifest_sha256"]})
        (local / "dataset_manifest.json").write_bytes(context["manifest_bytes"])
        (local / "full_plane_manifest.json").write_bytes(context["full_manifest_bytes"])
        mlflow.log_artifact(str(local / "dataset_manifest.json"), "data")
        mlflow.log_artifact(str(local / "full_plane_manifest.json"), "data")
        models = []
        for uri in params["models"]:
            info = model_version_info(client, uri, params["serving_endpoint_version"])
            model = mlflow.pyfunc.load_model(uri)
            metadata_classes = (getattr(model.metadata, "metadata", None) or {}).get("classes")
            if metadata_classes is not None and list(metadata_classes) != classes:
                raise ValueError(f"{uri} classes {metadata_classes} differ from {classes}")

            def predict_fn(values, model=model):
                frame = model.predict(pd.DataFrame({"image_base64": values}))
                return frame.to_dict(orient="records")

            with mlflow.start_run(run_name=f"v{info['model_version']}", nested=True,
                                  experiment_id=parent.info.experiment_id) as child:
                evaluation = evaluate_model(predict_fn, sets, classes, params["batch_size"])
                mlflow.set_tags({"task": "model_comparison_member", "parent_eval_run_id": run_id})
                mlflow.log_params({"model_uri": uri, "version": info["model_version"],
                                   "training_run_id": info["training_run_id"], "recipe": info["recipe"],
                                   "dataset_version": info["dataset_version"]})
                flat = flat_metrics(evaluation)
                keys = list(flat)
                for offset in range(0, len(keys), METRIC_BATCH):
                    mlflow.log_metrics({k: flat[k] for k in keys[offset:offset + METRIC_BATCH]})
                _, _, child_predictions = build_rows(run_id, "", [{**info, "evaluation": evaluation}],
                                                     sets, classes)
                mlflow.log_table(data=pd.DataFrame(child_predictions), artifact_file="predictions.json")
            models.append({**info, "child_run_id": child.info.run_id, "evaluation": evaluation})
            del model
        selection, rationale = apply_selection(models)
        for model in models:
            client.set_tag(model["child_run_id"], "selected", str(model["selected"]).lower())
        created_at = em.utc_timestamp()
        runs, metrics, predictions = build_rows(run_id, created_at, models, sets, classes)
        files = {"runs.jsonl": em.jsonl(runs, em.RUN_COLUMNS),
                 "metrics.jsonl": em.jsonl(metrics, em.METRIC_COLUMNS),
                 "predictions.jsonl": em.jsonl(predictions, em.PREDICTION_COLUMNS)}
        tables = local / "tables"
        tables.mkdir()
        for name, text in files.items():
            (tables / name).write_text(text, encoding="utf-8")
            mlflow.log_artifact(str(tables / name), "tables")
        result = result_document(run_id, created_at, params, context, sets, models, selection, rationale)
        result["predictions"] = predictions
        (local / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        mlflow.log_artifact(str(local / "result.json"))
        # Serialize locally first, then copy to the Volume (FUSE) output directory.
        for name in files:
            shutil.copyfile(tables / name, output / name)
        shutil.copyfile(local / "result.json", output / "result.json")
        mlflow.set_tags({"selected_model_version": selection or "none",
                         "crop_parity_mismatches": len(context["crop_parity"]["mismatched_ids"])})
        summary = {"eval_run_id": run_id, "output": str(output), "selected": selection,
                   "rows": {"runs": len(runs), "metrics": len(metrics), "predictions": len(predictions)},
                   "v2_val": {m["model_version"]: m["evaluation"][("v2_val", "tail_crop")]["metrics"]["accuracy"]
                              for m in models},
                   "v1_test": {m["model_version"]: m["evaluation"][("v1_test", "tail_crop")]["metrics"]["accuracy"]
                               for m in models}}
        print("COMPARISON_RESULT=" + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
