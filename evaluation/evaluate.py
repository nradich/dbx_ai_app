"""Paired full-frame/manual-tail evaluation of an immutable registered model."""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def image_input(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    params = yaml.safe_load(Path(os.environ["HYPERPARAMETERS_PATH"]).read_text())
    run_id = os.environ.pop("MLFLOW_RUN_ID")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    root = Path(params["dataset_root"])
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    classes, rows = manifest["classes"], manifest["records"]
    if len(rows) != 10 or any(sum(r["airline"] == name for r in rows) != 2 for name in classes):
        raise ValueError("Expected the frozen ten-photo set, two per airline")
    model_uri = params["model_uri"]
    if not model_uri.endswith("/1"):
        raise ValueError("This evaluation must use immutable model version 1")
    output = Path(params["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({**params, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                           "photos": len(rows), "model_retrained": False})
        mlflow.set_tags({"task": "full_plane_external_evaluation", "manual_crops": "comparison_only_not_detection"})
        model = mlflow.pyfunc.load_model(model_uri)
        records = []
        for row in rows:
            path = root / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Photo hash changed: {row['id']}")
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            left, top, right, bottom = row["bbox"]
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError(f"Invalid tail box: {row['id']}")
            crop = image.crop((math.floor(left * image.width), math.floor(top * image.height),
                               math.ceil(right * image.width), math.ceil(bottom * image.height)))
            inputs = pd.DataFrame({"image_base64": [image_input(image), image_input(crop)]})
            predictions = model.predict(inputs)
            for variant, prediction in zip(("full_plane", "manual_tail_crop"), predictions.to_dict(orient="records")):
                probabilities = json.loads(prediction.pop("probabilities_json"))
                if set(probabilities) != set(classes):
                    raise ValueError("Prediction class list differs from evaluation labels")
                records.append({"id": row["id"], "variant": variant, "actual": row["airline"],
                                "predicted": prediction["airline"], "confidence": prediction["confidence"],
                                "correct": row["airline"] == prediction["airline"],
                                "probabilities": probabilities})
        metrics, by_variant = {}, {}
        for variant in ("full_plane", "manual_tail_crop"):
            subset = [r for r in records if r["variant"] == variant]
            actual, predicted = [r["actual"] for r in subset], [r["predicted"] for r in subset]
            accuracy = float(accuracy_score(actual, predicted))
            f1 = float(f1_score(actual, predicted, labels=classes, average="macro", zero_division=0))
            metrics.update({f"{variant}_accuracy": accuracy, f"{variant}_macro_f1": f1})
            by_variant[variant] = {"correct": sum(r["correct"] for r in subset), "total": len(subset),
                                   "accuracy": accuracy, "macro_f1": f1,
                                   "confusion_matrix": confusion_matrix(actual, predicted, labels=classes).tolist()}
        result = {"run_id": run_id, "model_uri": model_uri, "dataset_root": str(root),
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "classes": classes,
                  "metrics": metrics, "summary": by_variant, "predictions": records,
                  "scope": "Ten external photos; paired manual crops are the same sources, not independent tests; no retraining."}
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(root / "manifest.json"), "data")
        mlflow.log_artifact(str(root / "SOURCES.md"), "data")
        mlflow.log_dict(result, "result.json")
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("EVALUATION_RESULT=" + json.dumps({"result_path": str(output / "result.json"), "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
