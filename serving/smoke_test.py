"""Verify the serving endpoint against all ten original held-out CPU predictions.

Requires only Pillow locally; Databricks CLI supplies authentication. No local
PyTorch, MLflow, model download or training is required.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

from PIL import Image, ImageOps


def cli(profile, *args):
    result = subprocess.run(["databricks", *args, "--profile", profile, "--output", "json"],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--endpoint", default="airplane-tail-identifier")
    parser.add_argument("--dataset", type=Path, default=Path(".local/dataset"))
    parser.add_argument("--reference", type=Path, default=Path(".local/training_result.json"))
    parser.add_argument("--output", type=Path, default=Path(".local/serving_smoke_result.json"))
    args = parser.parse_args()
    endpoint = cli(args.profile, "serving-endpoints", "get", args.endpoint)
    if endpoint["state"] != {"ready": "READY", "config_update": "NOT_UPDATING"}:
        raise RuntimeError(f"Endpoint is not fully ready: {endpoint['state']}")
    entities = endpoint["config"]["served_entities"]
    expected_name = "dbx_ai_app.airplane_tail_id.airplane_tail_identifier"
    if len(entities) != 1 or entities[0]["entity_name"] != expected_name or entities[0]["entity_version"] != "1":
        raise RuntimeError("Endpoint must serve only the evaluated model version 1")
    manifest_bytes = (args.dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if (reference["registered_model"] != expected_name or str(reference["version"]) != "1"
            or reference["model_uri"] != f"models:/{expected_name}/1"):
        raise ValueError("Reference must describe the same registered model version 1")
    if hashlib.sha256(manifest_bytes).hexdigest() != reference["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the evaluated run")
    expected = {r["id"]: r for r in reference["cpu_reference"]["predictions"]}
    classes = reference["classes"]
    records = []
    for row in manifest["records"]:
        if row["split"] != "test":
            continue
        raw = (args.dataset / row["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Crop hash mismatch: " + row["id"])
        # Keep the transmitted crop RGB and metadata-free, as an app should.
        with Image.open(io.BytesIO(raw)) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        payload = {"dataframe_records": [{"image_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}]}
        with tempfile.TemporaryDirectory() as temp:
            request = Path(temp) / "request.json"
            request.write_text(json.dumps(payload), encoding="utf-8")
            started = time.perf_counter()
            response = cli(args.profile, "serving-endpoints", "query", args.endpoint, "--json", "@" + str(request))
            elapsed = time.perf_counter() - started
        predictions = response["predictions"]
        if len(predictions) != 1:
            raise AssertionError("Expected one prediction for one image")
        prediction = predictions[0]
        target = expected[row["id"]]
        if prediction["airline"] != target["predicted"]:
            raise AssertionError("Endpoint label mismatch: " + row["id"])
        probabilities = json.loads(prediction["probabilities_json"])
        if set(probabilities) != set(classes):
            raise AssertionError("Endpoint class list mismatch")
        for actual, wanted in zip([prediction["confidence"], *[probabilities[c] for c in classes]],
                                  [target["confidence"], *target["probabilities"]]):
            if not math.isclose(actual, wanted, rel_tol=1e-4, abs_tol=1e-5):
                raise AssertionError("Endpoint probability mismatch: " + row["id"])
        records.append({"id": row["id"], "actual": row["airline"], "predicted": prediction["airline"],
                        "confidence": prediction["confidence"], "correct": row["airline"] == prediction["airline"],
                        "matches_registered_cpu_reference": True, "cli_roundtrip_seconds": elapsed})
        print(row["id"], prediction["airline"], f"{prediction['confidence']:.1%}", flush=True)
    if len(records) != 10:
        raise AssertionError("Expected all ten held-out requests")
    result = {"endpoint": args.endpoint, "model": expected_name, "version": "1",
              "checked": len(records), "correct": sum(r["correct"] for r in records),
              "all_match_registered_cpu_reference": True, "predictions": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}))


if __name__ == "__main__":
    main()
