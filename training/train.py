"""Train on Databricks AIR, evaluate once, and register a reload-tested UC model."""
import base64
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from torch.utils.data import DataLoader, Dataset

from vision import INPUT_SIZE, MEAN, STD, build_model, evaluation_transform, open_rgb, training_transform


class TailDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(open_rgb(self.root / row["path"])), row["label"], row["id"]


def score(model, loader, classes):
    model.eval()
    records, labels, probabilities = [], [], []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for images, targets, ids in loader:
            probs = model(images.to(device)).softmax(-1).cpu().numpy()
            for image_id, label, vector in zip(ids, targets.tolist(), probs):
                index = int(vector.argmax())
                labels.append(label)
                probabilities.append(vector.tolist())
                records.append({"id": image_id, "actual": classes[label], "predicted": classes[index],
                                "confidence": float(vector[index]), "probabilities": vector.tolist(),
                                "correct": label == index})
    predicted = np.argmax(probabilities, axis=1)
    normalized = np.asarray(probabilities, dtype=np.float64)
    normalized /= normalized.sum(axis=1, keepdims=True)
    indices = list(range(len(classes)))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=indices, average="macro", zero_division=0)),
        "loss": float(log_loss(labels, normalized, labels=indices)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=indices).tolist(),
        "predictions": records,
    }


def main():
    params = yaml.safe_load(Path(os.environ["HYPERPARAMETERS_PATH"]).read_text())
    run_id = os.environ.pop("MLFLOW_RUN_ID")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    seed = int(params["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Submit this training script through AIR with one GPU")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = Path(params["dataset_root"])
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    classes, rows = manifest["classes"], manifest["records"]
    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    if any(not subset for subset in splits.values()):
        raise ValueError("All three dataset splits must be nonempty")
    for key in ("group", "source_sha256", "sha256"):
        assignments = {}
        for row in rows:
            previous = assignments.setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Cross-split leakage in {key}: {row['id']}")
    output = Path(params["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with mlflow.start_run(run_id=run_id), tempfile.TemporaryDirectory() as temp:
        local = Path(temp)
        data = local / "data"
        for row in rows:
            destination = data / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / row["path"], destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Dataset integrity check failed: {row['id']}")
        # Preserve exact inputs for historical results playback independently of dataset_v1.
        snapshot = output / "dataset"
        shutil.copytree(data, snapshot)
        shutil.copyfile(source / "manifest.json", snapshot / "manifest.json")
        shutil.copytree(source / "sources", snapshot / "sources")
        shutil.copytree(source / "review", snapshot / "review")
        mlflow.log_params({**params, "architecture": "resnet18", "backbone": "frozen ImageNet1K V1",
                           "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                           **{f"{name}_images": len(subset) for name, subset in splits.items()}})
        mlflow.set_tags({"task": "airline_tail_classification", "input_scope": "user-selected tail crop",
                         "checkpoint_selection": "validation accuracy then validation loss"})
        mlflow.log_artifact(str(source / "manifest.json"), "dataset")
        mlflow.log_artifacts(str(source / "sources"), "dataset/sources")
        mlflow.log_artifacts(str(source / "review"), "dataset/review")
        loaders = {
            name: DataLoader(TailDataset(data, subset, training_transform() if name == "train" else evaluation_transform()),
                             batch_size=int(params["batch_size"]), shuffle=name == "train", num_workers=0,
                             generator=torch.Generator().manual_seed(seed))
            for name, subset in splits.items()
        }
        model = build_model(len(classes), pretrained=True).cuda()
        initial_state = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=float(params["learning_rate"]), weight_decay=0.01)
        loss_function = torch.nn.CrossEntropyLoss()
        best_key, best_state, best_epoch, history = (-1.0, -float("inf")), None, 0, []
        torch.cuda.synchronize()
        train_started = time.perf_counter()
        for epoch in range(1, int(params["epochs"]) + 1):
            model.eval()  # Keep frozen backbone batch-normalization statistics fixed.
            model.fc.train()
            loss_sum, count = 0.0, 0
            for images, labels, _ in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(images.cuda()), labels.cuda())
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
            validation = score(model, loaders["val"], classes)
            metrics = {"train_loss": loss_sum / count, "val_accuracy": validation["accuracy"],
                       "val_loss": validation["loss"], "val_macro_f1": validation["macro_f1"]}
            history.append({"epoch": epoch, **metrics})
            mlflow.log_metrics(metrics, step=epoch)
            key = (validation["accuracy"], -validation["loss"])
            if key > best_key:
                best_key, best_epoch, best_state = key, epoch, copy.deepcopy(model.state_dict())
            print(json.dumps({"epoch": epoch, **metrics}), flush=True)
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - train_started
        # Test data is consulted only after the final checkpoint has been selected.
        model.load_state_dict(best_state)
        trained = score(model, loaders["test"], classes)
        model.load_state_dict(initial_state)
        baseline = score(model, loaders["test"], classes)
        model.load_state_dict(best_state)
        model.cpu().eval()
        cpu_reference = score(model, loaders["test"], classes)
        if [r["predicted"] for r in cpu_reference["predictions"]] != [r["predicted"] for r in trained["predictions"]]:
            raise AssertionError("CPU/GPU evaluation label mismatch")
        np.testing.assert_allclose([r["probabilities"] for r in cpu_reference["predictions"]],
                                   [r["probabilities"] for r in trained["predictions"]], atol=1e-3, rtol=1e-3)
        checkpoint = local / "checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "classes": classes, "input_size": INPUT_SIZE,
                    "normalization_mean": MEAN, "normalization_std": STD}, checkpoint)
        shutil.copyfile(checkpoint, output / checkpoint.name)  # Serialize locally before Volume FUSE copy.
        sample = pd.DataFrame({"image_base64": [base64.b64encode((data / row["path"]).read_bytes()).decode("ascii")
                                                for row in splits["test"]]})
        example_output = pd.DataFrame({"airline": [classes[0]], "confidence": [0.5],
                                       "probabilities_json": [json.dumps({name: 1 / len(classes) for name in classes})]})
        dependencies = [f"{name}=={importlib.metadata.version(name).split('+')[0]}"
                        for name in ("torch", "torchvision", "pillow", "pandas")]
        dependencies.append(f"mlflow=={mlflow.__version__}")
        environment = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                       "cudnn": torch.backends.cudnn.version(),
                       "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                                    if dist.metadata["Name"]}}
        mlflow.log_dict(environment, "environment.json")
        code = Path(__file__).parent
        info = mlflow.pyfunc.log_model(
            name="airline_tail_classifier", python_model=str(code / "model.py"),
            code_paths=[str(code / "vision.py")], artifacts={"checkpoint": str(checkpoint)},
            input_example=sample.iloc[:1], signature=infer_signature(sample.iloc[:1], example_output),
            pip_requirements=dependencies,
            metadata={"input_scope": "tail-focused RGB image, base64 encoded; no automatic tail detection",
                      "classes": classes, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        )
        sample.to_json(local / "inputs.json", orient="records")
        (local / "expected.json").write_text(json.dumps({"classes": classes, "predictions": cpu_reference["predictions"]}))
        verifier = (code / "verify_model.py").read_text()
        def verify_predictions(uri):
            # Fresh interpreter, no source directory/PYTHONPATH: only logged code can supply vision.
            environment_vars = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            subprocess.run([sys.executable, "-c", verifier, uri, str(local / "inputs.json"),
                            str(local / "expected.json"), str(local / "verified.json")],
                           cwd=local, env=environment_vars, check=True, timeout=300)
            return pd.read_json(local / "verified.json", orient="records")
        verify_predictions(info.model_uri)
        name = params["registered_model_name"]
        registered = mlflow.register_model(info.model_uri, name)
        registered_uri = f"models:/{name}/{registered.version}"
        verified = verify_predictions(registered_uri)
        client = mlflow.MlflowClient()
        client.set_model_version_tag(name, registered.version, "reload_test", "passed_all_test_images")
        majority = max(range(len(classes)), key=lambda label: sum(row["label"] == label for row in splits["train"]))
        majority_accuracy = sum(row["label"] == majority for row in splits["test"]) / len(splits["test"])
        metrics = {"test_accuracy": trained["accuracy"], "test_macro_f1": trained["macro_f1"],
                   "baseline_test_accuracy": baseline["accuracy"], "majority_accuracy": majority_accuracy,
                   "training_seconds": training_seconds, "best_epoch": best_epoch}
        mlflow.log_metrics(metrics)
        result = {"run_id": run_id, "registered_model": name, "version": registered.version, "alias": "demo",
                  "model_uri": registered_uri, "classes": classes, "counts": {k: len(v) for k, v in splits.items()},
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "dataset_root": str(snapshot), "gpu": torch.cuda.get_device_name(0),
                  "training_environment": environment,
                  "dependencies": dependencies, "metrics": metrics, "history": history,
                  "trained": trained, "baseline": baseline, "cpu_reference": cpu_reference,
                  "reload_test": "passed_all_test_images",
                  "registered_predictions": verified.to_dict(orient="records"),
                  "total_seconds": time.perf_counter() - started}
        mlflow.log_dict(result, "result.json")
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        client.set_registered_model_alias(name, "demo", registered.version)
        latest = {"result_path": str(output / "result.json")}
        (Path(params["output_root"]).parent / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
        mlflow.set_tags({"registered_model": name, "registered_model_version": registered.version,
                         "reload_test": "passed_all_test_images"})
        print("TRAINING_RESULT=" + json.dumps({"result_path": latest["result_path"], "metrics": metrics,
                                                "model_uri": registered_uri}), flush=True)


if __name__ == "__main__":
    main()
