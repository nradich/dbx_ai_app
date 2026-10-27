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

from metrics_utils import (brier_score, calibration_bins, expected_calibration_error, per_class_metrics,
                           sample_context_box)
from vision import INPUT_SIZE, MEAN, STD, build_model, evaluation_transform, open_rgb, training_transform

# Defaults reproduce the v1 recipe when a key is absent from the workload parameters.
DEFAULTS = {
    "recipe": "v1", "alias": "demo", "latest_pointer": "latest.json", "label_smoothing": 0.0,
    "fine_tune_layer4_lr": 0.0, "context_jitter": False, "context_scale_range": [1.0, 2.0],
    "random_erasing_p": 0.0, "lr_schedule": "none", "crop_scale": [0.85, 1.0], "crop_ratio": [0.9, 1.1],
    "rotation_degrees": 8.0, "color_jitter": [0.15, 0.15, 0.1, 0.0],
}
CONTEXT_FIELDS = ("context_path", "context_bbox", "context_sha256")


def warn(message):
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no", ""}:
        return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"Expected a boolean, got {value!r}")


def parse_floats(value, length, name):
    if isinstance(value, str):  # Allows CLI overrides such as "[1.0, 2.0]" or "1.0,2.0".
        value = yaml.safe_load(value if value.strip().startswith("[") else f"[{value}]")
    values = [float(v) for v in value]
    if len(values) != length:
        raise ValueError(f"{name} needs {length} numbers, got {value!r}")
    return values


def resolve_config(params):
    raw = {**DEFAULTS, **params}
    config = {
        "recipe": str(raw["recipe"]),
        "alias": "" if raw["alias"] is None else str(raw["alias"]),
        "latest_pointer": "" if raw["latest_pointer"] is None else str(raw["latest_pointer"]),
        "label_smoothing": float(raw["label_smoothing"]),
        "fine_tune_layer4_lr": float(raw["fine_tune_layer4_lr"]),
        "context_jitter": parse_bool(raw["context_jitter"]),
        "context_scale_range": parse_floats(raw["context_scale_range"], 2, "context_scale_range"),
        "random_erasing_p": float(raw["random_erasing_p"]),
        "lr_schedule": str(raw["lr_schedule"]),
        "crop_scale": parse_floats(raw["crop_scale"], 2, "crop_scale"),
        "crop_ratio": parse_floats(raw["crop_ratio"], 2, "crop_ratio"),
        "rotation_degrees": float(raw["rotation_degrees"]),
        "color_jitter": parse_floats(raw["color_jitter"], 4, "color_jitter"),
    }
    if config["recipe"] not in {"v1", "improved"}:
        raise ValueError(f"recipe must be 'v1' or 'improved', got {config['recipe']!r}")
    if config["lr_schedule"] not in {"none", "cosine"}:
        raise ValueError(f"lr_schedule must be 'none' or 'cosine', got {config['lr_schedule']!r}")
    if not 0 <= config["label_smoothing"] < 1 or not 0 <= config["random_erasing_p"] <= 1:
        raise ValueError("label_smoothing must be in [0, 1) and random_erasing_p in [0, 1]")
    if config["fine_tune_layer4_lr"] < 0:
        raise ValueError("fine_tune_layer4_lr must be >= 0 (0 keeps the backbone frozen)")
    low, high = config["context_scale_range"]
    if not 0 < low <= high:
        raise ValueError("context_scale_range must satisfy 0 < low <= high")
    pointer = Path(config["latest_pointer"])
    if config["latest_pointer"] and (pointer.is_absolute() or len(pointer.parts) != 1):
        raise ValueError("latest_pointer must be a plain file name (or empty to skip)")
    return config


class TailDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(open_rgb(self.root / row["path"])), row["label"], row["id"]


class ContextJitterDataset(TailDataset):
    """Training samples cropped from the context image around a scaled, shifted tail box."""

    def __init__(self, root, rows, transform, scale_range, seed):
        super().__init__(root, rows, transform)
        self.scale_range, self.rng = scale_range, random.Random(seed)

    def __getitem__(self, index):
        row = self.rows[index]
        image = open_rgb(self.root / row["context_path"])
        box = sample_context_box(row["context_bbox"], image.size, self.rng, self.scale_range)
        return self.transform(image.crop(box)), row["label"], row["id"]


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


def log_dataset_inputs(source, splits, dataset_version):
    """Record each split as an MLflow dataset input; never fatal."""
    name = f"airline_tails_{dataset_version}"
    for split, subset in splits.items():
        try:
            import mlflow.data
            frame = pd.DataFrame([{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
                                  for row in subset])
            try:
                dataset = mlflow.data.from_pandas(frame, source=str(source / "manifest.json"), name=name,
                                                  targets="label")
            except Exception:  # Source resolution differs between MLflow versions/locations.
                dataset = mlflow.data.from_pandas(frame, name=name, targets="label")
            mlflow.log_input(dataset, context=split)
        except Exception as error:  # noqa: BLE001 - dataset lineage must never kill training.
            warn(f"mlflow.log_input failed for split {split}: {error!r}")


def log_figures(history, confusion, classes, bins):
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        mlflow.set_tag("figures", "unavailable")
        return
    try:
        epochs = [row["epoch"] for row in history]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
        axes[0].plot(epochs, [r["val_loss"] for r in history], label="validation")
        axes[0].set(title="Loss", xlabel="epoch")
        axes[1].plot(epochs, [r["train_accuracy"] for r in history], label="train")
        axes[1].plot(epochs, [r["val_accuracy"] for r in history], label="validation")
        axes[1].set(title="Accuracy", xlabel="epoch", ylim=(0, 1.05))
        axes[2].plot(epochs, [r["lr"] for r in history])
        axes[2].set(title="Head learning rate", xlabel="epoch")
        for axis in axes[:2]:
            axis.legend()
        figure.tight_layout()
        mlflow.log_figure(figure, "figures/learning_curves.png")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(6, 5))
        matrix = np.asarray(confusion)
        image = axis.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, str(value), ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black")
        axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
                 xlabel="predicted", ylabel="actual", title="Test confusion matrix")
        axis.tick_params(axis="x", rotation=45)
        figure.colorbar(image)
        figure.tight_layout()
        mlflow.log_figure(figure, "figures/confusion_matrix.png")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(5, 5))
        width = bins[0]["upper"] - bins[0]["lower"]
        filled = [b for b in bins if b["count"]]
        axis.bar([b["lower"] for b in filled], [b["accuracy"] for b in filled], width=width, align="edge",
                 edgecolor="black", alpha=0.7, label="accuracy")
        axis.plot([b["confidence"] for b in filled], [b["accuracy"] for b in filled], "o", color="tab:red",
                  label="mean confidence")
        for b in filled:
            axis.text(b["lower"] + width / 2, min(b["accuracy"] + 0.02, 1.0), f"n={b['count']}", ha="center")
        axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        axis.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="confidence", ylabel="accuracy", title="Test reliability")
        axis.legend(loc="upper left")
        figure.tight_layout()
        mlflow.log_figure(figure, "figures/reliability.png")
        plt.close(figure)
    except Exception as error:  # noqa: BLE001 - figures are diagnostics only.
        warn(f"figure logging failed: {error!r}")
        mlflow.set_tag("figures", "error")


def verifier_source(code):
    verifier = (code / "verify_model.py").read_text()
    # Local smoke tests only: point the isolated verifier at the overridden stores. On AIR (no
    # overrides) the verifier text runs unchanged.
    for variable, default, setter in (("TAILSPOT_MLFLOW_TRACKING_URI", "databricks", "set_tracking_uri"),
                                      ("TAILSPOT_MLFLOW_REGISTRY_URI", "databricks-uc", "set_registry_uri")):
        if variable in os.environ:
            verifier = verifier.replace(f'mlflow.{setter}("{default}")',
                                        f'mlflow.{setter}(__import__("os").environ["{variable}"])')
    return verifier


def main():
    params = yaml.safe_load(Path(os.environ["HYPERPARAMETERS_PATH"]).read_text())
    config = resolve_config(params)
    run_id = os.environ.pop("MLFLOW_RUN_ID")
    mlflow.set_tracking_uri(os.environ.get("TAILSPOT_MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_registry_uri(os.environ.get("TAILSPOT_MLFLOW_REGISTRY_URI", "databricks-uc"))
    seed = int(params["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available() and os.environ.get("TAILSPOT_ALLOW_CPU") != "1":
        raise RuntimeError("Submit this training script through AIR with one GPU")
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    synchronize = torch.cuda.synchronize if cuda else (lambda: None)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source = Path(params["dataset_root"])
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    dataset_version = str(manifest.get("dataset_version") or "v1")
    parent_manifest_sha256 = manifest.get("parent_manifest_sha256")
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
    if config["context_jitter"]:
        missing = [row["id"] for row in splits["train"] if any(not row.get(field) for field in CONTEXT_FIELDS)]
        if missing:
            raise ValueError(f"context_jitter=true needs a v2 manifest with {', '.join(CONTEXT_FIELDS)} on every "
                             f"train record (dataset_version={dataset_version}); missing for {len(missing)} "
                             f"records, e.g. {missing[:3]}")
    output = Path(params["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with mlflow.start_run(run_id=run_id) as active_run, tempfile.TemporaryDirectory() as temp:
        experiment_id = active_run.info.experiment_id
        local = Path(temp)
        data = local / "data"
        copies = [(row["path"], row["sha256"], row["id"]) for row in rows]
        if config["context_jitter"]:
            copies += [(row["context_path"], row["context_sha256"], f"{row['id']} (context)")
                       for row in splits["train"]]
        for relative, expected_sha256, label in copies:
            destination = data / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
                raise ValueError(f"Dataset integrity check failed: {label}")
        # Preserve exact inputs for historical results playback independently of the source dataset.
        snapshot = output / "dataset"
        shutil.copytree(data, snapshot)
        shutil.copyfile(source / "manifest.json", snapshot / "manifest.json")
        shutil.copytree(source / "sources", snapshot / "sources")
        shutil.copytree(source / "review", snapshot / "review")
        mlflow.log_params({**params, **{key: value for key, value in config.items() if key not in params},
                           "dataset_version": dataset_version, "architecture": "resnet18",
                           "backbone": "frozen ImageNet1K V1" if not config["fine_tune_layer4_lr"]
                           else "ImageNet1K V1, layer4 fine-tuned (BatchNorm running statistics fixed)",
                           "manifest_sha256": manifest_sha256,
                           **{f"{name}_images": len(subset) for name, subset in splits.items()}})
        tags = {"task": "airline_tail_classification", "input_scope": "user-selected tail crop",
                "checkpoint_selection": "validation accuracy then validation loss",
                "dataset_version": dataset_version, "recipe": config["recipe"]}
        if parent_manifest_sha256:
            tags["parent_manifest_sha256"] = parent_manifest_sha256
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(source / "manifest.json"), "dataset")
        mlflow.log_artifacts(str(source / "sources"), "dataset/sources")
        mlflow.log_artifacts(str(source / "review"), "dataset/review")
        log_dataset_inputs(source, splits, dataset_version)
        train_transform = training_transform(
            crop_scale=config["crop_scale"], crop_ratio=config["crop_ratio"],
            rotation_degrees=config["rotation_degrees"], color_jitter=config["color_jitter"],
            random_erasing_p=config["random_erasing_p"])

        def make_dataset(name, subset):
            if name != "train":
                return TailDataset(data, subset, evaluation_transform())
            if config["context_jitter"]:
                return ContextJitterDataset(data, subset, train_transform, config["context_scale_range"], seed)
            return TailDataset(data, subset, train_transform)

        loaders = {
            name: DataLoader(make_dataset(name, subset),
                             batch_size=int(params["batch_size"]), shuffle=name == "train", num_workers=0,
                             generator=torch.Generator().manual_seed(seed))
            for name, subset in splits.items()
        }
        model = build_model(len(classes), pretrained=True).to(device)
        initial_state = copy.deepcopy(model.state_dict())
        learning_rate = float(params["learning_rate"])
        if config["fine_tune_layer4_lr"] > 0:
            for parameter in model.layer4.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": list(model.fc.parameters()), "lr": learning_rate},
                {"params": list(model.layer4.parameters()), "lr": config["fine_tune_layer4_lr"]},
            ], weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=learning_rate, weight_decay=0.01)
        epochs = int(params["epochs"])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                     if config["lr_schedule"] == "cosine" else None)
        loss_function = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        best_key, best_state, best_epoch, history = (-1.0, -float("inf")), None, 0, []
        synchronize()
        train_started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            # All BatchNorm layers stay in eval mode (running statistics fixed, small batches); only the
            # head (and layer4 conv weights when fine-tuning) receive gradient updates.
            model.eval()
            model.fc.train()
            head_lr = optimizer.param_groups[0]["lr"]
            loss_sum, count, correct = 0.0, 0, 0
            for images, labels, _ in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                labels = labels.to(device)
                logits = model(images.to(device))
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                count += len(labels)
                correct += int((logits.detach().argmax(-1) == labels).sum().item())
            if scheduler is not None:
                scheduler.step()
            validation = score(model, loaders["val"], classes)
            synchronize()
            metrics = {"train_loss": loss_sum / count, "train_accuracy": correct / count,
                       "val_accuracy": validation["accuracy"], "val_loss": validation["loss"],
                       "val_macro_f1": validation["macro_f1"], "lr": head_lr,
                       "epoch_seconds": time.perf_counter() - epoch_started}
            history.append({"epoch": epoch, **metrics})
            mlflow.log_metrics(metrics, step=epoch)
            key = (validation["accuracy"], -validation["loss"])
            if key > best_key:
                best_key, best_epoch, best_state = key, epoch, copy.deepcopy(model.state_dict())
            print(json.dumps({"epoch": epoch, **metrics}), flush=True)
        synchronize()
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
        # Test-set diagnostics (selected checkpoint): calibration, per-class metrics, tables and figures.
        test_labels = [classes.index(r["actual"]) for r in trained["predictions"]]
        test_probabilities = [r["probabilities"] for r in trained["predictions"]]
        test_predicted = [classes.index(r["predicted"]) for r in trained["predictions"]]
        reliability = calibration_bins(test_probabilities, test_labels)
        per_class = per_class_metrics(test_labels, test_predicted, classes)
        mlflow.log_table(pd.DataFrame([
            {"id": r["id"], "actual": r["actual"], "predicted": r["predicted"], "confidence": r["confidence"],
             "true_prob": r["probabilities"][label], "correct": r["correct"],
             **{f"prob_{name}": r["probabilities"][k] for k, name in enumerate(classes)}}
            for r, label in zip(trained["predictions"], test_labels)]), artifact_file="eval/test_predictions.json")
        mlflow.log_table(pd.DataFrame(per_class), artifact_file="eval/test_per_class.json")
        log_figures(history, trained["confusion_matrix"], classes, reliability)
        code = Path(__file__).parent
        info = mlflow.pyfunc.log_model(
            name="airline_tail_classifier", python_model=str(code / "model.py"),
            code_paths=[str(code / "vision.py")], artifacts={"checkpoint": str(checkpoint)},
            input_example=sample.iloc[:1], signature=infer_signature(sample.iloc[:1], example_output),
            pip_requirements=dependencies,
            metadata={"input_scope": "tail-focused RGB image, base64 encoded; no automatic tail detection",
                      "classes": classes, "manifest_sha256": manifest_sha256},
        )
        sample.to_json(local / "inputs.json", orient="records")
        (local / "expected.json").write_text(json.dumps({"classes": classes, "predictions": cpu_reference["predictions"]}))
        verifier = verifier_source(code)
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
        best_val_accuracy, best_val_loss = best_key[0], -best_key[1]
        for tag, value in {"reload_test": "passed_all_test_images", "recipe": config["recipe"],
                           "dataset_version": dataset_version, "val_accuracy": repr(best_val_accuracy),
                           "val_loss": repr(best_val_loss), "training_run_id": run_id}.items():
            client.set_model_version_tag(name, registered.version, tag, value)
        majority = max(range(len(classes)), key=lambda label: sum(row["label"] == label for row in splits["train"]))
        majority_accuracy = sum(row["label"] == majority for row in splits["test"]) / len(splits["test"])
        metrics = {"test_accuracy": trained["accuracy"], "test_macro_f1": trained["macro_f1"],
                   "baseline_test_accuracy": baseline["accuracy"], "majority_accuracy": majority_accuracy,
                   "training_seconds": training_seconds, "best_epoch": best_epoch,
                   "best_val_accuracy": best_val_accuracy, "best_val_loss": best_val_loss,
                   "test_loss": trained["loss"],
                   "test_ece": expected_calibration_error(test_probabilities, test_labels),
                   "test_brier": brier_score(test_probabilities, test_labels),
                   **{f"test_{metric}_{row['class']}": row[metric]
                      for row in per_class for metric in ("f1", "precision", "recall")}}
        mlflow.log_metrics(metrics)
        alias = config["alias"]
        result = {"run_id": run_id, "experiment_id": experiment_id, "registered_model": name,
                  "version": registered.version, "alias": alias or None,
                  "model_uri": registered_uri, "classes": classes, "counts": {k: len(v) for k, v in splits.items()},
                  "manifest_sha256": manifest_sha256, "dataset_version": dataset_version,
                  "parent_manifest_sha256": parent_manifest_sha256, "recipe": config["recipe"], "config": config,
                  "dataset_root": str(snapshot), "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
                  "training_environment": environment,
                  "dependencies": dependencies, "metrics": metrics, "history": history,
                  "trained": trained, "baseline": baseline, "cpu_reference": cpu_reference,
                  "per_class": per_class, "reliability_bins": reliability,
                  "reload_test": "passed_all_test_images",
                  "registered_predictions": verified.to_dict(orient="records"),
                  "total_seconds": time.perf_counter() - started}
        mlflow.log_dict(result, "result.json")
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if alias:
            client.set_registered_model_alias(name, alias, registered.version)
        result_path = str(output / "result.json")
        if config["latest_pointer"]:
            latest = {"result_path": result_path}
            (Path(params["output_root"]).parent / config["latest_pointer"]).write_text(json.dumps(latest),
                                                                                      encoding="utf-8")
        mlflow.set_tags({"registered_model": name, "registered_model_version": registered.version,
                         "reload_test": "passed_all_test_images"})
        print("TRAINING_RESULT=" + json.dumps({"result_path": result_path, "metrics": metrics,
                                                "model_uri": registered_uri}), flush=True)


if __name__ == "__main__":
    main()
