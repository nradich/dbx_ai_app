# Databricks notebook source
# MAGIC %md
# MAGIC # Airline-tail classifier: saved results and optional CPU inference
# MAGIC Read actual `training/train.py` results; these cells do not train a model.
# MAGIC Saved-result visualization needs pandas, matplotlib and Pillow, **not PyTorch or MLflow**.
# MAGIC No GPU is needed for visualization or inference. Run on CPU compute with Volume access.
# MAGIC
# MAGIC The model expects a tail-focused image and always chooses one of five airlines.
# MAGIC There is no detector or unknown-airline rejection. Confidence is an uncalibrated model score.

# COMMAND ----------
dbutils.widgets.text("volume_root", "/Volumes/dbx_ai_app/airplane_tail_id/training_data")
dbutils.widgets.text("image_path", "", "Optional tail-focused image in a Volume")
dbutils.widgets.text("result_path", "", "Specific result.json (blank = latest.json pointer)")

import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

root = Path(dbutils.widgets.get("volume_root").strip())
selected = dbutils.widgets.get("result_path").strip()
if not selected:
    pointer = root / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError("No latest.json: complete training or provide a saved result.json path.")
    selected = json.loads(pointer.read_text(encoding="utf-8"))["result_path"]
result = json.loads(Path(selected).read_text(encoding="utf-8"))
dataset_root = Path(result["dataset_root"])
manifest_bytes = (dataset_root / "manifest.json").read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != result["manifest_sha256"]:
    raise ValueError("Dataset manifest differs from the evaluated run.")
manifest = json.loads(manifest_bytes)
photos = {row["id"]: row for row in manifest["records"]}
classes = result["classes"]
print("Saved result:", selected)
display(pd.DataFrame([{key: result[key] for key in
                      ("run_id", "model_uri", "counts", "reload_test", "manifest_sha256")}]))
print("Logged inference dependencies:", ", ".join(result["dependencies"]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Baseline versus trained: held-out test metrics
# MAGIC The baseline has ImageNet-pretrained features and a random airline head. Training freezes
# MAGIC the ResNet-18 backbone and fits its five-class head. Validation accuracy, then validation
# MAGIC loss, selects the checkpoint; test images are scored only after selection.
# MAGIC Expected accuracy is not known in advance. Ten test images give a very noisy estimate:
# MAGIC one error changes accuracy by 10 percentage points. Macro F1 weights each class equally.

# COMMAND ----------
display(pd.DataFrame([
    {"model": name, **{key: result[section][key] for key in ("accuracy", "macro_f1", "loss")}}
    for name, section in [("Before: random airline head", "baseline"), ("After: selected head", "trained")]
]))
display(pd.DataFrame([result["metrics"]]))
print(f"Majority-class baseline accuracy: {result['metrics']['majority_accuracy']:.1%}")
correct = sum(row["correct"] for row in result["trained"]["predictions"])
total = len(result["trained"]["predictions"])
print(f"Trained: {correct}/{total} correct; each test error changes accuracy by {100 / total:.1f} points.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Learning curves (training and validation only)
# MAGIC Training loss includes augmentation; validation loss uses deterministic preprocessing.

# COMMAND ----------
history = pd.DataFrame(result["history"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(history["epoch"], history["train_loss"], label="Train")
axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
axes[1].plot(history["epoch"], history["val_macro_f1"], label="Validation macro F1")
axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1.05), title="Checkpoint selection")
for ax in axes:
    ax.axvline(result["metrics"]["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
    ax.legend()
fig.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Every held-out prediction and crop
# MAGIC Both tables include every saved test prediction and probability vector in `classes` order.
# MAGIC Matrix rows are actual airlines; columns are predicted airlines. Diagonal entries are correct.

# COMMAND ----------
print("Probability vector order:", classes)
for section in ("baseline", "trained"):
    print(section.title(), "test predictions")
    display(pd.DataFrame(result[section]["predictions"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, section in zip(axes, ("baseline", "trained")):
    matrix = result[section]["confusion_matrix"]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, max(map(max, matrix))))
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="Actual",
           title=f"{section.title()} confusion matrix")
    ax.tick_params(axis="x", rotation=45)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
fig.tight_layout()
display(fig)
plt.close(fig)

predictions = result["trained"]["predictions"]
before = {row["id"]: row for row in result["baseline"]["predictions"]}
fig, axes = plt.subplots((total + 2) // 3, 3, squeeze=False, figsize=(13, 3.5 * ((total + 2) // 3)))
for ax in axes.flat:
    ax.axis("off")
for ax, prediction in zip(axes.flat, predictions):
    photo = photos[prediction["id"]]
    if photo["split"] != "test":
        raise ValueError(f"Prediction is not held out: {photo['id']}")
    crop = dataset_root / photo["path"]
    if hashlib.sha256(crop.read_bytes()).hexdigest() != photo["sha256"]:
        raise ValueError(f"Crop hash mismatch: {photo['id']}")
    with Image.open(crop) as image:
        ax.imshow(image)
    initial = before[photo["id"]]
    ax.set_title(f"{photo['id']} | actual: {prediction['actual']}\n"
                 f"Before: {initial['predicted']} ({initial['confidence']:.1%})\n"
                 f"After: {prediction['predicted']} ({prediction['confidence']:.1%}) | "
                 f"{'correct' if prediction['correct'] else 'incorrect'}", fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional live inference — run only when `image_path` is supplied
# MAGIC Use a tail-focused image in a Volume. First install/run with the **matching logged
# MAGIC dependencies** printed above (also in the MLflow model's `requirements.txt`), then restart
# MAGIC Python if required and rerun cells. No fixed `%pip` environment is assumed here.
# MAGIC `load_model` uses the current Python environment; it does not install dependencies.
# MAGIC The immutable registered version URI in this result keeps inference tied to this run.
# MAGIC The `demo` alias is assigned only after logged-model and registered-model reloads match
# MAGIC labels, confidence and all class probabilities against the CPU reference for every test image.

# COMMAND ----------
image_path = dbutils.widgets.get("image_path").strip()
if image_path:
    import mlflow

    if not image_path.startswith("/Volumes/"):
        raise ValueError("image_path must be a /Volumes/... path to a tail-focused image.")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    model = mlflow.pyfunc.load_model(result["model_uri"])
    inputs = pd.DataFrame({"image_base64": [base64.b64encode(Path(image_path).read_bytes()).decode("ascii")]})
    display(model.predict(inputs))
else:
    print("Live inference skipped. Supply image_path and matching logged dependencies to run it.")
