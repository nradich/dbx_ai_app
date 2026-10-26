# Databricks notebook source
# MAGIC %md
# MAGIC # External full-plane test: whole image versus manual tail crop
# MAGIC Ten new photographs, two per airline, selected before inference. Version 1 is unchanged.
# MAGIC Manual crops are a paired comparison on the same sources, not an automatic detector.

# COMMAND ----------
dbutils.widgets.text("result_path", "", "Evaluation result.json in a Volume")

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

result_path = dbutils.widgets.get("result_path").strip()
if not result_path:
    raise ValueError("Set result_path to the completed evaluation's result.json")
result = json.loads(Path(result_path).read_text())
root = Path(result["dataset_root"])
manifest_bytes = (root / "manifest.json").read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest() != result["manifest_sha256"]:
    raise ValueError("Evaluation manifest has changed")
manifest = json.loads(manifest_bytes)
print("Immutable model:", result["model_uri"])
display(pd.DataFrame(result["summary"]).T.drop(columns="confusion_matrix"))
print("Only ten sources; each error changes accuracy by ten percentage points.")

# COMMAND ----------
predictions = {(row["id"], row["variant"]): row for row in result["predictions"]}
display(pd.DataFrame(result["predictions"]))
for row in manifest["records"]:
    path = root / row["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Image integrity check failed: " + row["id"])
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    left, top, right, bottom = row["bbox"]
    crop = image.crop((math.floor(left * image.width), math.floor(top * image.height),
                       math.ceil(right * image.width), math.ceil(bottom * image.height)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, view, variant in zip(axes, (image, crop), ("full_plane", "manual_tail_crop")):
        prediction = predictions[(row["id"], variant)]
        ax.imshow(view)
        ax.axis("off")
        ax.set_title(f"{variant}: {prediction['predicted']} ({prediction['confidence']:.1%})\n"
                     f"Actual: {row['airline']} — {'correct' if prediction['correct'] else 'incorrect'}")
    fig.suptitle(row["id"])
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    print(row["artist"], "|", row["license"], "|", row["source_url"], "|", row["license_url"])

# COMMAND ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
classes = result["classes"]
for ax, variant in zip(axes, ("full_plane", "manual_tail_crop")):
    matrix = result["summary"][variant]["confusion_matrix"]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=2)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted", ylabel="Actual", title=variant)
    ax.tick_params(axis="x", rotation=45)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
fig.tight_layout()
display(fig)
plt.close(fig)
