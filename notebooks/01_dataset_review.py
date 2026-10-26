# Databricks notebook source
# MAGIC %md
# MAGIC # Airline-tail dataset review
# MAGIC Five classes: Alaska, American, Delta, Southwest, United. Review the actual prepared
# MAGIC crops and manifest before training. The tiny dataset targets 50 images: 30 train,
# MAGIC 10 validation, 10 test. Related source/session and duplicate groups stay together;
# MAGIC augmentation applies only during training. No model or GPU is needed here.
# MAGIC
# MAGIC Requires pandas, matplotlib and Pillow on CPU compute; no automatic package installs.

# COMMAND ----------
dbutils.widgets.text("dataset_root", "/Volumes/dbx_ai_app/airplane_tail_id/training_data/dataset_v1")

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

root = Path(dbutils.widgets.get("dataset_root").strip())
manifest_bytes = (root / "manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
records = pd.DataFrame(manifest["records"])
classes = manifest["classes"]
print("Dataset:", root)
print("Manifest SHA256:", hashlib.sha256(manifest_bytes).hexdigest())
print("Split policy:", manifest["split_policy"])
counts = records.groupby(["airline", "split"]).size().unstack(fill_value=0)
display(counts.reindex(index=classes, columns=["train", "val", "test"], fill_value=0).reset_index())
display(records[["id", "airline", "split", "group", "source_path", "bbox", "path",
                 "source_sha256", "sha256"]])

# COMMAND ----------
# MAGIC %md
# MAGIC ## Integrity and review notes
# MAGIC SHA256 identifies source bytes and prepared crop bytes; it does not establish licensing.
# MAGIC Exact/decoded-pixel and perceptual duplicate candidates are conservatively co-split.
# MAGIC Attribution is copied into `sources/<class>/SOURCES.md`. Sources for the five existing
# MAGIC Southwest images 1–5 remain unknown; the presence of a source file does not resolve that gap.

# COMMAND ----------
for field in ("group", "source_sha256", "sha256"):
    if records.groupby(field)["split"].nunique().gt(1).any():
        raise ValueError(f"Cross-split leakage: {field}")
for row in manifest["records"]:
    if hashlib.sha256((root / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"Crop hash mismatch: {row['id']}")
print("Crop hashes and group/hash split isolation checked.")
print("Review notes:", manifest.get("review_notes", []))
print("Group reasons:", manifest.get("group_reasons", {}))
print("Duplicate candidates:", manifest["duplicate_checks"]["findings"])
print("Attribution files:", manifest["attribution"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## All prepared crops, labeled by image ID and split
# MAGIC Each class's contact sheet comes from `review/<class>/review.jpg`. These are manually
# MAGIC annotated tail crops, not automatic detections or independently verified model results.

# COMMAND ----------
for airline in classes:
    fig, ax = plt.subplots(figsize=(12, 13))
    with Image.open(root / "review" / airline / "review.jpg") as image:
        ax.imshow(image)
    ax.set_title(airline.title())
    ax.axis("off")
    fig.tight_layout()
    display(fig)
    plt.close(fig)
