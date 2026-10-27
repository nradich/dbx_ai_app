# Databricks notebook source
# MAGIC %md
# MAGIC # Model performance: comparing registered airline-tail model versions
# MAGIC Read-only review of one published comparison plus training history and app usage.
# MAGIC
# MAGIC **Data sources**
# MAGIC - Unity Catalog tables `model_eval_runs`, `model_eval_metrics`, `model_eval_predictions`, written by
# MAGIC   `evaluation/publish.py` from an `evaluation/compare.py` run (contracts: `docs/model_performance.md`).
# MAGIC - MLflow experiments `airplane-tail-identifier` (training runs) and `airplane-tail-evaluation`
# MAGIC   (one parent run per comparison; `eval_run_id` is its run ID).
# MAGIC - `photo_submissions` (photos scored by the app, with optional reviewer labels).
# MAGIC
# MAGIC **Caveats**
# MAGIC - Test sets are tiny (n is shown everywhere); one image can move accuracy by 6–17 points.
# MAGIC - Confidence is the uncalibrated softmax maximum; ECE on so few images is noisy.
# MAGIC - `full_plane_stress` is contaminated for v2-trained versions: v2 trained on related Acroterion BWI photos.
# MAGIC - `selected` marks the candidate chosen on validation data only; it is not a promotion decision.

# COMMAND ----------
dbutils.widgets.text("schema", "dbx_ai_app.airplane_tail_id", "Catalog.schema")
dbutils.widgets.text("eval_run_id", "", "Comparison run ID (blank = latest)")
dbutils.widgets.text("eval_set", "v2_test", "Eval set for detail sections")
dbutils.widgets.text("variant", "tail_crop", "Variant for detail sections")

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

schema_text = dbutils.widgets.get("schema").strip()
if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+", schema_text):
    raise ValueError("schema must look like catalog.schema (letters, digits, underscores)")
SCHEMA = ".".join(f"`{part}`" for part in schema_text.split("."))
eval_run_id = dbutils.widgets.get("eval_run_id").strip()
eval_set = dbutils.widgets.get("eval_set").strip()
variant = dbutils.widgets.get("variant").strip()


def show(fig):
    fig.tight_layout()
    display(fig)
    plt.close(fig)

# COMMAND ----------
if not eval_run_id:
    latest = spark.sql(f"SELECT eval_run_id FROM {SCHEMA}.model_eval_runs ORDER BY created_at DESC, eval_run_id DESC LIMIT 1").collect()
    eval_run_id = latest[0]["eval_run_id"] if latest else ""
args = {"id": eval_run_id}
runs = spark.sql(f"SELECT * FROM {SCHEMA}.model_eval_runs WHERE eval_run_id = :id", args=args).toPandas()
if runs.empty:
    message = ("No published model comparison found. Run evaluation/compare.py, then evaluation/publish.py."
               if not eval_run_id else f"No comparison with eval_run_id {eval_run_id!r}.")
    print(message)
    dbutils.notebook.exit(message)
metrics = spark.sql(f"SELECT * FROM {SCHEMA}.model_eval_metrics WHERE eval_run_id = :id", args=args).toPandas()
predictions = spark.sql(
    f"SELECT model_version, eval_set, variant, item_id, actual, predicted, confidence, true_prob, correct, source "
    f"FROM {SCHEMA}.model_eval_predictions WHERE eval_run_id = :id", args=args).toPandas()
versions = sorted(runs["model_version"].unique(), key=lambda v: (0, int(v), "") if str(v).isdigit() else (1, 0, str(v)))
print("Comparison:", eval_run_id, "| created", runs["created_at"].max(), "| versions:", ", ".join(versions))

# COMMAND ----------
# MAGIC %md ## (a) Models in this comparison

# COMMAND ----------
display(runs[["model_version", "recipe", "dataset_version", "val_accuracy", "val_loss", "aliases",
              "serving", "selected", "training_run_id", "child_run_id", "model_uri"]]
        .sort_values("model_version", key=lambda s: s.map(versions.index)))

# COMMAND ----------
# MAGIC %md ## (b) Headline metrics by eval set and variant (n = images scored)

# COMMAND ----------
headline = metrics[metrics["metric"].isin(["accuracy", "macro_f1", "ece"])]
pivot = headline.pivot_table(index=["eval_set", "variant"], columns=["metric", "model_version"],
                             values="value", aggfunc="first")
counts = metrics.pivot_table(index=["eval_set", "variant"], columns="model_version", values="n", aggfunc="max")
counts.columns = pd.MultiIndex.from_product([["n"], counts.columns])
display(pivot.join(counts).round(3).reset_index())

# COMMAND ----------
# MAGIC %md ## (c) Accuracy per eval set (tail_crop and whole_image variants)

# COMMAND ----------
acc = metrics[(metrics["metric"] == "accuracy") & metrics["variant"].isin(["tail_crop", "whole_image"])].copy()
acc["label"] = acc["eval_set"] + "\n" + acc["variant"]
labels = sorted(acc["label"].unique())
width = 0.8 / max(len(versions), 1)
fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 5))
for k, version in enumerate(versions):
    rows = acc[acc["model_version"] == version].set_index("label").reindex(labels)
    xs = np.arange(len(labels)) + k * width
    ax.bar(xs, rows["value"].fillna(0), width, label=f"v{version}")
    for x, value, n in zip(xs, rows["value"], rows["n"]):
        if pd.notna(value):
            ax.text(x, value + 0.01, f"n={int(n)}", ha="center", va="bottom", fontsize=7)
ax.set(xticks=np.arange(len(labels)) + width * (len(versions) - 1) / 2, ylim=(0, 1.12), ylabel="Accuracy")
ax.set_xticklabels(labels, fontsize=8)
ax.legend(title="Model version")
show(fig)

# COMMAND ----------
# MAGIC %md ## (d) Context sensitivity: does wider context pull predictions towards Southwest?

# COMMAND ----------
ctx = metrics[metrics["eval_set"] == "context_sensitivity"]
if ctx.empty:
    print("No context_sensitivity metrics in this comparison.")
else:
    order = ["context_1.0", "context_1.5", "context_2.0", "context_3.0", "context_full"]
    order = [v for v in order if v in set(ctx["variant"])] + sorted(set(ctx["variant"]) - set(order))
    sw_metric = next((m for m in ctx["metric"].unique()
                      if m.startswith("share_pred_") and "southwest" in m.lower()), None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("accuracy", sw_metric)):
        for version in versions:
            rows = ctx[(ctx["model_version"] == version) & (ctx["metric"] == metric)].set_index("variant").reindex(order)
            ax.plot(order, rows["value"], marker="o", label=f"v{version}")
        ax.set(title=metric or "share_pred_southwest (missing)", ylim=(0, 1.05), xlabel="Context scale")
        ax.legend()
    print("n per scale:", ctx.groupby("variant")["n"].max().reindex(order).to_dict())
    show(fig)

# COMMAND ----------
# MAGIC %md ## (e) Confusion matrices and mistakes for the selected eval set / variant

# COMMAND ----------
detail = predictions[(predictions["eval_set"] == eval_set) & (predictions["variant"] == variant)]
if detail.empty:
    print(f"No predictions for eval_set={eval_set!r}, variant={variant!r}.")
else:
    classes = sorted(set(detail["actual"]) | set(detail["predicted"]))
    fig, axes = plt.subplots(1, len(versions), figsize=(5 * len(versions), 4.5), squeeze=False)
    for ax, version in zip(axes[0], versions):
        rows = detail[detail["model_version"] == version]
        matrix = pd.crosstab(rows["actual"], rows["predicted"]).reindex(index=classes, columns=classes, fill_value=0)
        ax.imshow(matrix.values, cmap="Blues", vmin=0)
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
               xlabel="Predicted", ylabel="Actual", title=f"v{version} (n={len(rows)})")
        ax.tick_params(axis="x", rotation=45)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(matrix.values[i, j]), ha="center", va="center")
    fig.suptitle(f"{eval_set} / {variant}")
    show(fig)
    display(detail[~detail["correct"]].sort_values(["model_version", "item_id"]))

# COMMAND ----------
# MAGIC %md ## (f) Reliability diagram (5 equal-width confidence bins)

# COMMAND ----------
if not detail.empty:
    edges = np.linspace(0, 1, 6)
    fig, axes = plt.subplots(1, len(versions), figsize=(4.5 * len(versions), 4), squeeze=False)
    for ax, version in zip(axes[0], versions):
        rows = detail[detail["model_version"] == version]
        conf = rows["confidence"].to_numpy()
        hit = rows["correct"].astype(float).to_numpy()
        bins = np.minimum((conf * 5).astype(int), 4)  # same binning as evaluation/eval_metrics.py
        ece = 0.0
        for b in range(5):
            mask = bins == b
            if mask.any():
                ece += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
                ax.bar(edges[b], hit[mask].mean(), width=0.2, align="edge", alpha=0.7, edgecolor="black")
                ax.text(edges[b] + 0.1, hit[mask].mean() + 0.02, f"n={mask.sum()}", ha="center", fontsize=7)
        stored = metrics[(metrics["model_version"] == version) & (metrics["eval_set"] == eval_set)
                         & (metrics["variant"] == variant) & (metrics["metric"] == "ece")]["value"]
        stored_text = f", stored {stored.iloc[0]:.3f}" if len(stored) else ""
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1.1), xlabel="Confidence", ylabel="Accuracy",
               title=f"v{version}: ECE {ece:.3f}{stored_text}")
    show(fig)

# COMMAND ----------
# MAGIC %md ## (g) Per-class F1 for the selected eval set / variant

# COMMAND ----------
f1 = metrics[(metrics["eval_set"] == eval_set) & (metrics["variant"] == variant) & metrics["metric"].str.startswith("f1_")]
if f1.empty:
    print("No per-class F1 metrics for this selection.")
else:
    heat = f1.assign(cls=f1["metric"].str[3:]).pivot_table(index="model_version", columns="cls", values="value")
    heat = heat.reindex([v for v in versions if v in heat.index])
    fig, ax = plt.subplots(figsize=(1.4 * heat.shape[1] + 2, 0.7 * heat.shape[0] + 1.5))
    ax.imshow(heat.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(heat.shape[1]), yticks=range(heat.shape[0]), xticklabels=heat.columns,
           yticklabels=[f"v{v}" for v in heat.index], title=f"Per-class F1: {eval_set} / {variant}")
    ax.tick_params(axis="x", rotation=45)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.values[i, j]
            ax.text(j, i, "—" if np.isnan(value) else f"{value:.2f}", ha="center", va="center")
    show(fig)

# COMMAND ----------
# MAGIC %md ## (h) Training runs and learning curves (MLflow experiment `airplane-tail-identifier`)

# COMMAND ----------
import mlflow
from mlflow.tracking import MlflowClient

matches = [e for e in mlflow.search_experiments() if e.name.endswith("/airplane-tail-identifier")]
if len(matches) != 1:
    raise RuntimeError(f"Expected exactly one experiment ending in '/airplane-tail-identifier', found "
                       f"{len(matches)}: {[e.name for e in matches]}")
client = MlflowClient()
train_runs = client.search_runs([matches[0].experiment_id], order_by=["attributes.start_time DESC"], max_results=20)
display(pd.DataFrame([{
    "run_id": r.info.run_id, "run_name": r.info.run_name, "status": r.info.status,
    "start_time": pd.to_datetime(r.info.start_time, unit="ms"), "recipe": r.data.tags.get("recipe"),
    "dataset_version": r.data.tags.get("dataset_version"),
    "best_val_accuracy": r.data.metrics.get("best_val_accuracy"), "test_accuracy": r.data.metrics.get("test_accuracy"),
} for r in train_runs]))

v2_runs = [r for r in train_runs if r.data.tags.get("dataset_version") == "v2"]
if not v2_runs:
    print("No dataset_version=v2 runs among the newest 20.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in v2_runs:
        name = f"{r.info.run_name or r.info.run_id[:8]} ({r.data.tags.get('recipe')})"
        for ax, kind in zip(axes, ("loss", "accuracy")):
            for split, style in (("train", "--"), ("val", "-")):
                history = sorted(client.get_metric_history(r.info.run_id, f"{split}_{kind}"), key=lambda m: m.step)
                if history:
                    ax.plot([m.step for m in history], [m.value for m in history], style, label=f"{name} {split}")
    for ax, kind in zip(axes, ("loss", "accuracy")):
        ax.set(xlabel="Epoch", ylabel=kind, title=f"Train (dashed) / val (solid) {kind}")
    axes[1].legend(fontsize=7)
    show(fig)

# COMMAND ----------
# MAGIC %md ## (i) App submissions (`photo_submissions`)

# COMMAND ----------
subs = spark.sql(f"SELECT status, predicted_airline, confidence, review_label, model_version "
                 f"FROM {SCHEMA}.photo_submissions").toPandas()
if subs.empty:
    print("No app submissions yet.")
else:
    display(subs.groupby("status").size().rename("count").reset_index())
    display(subs.groupby(subs["predicted_airline"].fillna("(none)")).size().rename("count").reset_index())
    reviewed = subs[subs["review_label"].notna() & subs["predicted_airline"].notna()]
    correct = int((reviewed["review_label"] == reviewed["predicted_airline"]).sum())
    print(f"Reviewed: {len(reviewed)}; correct: {correct}"
          + (f" ({correct / len(reviewed):.1%})" if len(reviewed) else ""))
    if len(reviewed):
        display(reviewed.assign(correct=reviewed["review_label"] == reviewed["predicted_airline"])
                .groupby("model_version")["correct"].agg(["count", "mean"]).reset_index())
    conf = subs["confidence"].dropna()
    if len(conf):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(conf, bins=np.linspace(0, 1, 6), edgecolor="black")
        ax.set(xlabel="Confidence (uncalibrated)", ylabel="Submissions", title=f"App confidence (n={len(conf)})")
        show(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## (j) Promotion checklist (manual — nothing here is automated)
# MAGIC A challenger version replaces the `demo` version only if a reviewer confirms all of:
# MAGIC 1. No regression on `v1_test` (tail_crop) versus the current `demo` version.
# MAGIC 2. Improvement on `new_holdout` and `app_reviewed` (check n; one image is a large swing).
# MAGIC 3. No class collapse in the confusion matrices (every class still predicted; no column absorbing others).
# MAGIC 4. Lower `share_pred_southwest` at larger context scales in the context-sensitivity plot.
# MAGIC 5. The registered version's `reload_test` tag records a pass.
# MAGIC
# MAGIC Then:
# MAGIC 1. Move the registry alias `demo` to the challenger version.
# MAGIC 2. Update `entity_version` in `serving/endpoint.json`.
# MAGIC 3. Redeploy the serving endpoint.
# MAGIC 4. Run the serving smoke test and confirm it reports the new version.
