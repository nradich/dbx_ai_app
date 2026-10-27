"""Read-only model performance data for the Tailspot app (contract: docs/model_performance.md).

Sources: the evaluation tables published by evaluation/publish.py
(app_setup/create_eval_tables.sql), the training MLflow experiment and photo_submissions.
Every SQL value comes from configuration or a validated name and is passed as a bound
parameter; only the validated schema is interpolated (as backtick-quoted identifiers).
Results are cached briefly so page refreshes do not load the SQL warehouse.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import itertools
import json
import logging
import math
import os
import re
import threading
import time
from typing import Callable

import pipeline
import schedule

log = logging.getLogger("tailspot")

EXPERIMENT_ENV = "TAILSPOT_TRAINING_EXPERIMENT_ID"
# Kept below the browser's 60 s request timeout so a cold warehouse yields a quick 503.
SQL_WAIT = "20s"
SQL_MAX_WAIT_S = 25
SUMMARY_TTL_S = 30
PREDICTIONS_TTL_S = 30
TRAINING_RUNS_TTL_S = 10
HISTORY_RUNNING_TTL_S = 10
HISTORY_FINISHED_TTL_S = 300
SUBMISSIONS_TTL_S = 15
MAX_TRAINING_RUNS = 20
RECENT_SUBMISSIONS = 10
CACHE_ENTRIES = 256
HISTORY_KEYS = ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "lr")
FINISHED_STATUSES = frozenset({"FINISHED", "FAILED", "KILLED"})
# Confidence buckets of width 0.2; a value on an edge goes to the upper bucket, 1.0 to the last.
BUCKET_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
CONFIDENCE_BUCKET_SQL = ("CASE WHEN confidence IS NULL THEN NULL "
                         + " ".join(f"WHEN confidence < {edge} THEN {index}"
                                    for index, edge in enumerate(BUCKET_EDGES[1:-1]))
                         + f" ELSE {len(BUCKET_EDGES) - 2} END")
# Reviewed rows only; a submission without a prediction (failed) can never be 'correct'.
REVIEW_SQL = ("CASE WHEN review_label IS NULL THEN NULL "
              "WHEN predicted_airline IS NOT NULL AND review_label = predicted_airline "
              "THEN 'correct' ELSE 'incorrect' END")

_RUN_ID = re.compile(r"[0-9a-f]{32}")
_EXPERIMENT_ID = re.compile(r"[0-9A-Za-z_-]{1,64}")
_CONTEXT_VARIANT = re.compile(r"context_(?:full|[0-9]{1,2}\.[0-9]{1,6})")

# Evaluation sets in display order (evaluation/compare.py builds them; see
# docs/model_performance.md). Only sets with metrics in the latest comparison are shown.
EVAL_SETS = (
    {"eval_set": "v2_val", "label": "v2 validation crops", "variants": ("tail_crop",),
     "description": "Validation tail crops of dataset v2. Training used them to pick each run's "
                    "checkpoint and this comparison used them to select a model version, so these "
                    "scores are optimistic: they are not an independent test."},
    {"eval_set": "v1_test", "label": "v1 test crops", "variants": ("tail_crop",),
     "description": "The original 10 held-out tail crops from the dataset v1 test split. Dataset v2 "
                    "kept them in test, so every model version can be compared on them; no version "
                    "trained on them."},
    {"eval_set": "new_holdout", "label": "New held-out photos", "variants": ("tail_crop",),
     "description": "6 new tail photos added to the dataset v2 test split. No model version used them "
                    "for training, checkpoint choice or model selection. Two Delta photos were taken "
                    "on the ground to probe reliance on sky backgrounds."},
    {"eval_set": "v2_test", "label": "All v2 test crops", "variants": ("tail_crop",),
     "description": "Every dataset v2 test tail crop: the 10 v1 test crops plus the 6 new held-out "
                    "photos. Not used for training or selection by any version."},
    {"eval_set": "full_plane_stress", "label": "Full-plane stress test",
     "variants": ("whole_image", "tail_crop"),
     "description": "10 photos of whole aircraft. 'whole_image' sends the full photo, which is outside "
                    "the model's intended input (a tail crop); 'tail_crop' sends a reviewed tail box. "
                    "Caveat: six are Acroterion photos from BWI, and dataset v2 trained on related "
                    "Acroterion BWI photos, so v2-trained versions may score optimistically here."},
    {"eval_set": "context_sensitivity", "label": "Context sensitivity",
     "variants": ("context_1.0", "context_1.5", "context_2.0", "context_3.0", "context_full"),
     "description": "How predictions change as more of the aircraft and background is included. The "
                    "tail box is scaled about its centre by 1.0 (the exact tail box), 1.5, 2 or 3 "
                    "times its width and height, clamped to the photo; 'context_full' is the whole "
                    "photo. Sources: the 10 full-plane test photos (same caveat as the stress test), "
                    "dataset v2 test photos with a full original, and reviewed app submissions."},
    {"eval_set": "app_reviewed", "label": "Reviewed app submissions", "variants": ("tail_crop",),
     "description": "Tail crops submitted through this app whose airline a reviewer confirmed, scored "
                    "on the exact stored crop the model received. Few images and not a random "
                    "sample."},
)
_CATALOGUE = {entry["eval_set"]: entry for entry in EVAL_SETS}


class PerformanceError(Exception):
    """A failure to report with an HTTP status and a readable `detail`."""

    status_code = 400

    def __init__(self, detail: str, status_code: int | None = None):
        super().__init__(detail)
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code


class PerformanceUnavailable(PerformanceError):
    status_code = 503


class InvalidRequest(PerformanceError):
    status_code = 422


class NotFound(PerformanceError):
    status_code = 404


def experiment_id_from_env() -> str | None:
    """Training experiment ID (app.yaml valueFrom an experiment resource), or None."""
    value = (os.environ.get(EXPERIMENT_ENV) or "").strip()
    if not value:
        return None
    if not _EXPERIMENT_ID.fullmatch(value):
        log.warning("Ignoring invalid %s; training runs will not be shown", EXPERIMENT_ENV)
        return None
    return value


def known_variant(eval_set: str, variant: str) -> bool:
    entry = _CATALOGUE.get(eval_set)
    if entry is None:
        return False
    if variant in entry["variants"]:
        return True
    # compare.py accepts other context scales (context_<float repr>).
    return eval_set == "context_sensitivity" and bool(_CONTEXT_VARIANT.fullmatch(variant))


def _variant_key(eval_set: str, variant: str):
    if eval_set == "context_sensitivity":
        scale = variant.removeprefix("context_")
        return (1, 0.0) if scale == "full" else (0, float(scale))
    variants = _CATALOGUE[eval_set]["variants"]
    return (variants.index(variant), 0.0) if variant in variants else (len(variants), 0.0)


def _version_key(version) -> tuple:
    text = "" if version is None else str(version)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def _float(value) -> float | None:
    """SQL/MLflow number to a JSON-safe float (NULL and non-finite values become None)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def _bool(value) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _str(value) -> str | None:
    return None if value is None else str(value)


def _aliases(value) -> list[str]:
    try:
        aliases = json.loads(value) if value else []
    except (TypeError, ValueError):
        return []
    return [str(alias) for alias in aliases] if isinstance(aliases, list) else []


def _ms_to_iso(value) -> str | None:
    if value is None:
        return None
    return schedule.iso_z(datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc))


def _status(value) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


class _Cache:
    """Small TTL cache on a monotonic clock; values are deep-copied in and out."""

    def __init__(self, monotonic: Callable[[], float], max_entries: int = CACHE_ENTRIES):
        self._monotonic, self._max = monotonic, max_entries
        self._entries: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or self._monotonic() >= entry[0]:
                return None
            return copy.deepcopy(entry[1])

    def put(self, key, value, ttl: float) -> None:
        with self._lock:
            self._entries.pop(key, None)
            while len(self._entries) >= self._max:
                self._entries.pop(next(iter(self._entries)))
            self._entries[key] = (self._monotonic() + ttl, copy.deepcopy(value))


class PerformanceService:
    """Loads model performance data; Databricks is contacted only when an endpoint is called."""

    def __init__(self, settings: pipeline.Settings | None = None,
                 client_factory: Callable[[], object] | None = None, *,
                 training_experiment_id: str | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic):
        self.settings = settings or pipeline.Settings.from_env()
        self.training_experiment_id = training_experiment_id or None
        self._client_factory = client_factory
        self._client = None
        self._client_lock = threading.Lock()
        self._sleep, self._monotonic = sleep, monotonic
        self._cache = _Cache(monotonic)
        # Striped locks so concurrent requests for the same key run one load, with bounded memory.
        self._load_locks = tuple(threading.Lock() for _ in range(16))

    # Plumbing ----------------------------------------------------------------
    def client(self):
        with self._client_lock:
            if self._client is None:
                factory = self._client_factory or (lambda: pipeline._default_client("api"))
                self._client = factory()
            return self._client

    def _sql(self, statement: str, params: dict | None = None) -> list[dict]:
        return pipeline.execute_sql(self.client(), self.settings.warehouse_id, statement, params,
                                    wait_timeout=SQL_WAIT, max_wait_s=SQL_MAX_WAIT_S,
                                    sleep=self._sleep, monotonic=self._monotonic)

    def _cached(self, key, ttl: Callable[[dict], float], load: Callable[[], dict],
                unavailable: str) -> dict:
        value = self._cache.get(key)
        if value is not None:
            return value
        with self._load_locks[hash(key) % len(self._load_locks)]:
            value = self._cache.get(key)
            if value is not None:
                return value
            try:
                value = load()
            except PerformanceError:
                raise
            except Exception:
                log.exception("Could not load %s", key[0])
                raise PerformanceUnavailable(unavailable) from None
            self._cache.put(key, value, ttl(value))
            return copy.deepcopy(value)

    # Evaluation results ------------------------------------------------------
    def summary(self) -> dict:
        """Latest published comparison, its metrics and the live serving version."""
        return self._cached(("summary",), lambda _: SUMMARY_TTL_S, self._load_summary,
                            "Model evaluation results are temporarily unavailable.")

    def _load_summary(self) -> dict:
        runs_table = self.settings.table("model_eval_runs")
        rows = self._sql(
            "WITH latest AS (SELECT eval_run_id FROM " + runs_table + " "
            "ORDER BY created_at DESC, eval_run_id DESC LIMIT 1) "
            "SELECT r.eval_run_id, unix_timestamp(r.created_at) AS created_epoch, r.model_version, "
            "r.recipe, r.dataset_version, r.training_run_id, r.val_accuracy, r.val_loss, r.aliases, "
            "r.serving, r.selected "
            f"FROM {runs_table} r JOIN latest l ON r.eval_run_id = l.eval_run_id")
        serving = {"endpoint": self.settings.serving_endpoint,
                   "model_version": pipeline.model_metadata(
                       self.client(), self.settings.serving_endpoint, None)["model_version"]}
        if not rows:
            return {"available": False, "eval_run_id": None, "evaluated_at": None,
                    "serving": serving, "models": [], "eval_sets": [], "metrics": []}
        eval_run_id = rows[0]["eval_run_id"]
        epochs = [_int(row.get("created_epoch")) for row in rows]
        epochs = [epoch for epoch in epochs if epoch is not None]
        models = sorted(({
            "model_version": _str(row.get("model_version")), "recipe": row.get("recipe"),
            "dataset_version": row.get("dataset_version"),
            "training_run_id": row.get("training_run_id"),
            "val_accuracy": _float(row.get("val_accuracy")), "val_loss": _float(row.get("val_loss")),
            "aliases": _aliases(row.get("aliases")), "serving": bool(_bool(row.get("serving"))),
            "selected": bool(_bool(row.get("selected"))),
        } for row in rows), key=lambda model: _version_key(model["model_version"]))
        if serving["model_version"] is not None:
            # The recorded flag describes the endpoint when the comparison ran; show what serves now.
            for model in models:
                model["serving"] = model["model_version"] == serving["model_version"]
        metric_rows = self._sql(
            "SELECT model_version, eval_set, variant, metric, value, n "
            f"FROM {self.settings.table('model_eval_metrics')} WHERE eval_run_id = :eval_run_id",
            {"eval_run_id": eval_run_id})
        order = {name: index for index, name in enumerate(_CATALOGUE)}
        metrics = [{
            "model_version": _str(row.get("model_version")), "eval_set": row.get("eval_set"),
            "variant": row.get("variant"), "metric": row.get("metric"),
            "value": _float(row.get("value")), "n": _int(row.get("n")),
        } for row in metric_rows if known_variant(row.get("eval_set"), row.get("variant") or "")]
        metrics.sort(key=lambda m: (_version_key(m["model_version"]), order[m["eval_set"]],
                                    _variant_key(m["eval_set"], m["variant"]), m["metric"] or ""))
        present: dict[str, set] = {}
        for metric in metrics:
            present.setdefault(metric["eval_set"], set()).add(metric["variant"])
        eval_sets = [{"eval_set": entry["eval_set"], "label": entry["label"],
                      "description": entry["description"],
                      "variants": sorted(present[entry["eval_set"]],
                                         key=lambda v, s=entry["eval_set"]: _variant_key(s, v))}
                     for entry in EVAL_SETS if entry["eval_set"] in present]
        return {"available": True, "eval_run_id": eval_run_id,
                "evaluated_at": pipeline._epoch_to_iso(max(epochs)) if epochs else None,
                "serving": serving, "models": models, "eval_sets": eval_sets, "metrics": metrics}

    def predictions(self, eval_set: str | None, variant: str | None,
                    eval_run_id: str | None = None) -> dict:
        """Per-image predictions for one set and variant of a comparison (default: the latest)."""
        if not eval_set or eval_set not in _CATALOGUE:
            raise InvalidRequest("Unknown eval_set. Use one of: " + ", ".join(_CATALOGUE) + ".")
        if not variant or not known_variant(eval_set, variant):
            raise InvalidRequest(f"Unknown variant for {eval_set}. Use one of: "
                                 + ", ".join(_CATALOGUE[eval_set]["variants"]) + ".")
        if eval_run_id is not None and not _RUN_ID.fullmatch(eval_run_id):
            raise InvalidRequest("eval_run_id must be 32 lowercase hex characters.")
        return self._cached(("predictions", eval_set, variant, eval_run_id),
                            lambda _: PREDICTIONS_TTL_S,
                            lambda: self._load_predictions(eval_set, variant, eval_run_id),
                            "Evaluation predictions are temporarily unavailable.")

    def _load_predictions(self, eval_set: str, variant: str, eval_run_id: str | None = None) -> dict:
        if eval_run_id is not None:
            rows = self._sql(
                "SELECT model_version, item_id, actual, predicted, confidence, true_prob, correct "
                f"FROM {self.settings.table('model_eval_predictions')} "
                "WHERE eval_run_id = :eval_run_id AND eval_set = :eval_set AND variant = :variant",
                {"eval_run_id": eval_run_id, "eval_set": eval_set, "variant": variant})
            rows = [{**row, "latest_run_id": eval_run_id} for row in rows]
            if not rows:
                rows = [{"latest_run_id": eval_run_id}]
        else:
            rows = self._sql(
                "WITH latest AS (SELECT eval_run_id FROM " + self.settings.table("model_eval_runs")
                + " ORDER BY created_at DESC, eval_run_id DESC LIMIT 1) "
                "SELECT l.eval_run_id AS latest_run_id, p.model_version, p.item_id, p.actual, "
                "p.predicted, p.confidence, p.true_prob, p.correct "
                f"FROM latest l LEFT JOIN {self.settings.table('model_eval_predictions')} p "
                "ON p.eval_run_id = l.eval_run_id AND p.eval_set = :eval_set AND p.variant = :variant",
                {"eval_set": eval_set, "variant": variant})
        result = [{
            "model_version": _str(row.get("model_version")), "item_id": row.get("item_id"),
            "actual": row.get("actual"), "predicted": row.get("predicted"),
            "confidence": _float(row.get("confidence")), "true_prob": _float(row.get("true_prob")),
            "correct": _bool(row.get("correct")),
        } for row in rows if row.get("model_version") is not None]
        result.sort(key=lambda r: (_version_key(r["model_version"]), r["item_id"] or ""))
        return {"eval_run_id": rows[0].get("latest_run_id") if rows else None, "rows": result}

    # Training runs -----------------------------------------------------------
    def training_runs(self) -> dict:
        """Newest training runs (including RUNNING ones) from the training experiment."""
        if not self.training_experiment_id:
            return {"experiment_id": None, "runs": []}
        return self._cached(("training_runs",), lambda _: TRAINING_RUNS_TTL_S,
                            self._load_training_runs, "Training runs are temporarily unavailable.")

    def _load_training_runs(self) -> dict:
        found = self.client().experiments.search_runs(
            experiment_ids=[self.training_experiment_id], max_results=MAX_TRAINING_RUNS,
            order_by=["attributes.start_time DESC"])
        # The SDK iterator follows next_page_token; stop after one page's worth.
        runs = [self._run_summary(run) for run in itertools.islice(found, MAX_TRAINING_RUNS)]
        return {"experiment_id": self.training_experiment_id, "runs": runs}

    @staticmethod
    def _run_summary(run) -> dict:
        info, data = run.info, run.data
        metrics = {m.key: m for m in ((data.metrics if data else None) or []) if m.key}
        tags = {t.key: t.value for t in ((data.tags if data else None) or []) if t.key}

        def value(key):
            metric = metrics.get(key)
            return None if metric is None else _float(metric.value)

        best = value("best_val_accuracy")
        epoch = metrics.get("val_accuracy")
        status = _status(info.status)
        if best is None and status == "RUNNING":
            best = value("val_accuracy")  # provisional until training logs best_val_accuracy
        return {
            "run_id": info.run_id, "run_name": info.run_name or tags.get("mlflow.runName"),
            "status": status, "start_time": _ms_to_iso(info.start_time),
            "end_time": _ms_to_iso(info.end_time),
            "recipe": tags.get("recipe") or "v1",
            "dataset_version": tags.get("dataset_version") or "v1",
            "registered_model_version": tags.get("registered_model_version"),
            "best_val_accuracy": best,
            "test_accuracy": value("test_accuracy"), "test_macro_f1": value("test_macro_f1"),
            "epochs_completed": None if epoch is None or epoch.step is None else int(epoch.step),
        }

    def history(self, run_id: str) -> dict:
        """Per-epoch metric history of one run in the training experiment."""
        if not self.training_experiment_id or not isinstance(run_id, str) \
                or not _RUN_ID.fullmatch(run_id):
            raise NotFound("Training run not found.")
        return self._cached(
            ("history", run_id),
            lambda result: (HISTORY_FINISHED_TTL_S if result["status"] in FINISHED_STATUSES
                            else HISTORY_RUNNING_TTL_S),
            lambda: self._load_history(run_id), "Training history is temporarily unavailable.")

    def _load_history(self, run_id: str) -> dict:
        from databricks.sdk import errors

        experiments = self.client().experiments
        try:
            run = experiments.get_run(run_id).run
        except (errors.NotFound, errors.InvalidParameterValue):
            raise NotFound("Training run not found.") from None
        info = run.info if run else None
        if (info is None or str(info.experiment_id) != self.training_experiment_id
                or info.lifecycle_stage == "deleted"):
            raise NotFound("Training run not found.")
        metrics = {}
        for key in HISTORY_KEYS:
            latest: dict[int, tuple] = {}
            for metric in experiments.get_history(metric_key=key, run_id=run_id):
                step = int(metric.step or 0)
                stamp = metric.timestamp or 0
                if step not in latest or stamp >= latest[step][0]:
                    latest[step] = (stamp, _float(metric.value))
            metrics[key] = [[step, latest[step][1]] for step in sorted(latest)]
        return {"run_id": run_id, "status": _status(info.status), "metrics": metrics}

    # App submissions ---------------------------------------------------------
    def submissions(self) -> dict:
        """Aggregate statistics and the latest submissions (no storage paths)."""
        return self._cached(("submissions",), lambda _: SUBMISSIONS_TTL_S, self._load_submissions,
                            "Submission statistics are temporarily unavailable.")

    def _load_submissions(self) -> dict:
        table = self.settings.table("photo_submissions")
        groups = self._sql(
            "WITH s AS (SELECT status, predicted_airline, "
            f"{CONFIDENCE_BUCKET_SQL} AS bucket, {REVIEW_SQL} AS review "
            f"FROM {table}) "
            "SELECT 'status' AS kind, status AS label, COUNT(*) AS n FROM s GROUP BY status "
            "UNION ALL SELECT 'airline', predicted_airline, COUNT(*) FROM s "
            "WHERE predicted_airline IS NOT NULL GROUP BY predicted_airline "
            "UNION ALL SELECT 'bucket', CAST(bucket AS STRING), COUNT(*) FROM s "
            "WHERE bucket IS NOT NULL GROUP BY bucket "
            "UNION ALL SELECT 'review', review, COUNT(*) FROM s WHERE review IS NOT NULL GROUP BY review")
        recent = self._sql(
            "SELECT submission_id, unix_timestamp(submitted_at) AS submitted_epoch, "
            "predicted_airline, confidence, review_label, model_version "
            f"FROM {table} ORDER BY submitted_at DESC, submission_id DESC "
            f"LIMIT {RECENT_SUBMISSIONS}")
        counts: dict[tuple, int] = {}
        for row in groups:
            key = (row.get("kind"), row.get("label"))
            counts[key] = counts.get(key, 0) + (_int(row.get("n")) or 0)
        airlines = sorted(((label, n) for (kind, label), n in counts.items()
                           if kind == "airline" and label is not None),
                          key=lambda item: (-item[1], item[0]))
        buckets = [{"lower": lower, "upper": upper, "count": counts.get(("bucket", str(index)), 0)}
                   for index, (lower, upper) in enumerate(zip(BUCKET_EDGES, BUCKET_EDGES[1:]))]
        correct = counts.get(("review", "correct"), 0)
        return {
            "total": sum(n for (kind, _), n in counts.items() if kind == "status"),
            "succeeded": counts.get(("status", "succeeded"), 0),
            "failed": counts.get(("status", "failed"), 0),
            "reviewed": correct + counts.get(("review", "incorrect"), 0),
            "reviewed_correct": correct,
            "by_predicted_airline": [{"airline": label, "count": n} for label, n in airlines],
            "confidence_buckets": buckets,
            "recent": [{
                "submission_id": row.get("submission_id"),
                "submitted_at": pipeline._epoch_to_iso(row.get("submitted_epoch")),
                "predicted_airline": row.get("predicted_airline"),
                "confidence": _float(row.get("confidence")),
                "review_label": row.get("review_label"),
                "model_version": _str(row.get("model_version")),
            } for row in recent],
        }
