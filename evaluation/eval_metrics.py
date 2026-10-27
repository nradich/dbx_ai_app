"""Pure-Python helpers for evaluation/compare.py and evaluation/publish.py.

No third-party imports, so everything here is unit-testable without torch, MLflow,
pandas or sklearn. Definitions (documented in docs/model_performance.md):

- accuracy: share of items whose predicted class equals the actual class.
- precision/recall/f1 per class: sklearn semantics with zero_division=0, i.e. a 0/0
  ratio is 0.0 (and F1 is 0.0 when precision + recall == 0).
- macro_f1: unweighted mean of per-class F1 over ALL model classes (sklearn
  ``f1_score(labels=classes, average="macro", zero_division=0)``). Classes absent from
  both actual and predicted contribute 0.0, so small sets that do not cover every class
  have macro_f1 < accuracy even when every prediction is correct.
- nll: mean of -ln(max(p_actual, 1e-12)).
- brier: multi-class Brier score, mean over items of sum_k (p_k - 1[k == actual])^2
  (range 0..2).
- ece: 5 equal-width bins on the winning-class confidence, [0, 0.2), [0.2, 0.4), ...,
  [0.8, 1.0]; a confidence on an edge goes to the upper bin and 1.0 to the last bin.
  ECE = sum_b (n_b / n) * |accuracy_b - mean_confidence_b|.
- mean_confidence, mean_true_prob: means of the winning probability and of p_actual.
- share_pred_<class>: share of items predicted as <class>.
Every metric is defined (never null) for a nonempty set; empty sets are not scored.
"""
import json
import math
import re
from datetime import datetime, timezone

EVAL_SETS = ("v2_val", "v1_test", "new_holdout", "v2_test", "full_plane_stress",
             "context_sensitivity", "app_reviewed")
ECE_BINS = 5
PROBABILITY_FLOOR = 1e-12
MODEL_URI = re.compile(r"^models:/([A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+)/([1-9][0-9]*)$")

# Column order of app_setup/create_eval_tables.sql (tests parse the SQL to confirm).
RUN_COLUMNS = ("eval_run_id", "created_at", "model_name", "model_version", "model_uri",
               "training_run_id", "recipe", "dataset_version", "val_accuracy", "val_loss",
               "aliases", "serving", "selected", "child_run_id")
METRIC_COLUMNS = ("eval_run_id", "model_version", "eval_set", "variant", "metric", "value", "n")
PREDICTION_COLUMNS = ("eval_run_id", "model_version", "eval_set", "variant", "item_id", "actual",
                      "predicted", "confidence", "true_prob", "probabilities_json", "correct",
                      "source")


# ---------------------------------------------------------------- metrics
def confusion_matrix(actual, predicted, classes):
    """Rows = actual class, columns = predicted class, both in ``classes`` order."""
    index = {name: i for i, name in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for a, p in zip(actual, predicted, strict=True):
        matrix[index[a]][index[p]] += 1
    return matrix


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def per_class(actual, predicted, classes):
    """{class: {precision, recall, f1, support}} with zero_division=0."""
    stats = {}
    for name in classes:
        tp = sum(a == name and p == name for a, p in zip(actual, predicted))
        predicted_n = sum(p == name for p in predicted)
        support = sum(a == name for a in actual)
        precision, recall = _ratio(tp, predicted_n), _ratio(tp, support)
        f1 = _ratio(2 * precision * recall, precision + recall)
        stats[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return stats


def ece_bin(confidence):
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence outside [0, 1]: {confidence}")
    return min(int(math.floor(confidence * ECE_BINS)), ECE_BINS - 1)


def expected_calibration_error(confidences, correct):
    bins = [[] for _ in range(ECE_BINS)]
    for confidence, hit in zip(confidences, correct, strict=True):
        bins[ece_bin(confidence)].append((confidence, bool(hit)))
    total = len(confidences)
    return sum(len(b) / total * abs(sum(h for _, h in b) / len(b) - sum(c for c, _ in b) / len(b))
               for b in bins if b)


def compute_metrics(rows, classes):
    """Metrics for one (model, eval_set, variant).

    rows: [{"actual", "predicted", "confidence", "probabilities": {class: p}}], nonempty.
    Returns (metrics: dict[str, float], confusion: list[list[int]]); metric names are the
    `metric` column values of model_eval_metrics.
    """
    if not rows:
        raise ValueError("cannot score an empty set")
    classes = list(classes)
    actual = [r["actual"] for r in rows]
    predicted = [r["predicted"] for r in rows]
    for name in actual + predicted:
        if name not in classes:
            raise ValueError(f"unknown class: {name}")
    n = len(rows)
    true_probs = [float(r["probabilities"][r["actual"]]) for r in rows]
    confidences = [float(r["confidence"]) for r in rows]
    correct = [a == p for a, p in zip(actual, predicted)]
    stats = per_class(actual, predicted, classes)
    metrics = {
        "accuracy": sum(correct) / n,
        "macro_f1": sum(s["f1"] for s in stats.values()) / len(classes),
        "nll": sum(-math.log(max(p, PROBABILITY_FLOOR)) for p in true_probs) / n,
        "brier": sum(sum((float(r["probabilities"][c]) - (c == r["actual"])) ** 2 for c in classes)
                     for r in rows) / n,
        "ece": expected_calibration_error(confidences, correct),
        "mean_confidence": sum(confidences) / n,
        "mean_true_prob": sum(true_probs) / n,
    }
    for name in classes:
        metrics[f"share_pred_{name}"] = sum(p == name for p in predicted) / n
    for name in classes:
        s = stats[name]
        metrics.update({f"f1_{name}": s["f1"], f"precision_{name}": s["precision"],
                        f"recall_{name}": s["recall"], f"support_{name}": float(s["support"])})
    return {k: float(v) for k, v in metrics.items()}, confusion_matrix(actual, predicted, classes)


# ---------------------------------------------------------------- geometry
def validate_box(box):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in box)):
        raise ValueError("box must contain four finite normalized coordinates")
    left, top, right, bottom = (float(v) for v in box)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("box must be within the image and have positive area")
    return left, top, right, bottom


def scaled_box(box, scale):
    """Scale a normalized box's width and height by ``scale`` about its centre, then clamp.

    scale == 1 returns the box unchanged so context_1.0 equals the tail crop exactly;
    other results are rounded to 12 decimals to remove float noise before pixel rounding.
    """
    left, top, right, bottom = validate_box(box)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) \
            or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    if scale == 1:
        return left, top, right, bottom
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_w, half_h = (right - left) * scale / 2, (bottom - top) * scale / 2
    return tuple(round(v, 12) for v in (max(0.0, cx - half_w), max(0.0, cy - half_h),
                                        min(1.0, cx + half_w), min(1.0, cy + half_h)))


def crop_pixels(size, box):
    """Outward-rounded pixel box, identical to app/serving_client.crop_pixels."""
    left, top, right, bottom = validate_box(box)
    width, height = size
    pixels = (math.floor(left * width), math.floor(top * height),
              math.ceil(right * width), math.ceil(bottom * height))
    if pixels[2] - pixels[0] < 2 or pixels[3] - pixels[1] < 2:
        raise ValueError("box must cover at least two pixels in each dimension")
    return pixels


def context_variant(scale):
    """'context_1.0', 'context_1.5', ... (repr keeps 1.25 as '1.25')."""
    return f"context_{float(scale)!r}"


# ---------------------------------------------------------------- registry helpers
def parse_model_uri(uri):
    match = MODEL_URI.match(uri) if isinstance(uri, str) else None
    if not match:
        raise ValueError(f"model URI must be models:/<catalog>.<schema>.<model>/<version>: {uri!r}")
    return match.group(1), match.group(2)


def selected_checkpoint(accuracy_history, loss_history):
    """(val_accuracy, val_loss) of the epoch train.py keeps: max accuracy, then min loss,
    earliest epoch on exact ties. Histories are [(step, value)]. None if unavailable."""
    losses = dict(loss_history)
    joined = sorted((step, acc, losses[step]) for step, acc in accuracy_history if step in losses)
    if not joined:
        return None
    best = None
    for step, acc, loss in joined:
        if best is None or (acc, -loss) > (best[1], -best[2]):
            best = (step, acc, loss)
    return float(best[1]), float(best[2])


def version_aliases(version, model_version_aliases=None, registered_aliases=None):
    """Sorted aliases pointing at ``version`` from ModelVersion.aliases (list of str) and/or
    RegisteredModel.aliases (dict alias->version, or objects with .alias/.version)."""
    names = set(model_version_aliases or [])
    if isinstance(registered_aliases, dict):
        names |= {a for a, v in registered_aliases.items() if str(v) == str(version)}
    else:
        for item in registered_aliases or []:
            alias, target = getattr(item, "alias", None), getattr(item, "version", None)
            if alias and str(target) == str(version):
                names.add(alias)
    return sorted(names)


def select_candidate(entries):
    """Pick the comparison's selected model version.

    entries: [{"model_version", "dataset_version", "accuracy", "nll"}] where accuracy/nll
    are this comparison's v2_val tail_crop metrics. Candidates are versions not trained on
    v1 data; best by (accuracy, -nll), exact ties keep the lowest version number.
    Returns (model_version or None, rationale dict). Validation data only; test sets are
    never consulted.
    """
    candidates = [e for e in entries if e.get("dataset_version") != "v1"]
    rule = ("Among versions whose training dataset_version != 'v1', maximise v2_val/tail_crop "
            "accuracy, then minimise v2_val/tail_crop nll, then lowest version; validation only.")
    if not candidates:
        return None, {"rule": rule, "candidates": [], "selected": None,
                      "reason": "no candidate trained on a dataset other than v1"}
    best = max(candidates, key=lambda e: (e["accuracy"], -e["nll"], -int(e["model_version"])))
    return best["model_version"], {
        "rule": rule, "selected": best["model_version"],
        "candidates": [{"model_version": e["model_version"], "dataset_version": e["dataset_version"],
                        "v2_val_accuracy": e["accuracy"], "v2_val_nll": e["nll"]} for e in candidates],
        "excluded": [e["model_version"] for e in entries if e not in candidates]}


# ---------------------------------------------------------------- table rows
def utc_timestamp(moment=None):
    """ISO-8601 UTC with microseconds, e.g. 2026-10-27T20:00:00.000000Z."""
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _optional_float(value):
    return None if value is None else float(value)


def run_row(eval_run_id, created_at, model):
    """model: dict with model_name, model_version, model_uri, training_run_id, recipe,
    dataset_version, val_accuracy, val_loss, aliases (list), serving, selected, child_run_id."""
    return {
        "eval_run_id": eval_run_id, "created_at": created_at,
        "model_name": model["model_name"], "model_version": str(model["model_version"]),
        "model_uri": model["model_uri"], "training_run_id": model.get("training_run_id"),
        "recipe": model.get("recipe"), "dataset_version": model.get("dataset_version"),
        "val_accuracy": _optional_float(model.get("val_accuracy")),
        "val_loss": _optional_float(model.get("val_loss")),
        "aliases": json.dumps(sorted(model.get("aliases") or [])),
        "serving": bool(model["serving"]), "selected": bool(model["selected"]),
        "child_run_id": model.get("child_run_id"),
    }


def metric_rows(eval_run_id, model_version, eval_set, variant, metrics, n):
    return [{"eval_run_id": eval_run_id, "model_version": str(model_version), "eval_set": eval_set,
             "variant": variant, "metric": name, "value": float(value), "n": int(n)}
            for name, value in metrics.items()]


def prediction_row(eval_run_id, model_version, eval_set, variant, item, prediction, classes):
    """item: {item_id, actual, source}; prediction: {predicted, confidence, probabilities}."""
    probabilities = {c: float(prediction["probabilities"][c]) for c in classes}
    return {
        "eval_run_id": eval_run_id, "model_version": str(model_version), "eval_set": eval_set,
        "variant": variant, "item_id": item["item_id"], "actual": item["actual"],
        "predicted": prediction["predicted"], "confidence": float(prediction["confidence"]),
        "true_prob": probabilities[item["actual"]],
        "probabilities_json": json.dumps(probabilities, separators=(",", ":")),
        "correct": item["actual"] == prediction["predicted"], "source": item.get("source"),
    }


def jsonl(rows, columns):
    """Serialize rows (exactly ``columns`` keys, in column order); NaN/inf are rejected."""
    lines = []
    for row in rows:
        if set(row) != set(columns):
            raise ValueError(f"row keys {sorted(row)} differ from columns {sorted(columns)}")
        lines.append(json.dumps({c: row[c] for c in columns}, allow_nan=False, separators=(",", ":")))
    return "".join(line + "\n" for line in lines)
