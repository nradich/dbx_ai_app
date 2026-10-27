"""Pure-Python helpers for training: context-crop geometry and calibration/per-class metrics.

No torch/numpy imports so the helpers can be unit tested without the training stack.
"""
import math

CONTEXT_SHIFT_FRACTION = 0.10
ECE_BINS = 5


def jittered_context_box(context_bbox, image_size, scale, shift_x, shift_y,
                         shift_fraction=CONTEXT_SHIFT_FRACTION):
    """Return an integer pixel crop box (left, top, right, bottom) within the context image.

    context_bbox: tail box normalized (left, top, right, bottom) within the context image.
    scale: width/height multiplier applied about the box centre.
    shift_x, shift_y: in [-1, 1]; the centre moves by shift * shift_fraction * tail box size.
    The scaled box is clamped to the image, so it may shrink at the borders.
    """
    width, height = image_size
    left, top, right, bottom = (float(v) for v in context_bbox)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"bad normalized context_bbox: {context_bbox}")
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be positive: {scale}")
    if not (-1 <= shift_x <= 1 and -1 <= shift_y <= 1):
        raise ValueError("shift values must lie in [-1, 1]")
    box_w, box_h = (right - left) * width, (bottom - top) * height
    cx = (left + right) / 2 * width + shift_x * shift_fraction * box_w
    cy = (top + bottom) / 2 * height + shift_y * shift_fraction * box_h
    half_w, half_h = box_w * scale / 2, box_h * scale / 2
    x0 = max(0, math.floor(cx - half_w))
    y0 = max(0, math.floor(cy - half_h))
    x1 = min(width, math.ceil(cx + half_w))
    y1 = min(height, math.ceil(cy + half_h))
    # Keep at least one pixel even for pathological tiny boxes near a border.
    if x1 <= x0:
        x0, x1 = (x0, x0 + 1) if x0 < width else (width - 1, width)
    if y1 <= y0:
        y0, y1 = (y0, y0 + 1) if y0 < height else (height - 1, height)
    return x0, y0, x1, y1


def sample_context_box(context_bbox, image_size, rng, scale_range):
    """Draw scale ~ U(scale_range) and centre shifts ~ U(-1, 1) from ``rng`` (random.Random)."""
    low, high = (float(v) for v in scale_range)
    if not (0 < low <= high):
        raise ValueError(f"context_scale_range must satisfy 0 < low <= high: {scale_range}")
    scale = rng.uniform(low, high)
    return jittered_context_box(context_bbox, image_size, scale, rng.uniform(-1, 1), rng.uniform(-1, 1))


def calibration_bins(probabilities, labels, bins=ECE_BINS):
    """Equal-width bins over max-probability confidence: [{lower, upper, count, accuracy, confidence}]."""
    stats = [{"lower": i / bins, "upper": (i + 1) / bins, "count": 0, "correct": 0, "confidence_sum": 0.0}
             for i in range(bins)]
    for vector, label in zip(probabilities, labels):
        index = max(range(len(vector)), key=lambda k: vector[k])
        confidence = float(vector[index])
        slot = stats[min(bins - 1, max(0, int(confidence * bins)))]
        slot["count"] += 1
        slot["correct"] += int(index == label)
        slot["confidence_sum"] += confidence
    return [{"lower": s["lower"], "upper": s["upper"], "count": s["count"],
             "accuracy": s["correct"] / s["count"] if s["count"] else None,
             "confidence": s["confidence_sum"] / s["count"] if s["count"] else None}
            for s in stats]


def expected_calibration_error(probabilities, labels, bins=ECE_BINS):
    total = len(labels)
    if not total:
        raise ValueError("ECE needs at least one prediction")
    return sum(b["count"] / total * abs(b["accuracy"] - b["confidence"])
               for b in calibration_bins(probabilities, labels, bins) if b["count"])


def brier_score(probabilities, labels):
    """Mean over images of sum_k (p_k - y_k)^2 with one-hot y."""
    if not labels:
        raise ValueError("Brier score needs at least one prediction")
    return sum(sum((float(p) - (k == label)) ** 2 for k, p in enumerate(vector))
               for vector, label in zip(probabilities, labels)) / len(labels)


def per_class_metrics(labels, predicted, classes):
    """[{class, precision, recall, f1, support}] with zero_division=0 semantics."""
    rows = []
    for k, name in enumerate(classes):
        tp = sum(1 for y, p in zip(labels, predicted) if y == k and p == k)
        fp = sum(1 for y, p in zip(labels, predicted) if y != k and p == k)
        fn = sum(1 for y, p in zip(labels, predicted) if y == k and p != k)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": tp + fn})
    return rows
