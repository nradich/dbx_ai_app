"""Prepare reviewed, group-isolated airline tail crops using only Pillow/std lib.

Run from the repository root: python training/prepare_data.py
Manifest paths are relative to the output (path) or --source (source_path).
SHA256 hashes describe encoded files; decoded RGB hashes also detect re-encoding.
Perceptual candidates require both dHash <= 4/64 and pHash <= 8/63, on
either the oriented source or crop. Candidates are conservatively co-split;
they are not assertions that two photographs are identical.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import shutil

from PIL import Image, ImageDraw, ImageOps

CLASSES = ("alaska", "american", "delta", "southwest", "united")
SPLITS = ("train", "val", "test")
ANNOTATIONS = Path(__file__).resolve().parents[1] / "dataset/tail_annotations.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def crop_tail(image, bbox):
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in bbox)):
        raise ValueError("bbox must contain four finite numbers")
    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"bad normalized bbox: {bbox}")
    w, h = image.size
    box = (math.floor(left * w), math.floor(top * h),
           math.ceil(right * w), math.ceil(bottom * h))
    if min(box[2] - box[0], box[3] - box[1]) < 2:
        raise ValueError("bbox produces a degenerate crop")
    # A fresh image discards EXIF, ICC and all inherited source metadata.
    cropped = image.convert("RGB").crop(box)
    return Image.frombytes("RGB", cropped.size, cropped.tobytes())


def fingerprints(image):
    gray = image.convert("L")
    small = list(gray.resize((9, 8), Image.Resampling.LANCZOS).tobytes())
    dhash = sum((small[y * 9 + x] > small[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    pixels = list(gray.resize((32, 32), Image.Resampling.LANCZOS).tobytes())
    cosines = [[math.cos((2 * x + 1) * u * math.pi / 64)
                for x in range(32)] for u in range(8)]
    rows = [[sum(pixels[y * 32 + x] * cosines[u][x] for x in range(32))
             for u in range(8)] for y in range(32)]
    coeffs = [sum(rows[y][u] * cosines[v][y] for y in range(32))
              for v in range(8) for u in range(8) if (u, v) != (0, 0)]
    median = sorted(coeffs)[len(coeffs) // 2]
    phash = sum((v > median) << i for i, v in enumerate(coeffs))
    pixel_hash = sha256(f"{image.width}x{image.height}:".encode() + image.tobytes())
    return {"dhash": dhash, "phash": phash, "pixels": pixel_hash}


def duplicate_pairs(records):
    findings = []
    for a, b in itertools.combinations(records, 2):
        reasons = []
        if a["source_sha256"] == b["source_sha256"]:
            reasons.append("source_bytes_exact")
        if a["sha256"] == b["sha256"]:
            reasons.append("crop_bytes_exact")
        for view in ("source", "crop"):
            fa, fb = a["_fingerprints"][view], b["_fingerprints"][view]
            d = (fa["dhash"] ^ fb["dhash"]).bit_count()
            p = (fa["phash"] ^ fb["phash"]).bit_count()
            if fa["pixels"] == fb["pixels"]:
                reasons.append(f"{view}_pixels_exact")
            elif d <= 4 and p <= 8:
                reasons.append(f"{view}_perceptual(dhash={d},phash={p})")
        if reasons:
            findings.append({"ids": [a["id"], b["id"]], "reasons": reasons})
    return findings


def merge_duplicate_groups(records, findings):
    parents = {r["group"]: r["group"] for r in records}

    def root(g):
        while parents[g] != g:
            g = parents[g]
        return g

    by_id = {r["id"]: r for r in records}
    for pair in findings:
        a, b = (by_id[i] for i in pair["ids"])
        if a["airline"] != b["airline"]:
            raise ValueError(f"duplicate candidate with conflicting labels: {pair}")
        ga, gb = sorted((root(a["group"]), root(b["group"])))
        parents[gb] = ga
    for r in records:
        r["group"] = root(r["group"])


def assign_splits(records, classes=CLASSES, targets=(6, 2, 2)):
    """Deterministic whole-group allocation; exact targets whenever feasible.

    Solve globally, including session groups spanning multiple classes. If no
    exact solution exists, deterministic least-overflow allocation is reported
    explicitly in the manifest; groups are never broken to satisfy counts.
    """
    groups = defaultdict(list)
    for r in records:
        groups[r["group"]].append(r)
    ordered = sorted(groups, key=lambda g: (-len(groups[g]), g))
    vectors = [tuple(sum(r["airline"] == c for r in groups[g]) for c in classes)
               for g in ordered]
    initial = tuple(t for _ in classes for t in targets)
    failed = set()

    def search(i, remaining):
        if i == len(ordered):
            return [] if not any(remaining) else None
        key = (i, remaining)
        if key in failed:
            return None
        for s in range(3):
            updated = list(remaining)
            for c, n in enumerate(vectors[i]):
                updated[c * 3 + s] -= n
            if min(updated) < 0:
                continue
            result = search(i + 1, tuple(updated))
            if result is not None:
                return [s] + result
        failed.add(key)
        return None

    allocation = search(0, initial)
    exact = allocation is not None
    if not exact:
        allocation, used = [], [[0] * 3 for _ in classes]
        for vector in vectors:
            s = min(range(3), key=lambda s: (
                sum(max(0, used[c][s] + n - targets[s]) ** 2
                    for c, n in enumerate(vector)),
                sum((used[c][s] + n) / max(1, targets[s])
                    for c, n in enumerate(vector) if n), s))
            allocation.append(s)
            for c, n in enumerate(vector):
                used[c][s] += n
    for group, split in zip(ordered, allocation):
        for r in groups[group]:
            r["split"] = SPLITS[split]
    return exact


def validate_leakage(records, findings):
    by_id = {r["id"]: r for r in records}
    for field in ("group", "sha256", "source_sha256"):
        seen = {}
        for r in records:
            if r[field] in seen and seen[r[field]] != r["split"]:
                raise ValueError(f"cross-split {field}: {r['id']}")
            seen[r[field]] = r["split"]
    for pair in findings:
        if len({by_id[i]["split"] for i in pair["ids"]}) != 1:
            raise ValueError(f"cross-split duplicate: {pair}")


def review_sheet(records, crops, destination):
    sheet = Image.new("RGB", (1000, 4 * 270), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(records):
        x, y = i % 3 * 333, i // 3 * 270
        image = crops[r["id"]].copy()
        image.thumbnail((323, 230))
        sheet.paste(image, (x, y + 35))
        draw.text((x + 4, y + 4), f"{r['id']} | {r['split']}", fill="black")
    sheet.save(destination, quality=90)


def prepare(source, output, annotations=ANNOTATIONS):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from immutable source")
    spec = json.loads(Path(annotations).read_text(encoding="utf-8"))
    entries = spec["records"]
    expected = {r["source_path"] for r in entries}
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*")
              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}}
    if len(entries) != 50 or len(expected) != 50 or actual != expected:
        raise ValueError(f"expected exactly 50 annotated images; missing={expected-actual}, unexpected={actual-expected}")
    if Counter(PurePosixPath(p).parts[0] for p in expected) != Counter({c: 10 for c in CLASSES}):
        raise ValueError("expected exactly 10 source images per class")
    for c in CLASSES:
        if not (source / c / "SOURCES.md").is_file():
            raise ValueError(f"missing attribution: {c}/SOURCES.md")
    records, crops, encoded = [], {}, {}
    for entry in sorted(entries, key=lambda e: e["source_path"]):
        relative = PurePosixPath(entry["source_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("source paths must be portable relative paths")
        path = source.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(source):
            raise ValueError("source symlink escapes source root")
        airline, image_id = relative.parts[0], relative.stem
        if not isinstance(entry["group"], str) or not entry["group"].strip():
            raise ValueError("every annotation needs a nonempty group")
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        crop = crop_tail(image, entry["bbox"])
        from io import BytesIO
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        encoded[image_id] = buffer.getvalue()
        crops[image_id] = crop
        records.append({"id": image_id, "airline": airline, "label": CLASSES.index(airline),
                        "source_path": relative.as_posix(), "bbox": entry["bbox"],
                        "group": entry["group"], "sha256": sha256(encoded[image_id]),
                        "source_sha256": sha256(path.read_bytes()),
                        "_fingerprints": {"source": fingerprints(image), "crop": fingerprints(crop)}})
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("image IDs must be unique")
    findings = duplicate_pairs(records)
    merge_duplicate_groups(records, findings)
    exact = assign_splits(records)
    validate_leakage(records, findings)
    for r in records:
        r["path"] = f"{r['split']}/{r['airline']}/{r['id']}.png"
        del r["_fingerprints"]
    counts = {c: {s: sum(r["airline"] == c and r["split"] == s for r in records)
                  for s in SPLITS} for c in CLASSES}
    manifest = {"version": 1, "classes": list(CLASSES), "records": records,
                "counts": counts, "split_policy": {"targets_per_class": dict(zip(SPLITS, (6, 2, 2))),
                "exact_targets_met": exact, "note": "Whole source/session and duplicate groups preserved."},
                "duplicate_checks": {"method": __doc__.strip(), "findings": findings},
                "review_notes": spec.get("review_notes", []),
                "group_reasons": spec.get("group_reasons", {}),
                "attribution": [f"sources/{c}/SOURCES.md" for c in CLASSES]}
    if output.exists() and any(output.iterdir()):
        old_manifest = output / "manifest.json"
        if not old_manifest.exists() or json.loads(old_manifest.read_text(encoding="utf-8")) != manifest:
            raise ValueError("nonempty output differs from this preparation; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    for r in records:
        destination = output / r["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded[r["id"]])
    for c in CLASSES:
        directory = output / "sources" / c
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / c / "SOURCES.md", directory / "SOURCES.md")
        directory = output / "review" / c
        directory.mkdir(parents=True, exist_ok=True)
        review_sheet([r for r in records if r["airline"] == c], crops, directory / "review.jpg")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "exact_targets_met": exact, "duplicates": findings}, indent=2))
    return manifest


# ---------------------------------------------------------------------------
# Dataset v2 (contract: docs/model_performance.md, section "Dataset v2").
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_SOURCE_ROOT = "airline_tails"
V2_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
KINDS = ("tail-photo", "tail-crop", "full-plane")
CONTEXT_SCALE = 2.5
V2_DUPLICATE_METHOD = (
    "training.prepare_data fingerprints over all v2 records: exact source and crop SHA256, "
    "decoded RGB pixel hashes, and perceptual candidates requiring both dHash <= 4/64 and "
    "pHash <= 8/63 on either the oriented source or the tail crop. Splits are explicit: a "
    "candidate pair crossing splits (or with conflicting labels) is an error; pairs within one "
    "split are listed here. Heuristic screen, not proof of independent capture.")


def _png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _clean_rgb(image):
    """Metadata-free RGB copy (same approach as crop_tail)."""
    image = image.convert("RGB")
    return Image.frombytes("RGB", image.size, image.tobytes())


def _portable_relative(value, what):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{what} must be a portable relative path: {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ValueError(f"{what} must be a portable relative path: {value!r}")
    return relative


def _resolve_inside(base, relative, what):
    path = base.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"{what} escapes {base}: {relative}")
    return path


def context_geometry(size, bbox, scale=CONTEXT_SCALE):
    """Pixel tail box, pixel context box and the tail box normalized in the context.

    The normalized tail box is expanded ``scale`` times about its centre in width and
    height, clamped to the image, and converted to pixels with crop_tail's floor/ceil
    rounding. ``context_bbox`` is rounded to 6 decimals; when plain rounding would not
    reproduce the exact tail pixels under floor/ceil, the nearest 1e-6 neighbour that
    does is used, so crop_tail(context, context_bbox) == crop_tail(source, bbox).
    """
    w, h = size
    left, top, right, bottom = bbox
    tail = (math.floor(left * w), math.floor(top * h), math.ceil(right * w), math.ceil(bottom * h))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_w, half_h = (right - left) * scale / 2, (bottom - top) * scale / 2
    context = (min(tail[0], math.floor(max(0.0, cx - half_w) * w)),
               min(tail[1], math.floor(max(0.0, cy - half_h) * h)),
               max(tail[2], math.ceil(min(1.0, cx + half_w) * w)),
               max(tail[3], math.ceil(min(1.0, cy + half_h) * h)))
    cw, ch = context[2] - context[0], context[3] - context[1]
    targets = ((tail[0] - context[0], cw, math.floor), (tail[1] - context[1], ch, math.floor),
               (tail[2] - context[0], cw, math.ceil), (tail[3] - context[1], ch, math.ceil))
    normalized = []
    for pixel, extent, rounding in targets:
        base = round(pixel / extent, 6)
        for candidate in (base, round(base + 1e-6, 6), round(base - 1e-6, 6)):
            if 0 <= candidate <= 1 and rounding(candidate * extent) == pixel:
                normalized.append(candidate)
                break
        else:
            raise ValueError(f"cannot express tail box in context coordinates: {bbox}")
    return tail, context, normalized


def _load_v2_spec(annotations_path, repo_root):
    spec = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("version") != 2:
        raise ValueError(f"annotations version must be 2, got {spec.get('version') if isinstance(spec, dict) else spec!r}")
    roots = spec.get("source_roots")
    if (not isinstance(roots, list) or not roots or len(set(roots)) != len(roots)):
        raise ValueError("source_roots must be a nonempty list of unique paths")
    for root in roots:
        relative = _portable_relative(root, "source root")
        if len(relative.parts) != 1:
            raise ValueError(f"source root must be a top-level repository folder: {root}")
        if not _resolve_inside(repo_root, relative, "source root").is_dir():
            raise ValueError(f"missing source root: {root}")
    if not isinstance(spec.get("records"), list) or not spec["records"]:
        raise ValueError("annotations need a nonempty records list")
    return spec


def _validate_v2_entries(spec, repo_root):
    """Static validation (no image decoding); returns entries sorted by (airline, id)."""
    roots = spec["source_roots"]
    entries, seen_paths, ids = [], set(), {}
    for index, entry in enumerate(spec["records"]):
        where = f"record {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be an object")
        relative = _portable_relative(entry.get("source_path"), f"{where} source_path")
        where = f"record {entry['source_path']}"
        if relative.parts[0] not in roots:
            raise ValueError(f"{where}: source_path is not under a source root {roots}")
        if len(relative.parts) != 3:
            raise ValueError(f"{where}: expected <root>/<airline>/<file>")
        if relative.suffix.lower() not in V2_IMAGE_SUFFIXES:
            raise ValueError(f"{where}: unsupported image type {relative.suffix}")
        airline = entry.get("airline")
        if airline not in CLASSES:
            raise ValueError(f"{where}: unknown airline {airline!r}; expected one of {CLASSES}")
        if relative.parts[1] != airline:
            raise ValueError(f"{where}: airline {airline!r} does not match folder {relative.parts[1]!r}")
        if entry.get("split") not in SPLITS:
            raise ValueError(f"{where}: split must be one of {SPLITS}, got {entry.get('split')!r}")
        if entry.get("kind") not in KINDS:
            raise ValueError(f"{where}: kind must be one of {KINDS}, got {entry.get('kind')!r}")
        for field in ("group", "box_source"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"{where}: {field} must be a nonempty string")
        if entry["source_path"] in seen_paths:
            raise ValueError(f"{where}: annotated more than once")
        seen_paths.add(entry["source_path"])
        image_id = relative.stem
        if image_id in ids:
            raise ValueError(f"duplicate image ID {image_id!r}: {ids[image_id]} and {entry['source_path']}")
        ids[image_id] = entry["source_path"]
        _resolve_inside(repo_root, relative, where)
        entries.append(dict(entry, id=image_id, source_root=relative.parts[0]))
    actual = set()
    for root in roots:
        for path in (repo_root / root).rglob("*"):
            if path.is_file() and path.suffix.lower() in V2_IMAGE_SUFFIXES:
                actual.add(path.relative_to(repo_root).as_posix())
    missing, unannotated = sorted(seen_paths - actual), sorted(actual - seen_paths)
    if missing or unannotated:
        raise ValueError(f"annotated files must equal image files under the source roots; "
                         f"missing={missing}, unannotated={unannotated}")
    splits_by_group = defaultdict(set)
    for e in entries:
        splits_by_group[e["group"]].add(e["split"])
    spanning = {g: sorted(s) for g, s in sorted(splits_by_group.items()) if len(s) > 1}
    if spanning:
        raise ValueError(f"groups span splits: {spanning}")
    for root, airline in sorted({(e["source_root"], e["airline"]) for e in entries}):
        if not (repo_root / root / airline / "SOURCES.md").is_file():
            raise ValueError(f"missing attribution: {root}/{airline}/SOURCES.md")
    parent = spec.get("parent") or {}
    if parent.get("annotations"):
        parent_path = _resolve_inside(
            repo_root, _portable_relative(parent["annotations"], "parent annotations"), "parent annotations")
        by_path = {e["source_path"]: e for e in entries}
        for old in json.loads(parent_path.read_text(encoding="utf-8"))["records"]:
            new_path = f"{V1_SOURCE_ROOT}/{old['source_path']}"
            if new_path not in by_path:
                raise ValueError(f"v1 record missing from v2: {new_path}")
            if by_path[new_path]["bbox"] != old["bbox"]:
                raise ValueError(f"v1 bbox changed for {new_path}: {old['bbox']} -> {by_path[new_path]['bbox']}")
    return sorted(entries, key=lambda e: (e["airline"], e["id"]))


def _process_v2_entry(entry, repo_root):
    """Decode one annotated source; return the manifest record and encoded outputs."""
    path = repo_root.joinpath(*PurePosixPath(entry["source_path"]).parts)
    source_bytes = path.read_bytes()
    with Image.open(io.BytesIO(source_bytes)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    try:
        crop = crop_tail(image, entry["bbox"])
    except ValueError as exc:
        raise ValueError(f"record {entry['source_path']}: {exc}") from None
    tail, context_box, context_bbox = context_geometry(image.size, entry["bbox"])
    context = _clean_rgb(image.crop(context_box))
    split, airline, image_id = entry["split"], entry["airline"], entry["id"]
    outputs = {"path": f"{split}/{airline}/{image_id}.png",
               "context_path": f"context/{split}/{airline}/{image_id}.png",
               "original_path": (f"originals/{split}/{airline}/{image_id}.png"
                                 if entry["kind"] == "full-plane" else None)}
    encoded = {outputs["path"]: _png_bytes(crop), outputs["context_path"]: _png_bytes(context)}
    if outputs["original_path"]:
        encoded[outputs["original_path"]] = _png_bytes(_clean_rgb(image))
        tile = image.copy()
        tile.thumbnail((323, 230))
        sx, sy = tile.width / image.width, tile.height / image.height
        ImageDraw.Draw(tile).rectangle((tail[0] * sx, tail[1] * sy, tail[2] * sx - 1, tail[3] * sy - 1),
                                       outline=(255, 0, 0), width=3)
    else:
        tile = crop.copy()
        tile.thumbnail((323, 230))
    record = {"id": image_id, "airline": airline, "label": CLASSES.index(airline),
              "source_path": entry["source_path"], "bbox": entry["bbox"], "group": entry["group"],
              "sha256": sha256(encoded[outputs["path"]]), "source_sha256": sha256(source_bytes),
              "split": split, "path": outputs["path"], "source_root": entry["source_root"],
              "kind": entry["kind"], "box_source": entry["box_source"],
              "context_path": outputs["context_path"], "context_bbox": context_bbox,
              "context_sha256": sha256(encoded[outputs["context_path"]]),
              "original_path": outputs["original_path"],
              "original_sha256": (sha256(encoded[outputs["original_path"]])
                                  if outputs["original_path"] else None),
              "_fingerprints": {"source": fingerprints(image), "crop": fingerprints(crop)}}
    return record, encoded, tile


def _screen_images(manifest_relative, repo_root):
    """Fingerprint every image (and its bbox crop) listed in a screening manifest."""
    manifest_path = _resolve_inside(
        repo_root, _portable_relative(manifest_relative, "screen_against entry"), "screen manifest")
    if not manifest_path.is_file():
        raise ValueError(f"missing screen manifest: {manifest_relative}")
    folder = manifest_path.parent
    images = []
    for item in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]:
        relative = _portable_relative(item.get("path"), f"screen image in {manifest_relative}")
        data = _resolve_inside(folder, relative, "screen image").read_bytes()
        if "sha256" in item and sha256(data) != item["sha256"]:
            raise ValueError(f"screen image sha256 mismatch: {manifest_relative}:{item['path']}")
        with Image.open(io.BytesIO(data)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        try:
            crop = crop_tail(image, item["bbox"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"screen image {manifest_relative}:{item['path']}: bad bbox ({exc})") from None
        images.append({"id": f"{manifest_relative}:{item.get('id', relative.stem)}",
                       "airline": item.get("airline"), "source_sha256": sha256(data),
                       "sha256": sha256(_png_bytes(crop)),
                       "_fingerprints": {"source": fingerprints(image), "crop": fingerprints(crop)}})
    return images


def review_sheet_v2(records, tiles, destination):
    """Contact sheet with ceil(n / 3) rows; full-plane tiles show the tail box on the source."""
    rows = max(1, math.ceil(len(records) / 3))
    sheet = Image.new("RGB", (1000, rows * 270), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(records):
        x, y = i % 3 * 333, i // 3 * 270
        sheet.paste(tiles[r["id"]], (x, y + 35))
        draw.text((x + 4, y + 4), f"{r['id']} | {r['split']}", fill="black")
    sheet.save(destination, quality=90)


def prepare_v2(annotations_path, repo_root=REPO_ROOT, output=Path(".local/dataset_v2")):
    """Build dataset v2 from a version-2 annotation file with explicit splits."""
    annotations_path = Path(annotations_path).resolve()
    repo_root, output = Path(repo_root).resolve(), Path(output).resolve()
    spec = _load_v2_spec(annotations_path, repo_root)
    roots = spec["source_roots"]
    screens = spec.get("screen_against", [])
    if not isinstance(screens, list) or not all(isinstance(s, str) for s in screens):
        raise ValueError("screen_against must be a list of manifest paths")
    protected = [repo_root / r for r in roots] + [annotations_path.parent]
    protected += [(repo_root / s).parent for s in screens]
    if output == repo_root or output in repo_root.parents or any(
            output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError("output must be separate from immutable source, annotation and screen folders")
    entries = _validate_v2_entries(spec, repo_root)

    records, encoded, tiles = [], {}, {}
    for entry in entries:
        record, files, tile = _process_v2_entry(entry, repo_root)
        records.append(record)
        encoded.update(files)
        tiles[record["id"]] = tile
    by_id = {r["id"]: r for r in records}
    findings = duplicate_pairs(records)
    for pair in findings:
        a, b = (by_id[i] for i in pair["ids"])
        if a["airline"] != b["airline"]:
            raise ValueError(f"duplicate candidate with conflicting labels: {pair}")
        if a["split"] != b["split"]:
            raise ValueError(f"cross-split duplicate ({a['split']} vs {b['split']}): {pair}")
    validate_leakage(records, findings)

    screen_images = [image for s in screens for image in _screen_images(s, repo_root)]
    screen_findings = [finding for r in records for image in screen_images
                       for finding in duplicate_pairs([r, image])]
    if screen_findings:
        raise ValueError(f"dataset images duplicate screened evaluation images: {screen_findings}")

    for r in records:
        del r["_fingerprints"]
    counts = {c: {s: sum(r["airline"] == c and r["split"] == s for r in records)
                  for s in SPLITS} for c in CLASSES}
    sources = [(root, c) for root in roots for c in CLASSES
               if any(r["source_root"] == root and r["airline"] == c for r in records)]
    parent = spec.get("parent") or {}
    policy_note = spec.get("split_policy")
    manifest = {"version": 2, "dataset_version": "v2", "classes": list(CLASSES),
                "coordinate_system": spec.get("coordinate_system"),
                "source_roots": list(roots),
                "parent_annotations": parent.get("annotations"),
                "parent_manifest_sha256": parent.get("manifest_sha256"),
                "records": records, "counts": counts,
                "split_policy": {"method": "explicit",
                                 "note": policy_note if isinstance(policy_note, str)
                                 else "Explicit reviewed splits; a group never spans splits."},
                "context_scale": CONTEXT_SCALE,
                "duplicate_checks": {"method": V2_DUPLICATE_METHOD, "records_checked": len(records),
                                     "findings": findings},
                "screen_checks": {"against": list(screens), "images_checked": len(screen_images),
                                  "findings": screen_findings},
                "review_notes": spec.get("review_notes", []),
                "group_reasons": spec.get("group_reasons", {}),
                "attribution": [f"sources/{root}/{c}/SOURCES.md" for root, c in sources]}
    if output.exists() and any(output.iterdir()):
        old_manifest = output / "manifest.json"
        if not old_manifest.exists() or json.loads(old_manifest.read_text(encoding="utf-8")) != manifest:
            raise ValueError("nonempty output differs from this preparation; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    for relative, data in sorted(encoded.items()):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    for root, c in sources:
        directory = output / "sources" / root / c
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / root / c / "SOURCES.md", directory / "SOURCES.md")
    for c in CLASSES:
        members = [r for r in records if r["airline"] == c]
        if members:
            directory = output / "review" / c
            directory.mkdir(parents=True, exist_ok=True)
            review_sheet_v2(members, tiles, directory / "review.jpg")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "duplicates": findings,
                      "screen_images_checked": len(screen_images),
                      "screen_findings": screen_findings}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("airline_tails"))
    parser.add_argument("--output", type=Path, default=None,
                        help="default .local/dataset (v1) or .local/dataset_v2 (v2)")
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS,
                        help="v1 or version-2 annotation file (version 2 selects dataset v2)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help="base for v2 source_roots, parent and screen paths")
    args = parser.parse_args()
    try:
        spec = json.loads(args.annotations.read_text(encoding="utf-8"))
        version = spec.get("version") if isinstance(spec, dict) else None
        if version == 2:
            prepare_v2(args.annotations, args.repo_root, args.output or Path(".local/dataset_v2"))
        elif version == 1:
            prepare(args.source, args.output or Path(".local/dataset"), args.annotations)
        else:
            raise ValueError(f"unsupported annotations version: {version!r}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
