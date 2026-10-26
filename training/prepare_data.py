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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("airline_tails"))
    parser.add_argument("--output", type=Path, default=Path(".local/dataset"))
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
