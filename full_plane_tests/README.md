# External full-plane stress check

Ten external Wikimedia Commons photographs: **two each** for `alaska`, `american`, `delta`, `southwest`, and `united`, in that class order. Selected on visual quality, complete aircraft framing, side/three-quarter view, and identifiable airline tail **before inference**, without model access, prediction-based selection, training, or retraining. This is a small convenience-sampled stress check, not a representative benchmark or a statistically reliable estimate of generalization.

## Files and paired evaluation

- `images/<class>/<id>.jpg`: ten full photographs, preserving source framing. Originals are at least 1600 pixels wide. Saved RGB JPEGs have maximum dimension 1920, Pillow quality 90, and no inherited EXIF, ICC, XMP, or comments.
- `manifest.json`: class order; portable relative paths; normalized manual `bbox = [left, top, right, bottom]`; attribution and original/download URLs; original and downloaded resolutions; UTC download time; saved-file and downloaded-source SHA256; Commons original SHA1; fingerprints and duplicate-screen report. `source_sha256` hashes the actual downloaded bytes, which may be a Commons derivative; `commons_sha1` identifies the original separately.
- `SOURCES.md`: per-photo public creator attribution, license, linked original source page, dimensions, and bbox. Each photo/derivative is under its stated CC BY-SA license, not a blanket repository license. Changes: resizing, metadata removal, JPEG re-encoding, and manual cropping in review sheets. Preserve attribution, license links, change notice, and applicable share-alike terms when redistributing.
- `review/full_planes.jpg` and `review/tail_crops.jpg`: visually reviewed contact sheets, respectively showing boxes and the resulting tail crops. Rows follow class order, two images per row. These sheets inherit the constituent photographs' respective licenses; see `SOURCES.md` for each tile.
- `selection.json`: frozen choices and manual annotations. `candidates.json`: acquisition metadata audit (includes rejected candidates; these are not evaluation records).

For paired evaluation, load a manifest image as oriented RGB, evaluate the entire image, then crop the same image using the manual bbox and evaluate that crop with the same preprocessing/class order. Pixel edges follow `training.prepare_data.crop_tail`: floor left/top, ceil right/bottom. Do not resize first and reuse unscaled pixel coordinates. The crop is a manual vertical-tail region with some surrounding context, not a pixel segmentation or the entire horizontal empennage. **This comparison is not an aircraft/tail detector evaluation.** Measured results: [RESULTS.md](RESULTS.md), with all predictions in `results.json`.

## Source screening and limitations

Reviewed all five existing `airline_tails/*/SOURCES.md` files before acquisition. Selected source URLs/titles differ from those documented; all ten selected aircraft registrations are distinct and differ from registrations explicitly documented there. Five pre-existing Southwest photographs lack recorded attribution, and some other existing titles omit registrations; complete aircraft/session independence cannot be established from these records alone.

Imported the existing `training.prepare_data` fingerprint helpers without writing bytecode outside this folder. Screened the ten full images and their manual crops against all 50 local source images and their annotated crops, and against each other, using decoded RGB hashes and combined dHash <= 4 / pHash <= 8 thresholds (all full/crop combinations). Compared downloaded-source SHA256 against existing source bytes as well. **No candidates were flagged.** This is a heuristic check; remote originals of existing tail-only photographs were not re-downloaded. **No strict independence from ImageNet or other model-pretraining data is assured.**

All ten selected aircraft are Boeing 737s; the set has photographer, location, aircraft-family, and livery bias. United includes current blue and earlier gold globe tails. Alaska includes a special fuselage with the familiar face tail. One American tail is partially crossed by a winglet, and Southwest crops retain nearby wing/winglet context; these tails remain identifiable in the reviewed sheets. Tail boxes were visually reviewed, with the first American box expanded at the top to retain the fin tip.

Commons original downloads were throttled for some files. Acquisition falls back only to Commons' standard 1920px derivative when needed; the manifest explicitly records which bytes were obtained, their URL, dimensions and hashes. Original dimensions and Commons original SHA1 come from the Commons imageinfo API. Hashes do not imply that unavailable original bytes were downloaded.

## Acquisition and verification

Requires Python, Pillow and requests. From the repository root:

```text
python full_plane_tests/download_photos.py validate
```

For reacquisition using the frozen metadata/choices, run `download`, then `finalize`, then `validate`. `finalize` regenerates attribution, duplicate screening and contact sheets and removes temporary candidate preview images. `discover` reruns Commons search and replaces candidate metadata; search rankings may change, so it is not required to reproduce the frozen selection. All authored/generated outputs stay in this folder; the existing data and helper module are read-only inputs. No developer identity, machine-specific paths, workspace hostnames, credentials, or model access are needed.
