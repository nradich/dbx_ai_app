# Airline-tail classifier

Train a five-class airline classifier: **Alaska, American, Delta, Southwest, United**.
Input must be a **tail-focused image**: there is no tail detector or unknown-airline
rejection. ResNet-18 uses a frozen ImageNet backbone and trains only its airline head.

## Automated repository checks

[GitHub Actions](ci/README.md) runs code, dataset, schedule, serving-contract and
saved-photo checks on branch pushes and pull requests, including merges to main.
These CPU-only checks run on GitHub without Databricks credentials; deployment
and model evaluation remain separate.

## Measured demo result

Version **1**, alias **`demo`**:
`dbx_ai_app.airplane_tail_id.airplane_tail_identifier`.

| Held-out metric | Result |
|---|---:|
| Random airline head accuracy | 0/10 |
| Majority-class baseline accuracy | 2/10 |
| Trained head accuracy | **10/10** |
| Trained macro F1 | **1.00** |

Thirty epochs took **25.8 seconds** on one A10; the successful AIR job took about
163 seconds end to end. Checkpoint selection used validation accuracy/loss only.
All ten test images passed isolated logged-model and registered-model CPU reload
checks. This ten-image result is illustrative, not a robust estimate of accuracy
on arbitrary airplane photos. Inputs must focus on the tail.

### New full-plane stress check

On ten newly sourced full-plane photos (two per airline), the unchanged version 1
scored **3/10** on whole photographs and **8/10** on manual tail crops of those
same photographs. This confirms that full-plane submission needs additional work.
See [photos, attribution and predictions](full_plane_tests/RESULTS.md).
The visual notebook is `notebooks/03_full_plane_evaluation.py`; reproduce scoring
with `evaluation/workload.yaml` from its own directory after uploading the new folder.

## Model Serving for the app

Serving endpoint: **`airplane-tail-identifier`**, pinned to registered version **3**
(promoted from version 1 on 2026-10-27, see below), Small CPU with scale-to-zero.
See [deployment and app integration](serving/README.md)
for the base64-image API contract, app service-principal resource, backend client,
and end-to-end serving smoke test. The app should submit the selected tail crop.

## Dataset and provenance

- Tiny dataset: **50 photographs**, 10 per class. Grouped targets are **30 train / 10
  validation / 10 test** (6/2/2 per class). Source/session and duplicate groups stay
  together; the manifest reports whether exact targets were achieved.
- Manual tail boxes live in `dataset/tail_annotations.json`. Augmentation applies
  **only to training**; validation selects the checkpoint before test evaluation.
- Original `airline_tails/` files remain immutable. Preparation records source and
  crop SHA256 hashes, checks decoded-pixel/perceptual duplicate candidates and
  preserves groups. Training verifies crop hashes and logs the manifest hash.
  Use a fresh dataset version if source data or annotations change.
- [Additional photo candidates](additional_tail_photos/README.md) are kept separate
  from the original 50-image dataset v1; dataset v2 (below) adds them with reviewed
  tail boxes and explicit splits.
- Attribution: `airline_tails/<class>/SOURCES.md`, copied to prepared
  `sources/<class>/SOURCES.md` and MLflow artifacts. Sources for existing
   `southwest_tail_1.png`–`southwest_tail_5.png` remain unknown; do not infer licenses.
- `.local/` is ignored operational output (prepared data, scratch files), not source.
  Ten test images cannot establish robust real-world accuracy; confidence scores
  are uncalibrated. No claim is made about exclusion from ImageNet pretraining.

## Run

Use PowerShell from the repository root. Run each command separately. Install and
authenticate the Databricks CLI and AIR CLI first; replace `YOUR_PROFILE` with your
chosen authenticated profile. No workspace-specific identities are required here.

### 1. Prepare and test locally

```powershell
python -m pip install pillow
python training/prepare_data.py
python -m unittest discover -s tests -p test_prepare_data.py -v
```

Preparation writes `.local/dataset/manifest.json`, split/class PNG crops, attribution and
`review/<class>/review.jpg`. It refuses differing output in a nonempty directory.

### 2. Create storage and upload

For initial setup, create catalog `dbx_ai_app` through Catalog Explorer's UI
and select **Default Storage**. In this workspace a new catalog requires that UI
flow; a bare CLI catalog-create command is insufficient. With that catalog ready,
schema and managed Volume creation use the general CLI (skip creation if present):

```powershell
databricks schemas create airplane_tail_id dbx_ai_app --profile YOUR_PROFILE
databricks volumes create dbx_ai_app airplane_tail_id training_data MANAGED --profile YOUR_PROFILE
databricks fs cp -r .local/dataset dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/dataset_v1 --profile YOUR_PROFILE
```

The uploaded root must contain `dataset_v1/manifest.json` directly. Training needs
Volume read/write, schema model-creation and MLflow experiment access.

### 3. Train with AIR

Change to the `training` directory in PowerShell, then run the dry run and actual
submission as separate commands. The snapshot root in `workload.yaml` is `.`.

```powershell
Set-Location training
air run -f workload.yaml --dry-run -p YOUR_PROFILE
air run -f workload.yaml -p YOUR_PROFILE
```

The workload requests one A10 GPU and the configured managed AI environment.
Training writes `runs/<run_id>/result.json` and `checkpoint.pt` beneath
`/Volumes/dbx_ai_app/airplane_tail_id/training_data`, then publishes a `latest.json`
pointer after successful evaluation, registration and reload verification.
MLflow logs history, baseline/trained metrics, all test predictions, data hashes,
dependencies and a Unity Catalog pyfunc model.
Exact crop inputs are preserved beneath each run's `dataset/` directory; the resolved
training package versions, Python, CUDA and cuDNN are logged as `environment.json`.

The `demo` alias on `dbx_ai_app.airplane_tail_id.airplane_tail_identifier` is set
only after **every test image** matches the evaluated checkpoint through both
logged-model and registered-model reloads: same labels and numerically consistent
confidence and all class probabilities against a CPU reference, each in a fresh Python
interpreter. A separate CPU/GPU check requires identical labels and probability agreement
within `atol=1e-3, rtol=1e-3`. Results preserve the immutable model
version URI; prefer that URI when reproducing a particular run.

### 4. Review in Databricks (CPU compute)

Import the Python files as Databricks source notebooks:

- `notebooks/01_dataset_review.py`: manifest, split counts, hashes, notes and labeled
  contact sheets for every class. `dataset_root` defaults to the uploaded dataset.
- `notebooks/02_results_and_predict.py`: baseline/trained metrics, learning curves,
  confusion matrices, every held-out prediction and its crop. `volume_root` defaults
  to the Volume root; blank `result_path` follows `latest.json`. Provide an explicit
  `result.json` path to replay an older run. Blank `image_path` skips live inference.

Saved visualization uses pandas, matplotlib and Pillow without importing MLflow
or PyTorch. **No GPU is needed for results or inference.** For optional inference,
run with the exact dependencies logged in the result/model `requirements.txt`;
install them in your chosen CPU environment and restart Python if needed. No
hardcoded notebook `%pip` environment is prescribed, and `load_model` does not
install packages. Supply `image_path=/Volumes/.../tail.jpg` to execute the separate
MLflow loading cell.

## MLflow loading contract

In an authenticated environment with matching logged dependencies, use the model
version URI from `result.json` (or `@demo` for the current verified demo version):

```python
import base64
from pathlib import Path
import mlflow
import pandas as pd

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(
    "models:/dbx_ai_app.airplane_tail_id.airplane_tail_identifier@demo"
)
inputs = pd.DataFrame({
    "image_base64": [base64.b64encode(Path("/Volumes/.../tail.jpg").read_bytes()).decode("ascii")]
})
predictions = model.predict(inputs)
print(predictions)
```

Input: pandas DataFrame with string column `image_base64` containing encoded image
bytes (no data-URL prefix). Output: one row per input with `airline` (class name),
`confidence` (winning probability) and `probabilities_json` (JSON mapping all five
classes to probabilities). This is a contract example, not fabricated prediction output.

## Example flight schedule

[Schedule setup and CSVs](schedule/README.md) define two Unity Catalog demo tables,
a synthetic weekly San Diego schedule, and a candidate lookup by classifier label
and observation time. The fixture is not a real flight schedule; a tail prediction
and time alone cannot confirm a particular flight or destination.

## Tailspot Databricks App

[The Tailspot app](app/README.md) is a Databricks App (FastAPI and the adapted
wireframe UI). A user uploads a photo, outlines the tail and enters the San Diego
sighting time. The app stores an upright, metadata-free copy of the photo and the
exact tail crop in the Unity Catalog Volume `submitted_photos`, then queries the
serving endpoint, retrying while it starts from zero. It lists possible departures
from the synthetic schedule and records each submission in the `photo_submissions`
table. Deploy it with the repository's `databricks.yml` bundle; storage is created
by `app_setup/create_storage.sql`. The [API contract](app/API.md) documents the
browser–backend interface.

## Dataset v2, retraining and model performance

Live photos exposed a background bias in version 1: a Delta tail photographed from
above against grey tarmac was predicted Southwest, and recolouring only the tarmac
flipped the answer to Delta. Version 1 had learned "grey ground behind the tail
means Southwest" from six training photos per airline. Dataset v2 and two new
training recipes address this, and every step is tracked in MLflow and Unity
Catalog. Contracts: [docs/model_performance.md](docs/model_performance.md).

- **Dataset v2** (`dataset/tail_annotations_v2.json`, 83 images: 54 train / 13
  validation / 16 test). It adds the 33 attributed photos from
  `additional_tail_photos/`, including reviewed tail boxes on 11 full-plane
  Delta/Southwest photos. All v1 images keep their v1 split, so the v1 test set
  stays comparable. Related sessions stay in one split; the Acroterion series goes
  to train by decision, so the full-plane stress set is contaminated for v2 and is
  reported with that caveat. Build: `python training/prepare_data.py --annotations
  dataset/tail_annotations_v2.json` (writes `.local/dataset_v2`, including wider
  "context" crops and upright originals of full-plane photos), then upload to
  `/Volumes/dbx_ai_app/airplane_tail_id/training_data/dataset_v2`.
- **Training** (`training/workload_v2.yaml` = v1 recipe on v2 data;
  `training/workload_v2_improved.yaml` = context jitter around the tail box, random
  erasing, stronger colour jitter, label smoothing, cosine schedule and
  fine-tuning ResNet layer4). Run with `air run -f <file> -p YOUR_PROFILE` from
  `training/`. Each run logs per-epoch train/val loss and accuracy, learning rate,
  final test metrics (per-class F1, ECE, Brier), prediction tables and figures to
  the MLflow experiment `airplane-tail-identifier`, and registers a version without
  moving any alias.
- **Comparison** (`evaluation/workload_compare.yaml`, MLflow experiment
  `airplane-tail-evaluation`) scores every version on the same sets: v2 validation,
  the v1 test crops, the 6 new held-out photos, the full-plane stress set (whole
  image and tail crop), a context-sensitivity sweep (tail box ×1.0–3.0 and whole
  image), and reviewed app submissions. It selects the best new version by
  validation only. `python evaluation/publish.py --eval-run-id <id> --profile
  YOUR_PROFILE --warehouse-id <id>` loads the results into `model_eval_runs`,
  `model_eval_metrics` and `model_eval_predictions` (`app_setup/create_eval_tables.sql`).
- **Viewing results:** the Tailspot app's **Model performance** tab (live training
  curves from MLflow, comparison matrix, context-sensitivity charts, confusion
  matrices, every prediction, app-submission stats) and
  `notebooks/04_model_performance.py`.

### Comparison of 2026-10-27 (`df462c7da8374fd99ad5ebe795457305`)

| Accuracy (n) | v1 (former `demo`) | v2: v1 recipe, data v2 | v3: improved, data v2 (now `demo`, serving) |
|---|---:|---:|---:|
| v1 test crops (10) | 10/10 | 10/10 | 9/10 |
| New held-out photos (6) | 3/6 | 4/6 | **5/6** |
| All v2 test crops (16) | 13/16 | 14/16 | 14/16 |
| Your Delta app photo (1) | ✗ Southwest | ✗ United | **✓ Delta** (31%) |
| Context ×2.0 around the box (13) | 3/13 | 8/13 | **10/13** |
| Whole full-plane photo (10)* | 3/10 | 4/10 | 5/10 |
| Share predicted Southwest, v2 test | 31% | 25% | 6% |

\* Contaminated for v2/v3 (related Acroterion photos in training). Version 3 was
selected on validation (13/13 for v2 and v3, lower validation log loss) and
registered with alias `challenger`. Caveats: the sets are tiny, so a single
image moves accuracy by 6–17 points. Version 3 misses one v1 test image
(`southwest_tail_9` → Delta) and one new Delta ground photo. Its confidence on
your photo is low (31%). On whole full-plane photos, versions 2 and 3 now lean to
United instead of Southwest. A tail crop remains required.

**Promotion (2026-10-27).** After a second app photo that version 1 got wrong (an
American tail from above on tarmac: v1 Southwest 82%, v3 American 97%), version 3
was promoted. The `demo` alias moved from version 1 to version 3, and
`serving/endpoint.json` now pins version 3. The endpoint was updated in place;
version 1 kept serving until version 3 was ready. The serving smoke test matched all
16 v2 test crops to version 3's recorded CPU reference. To roll back, set
`entity_version` back to `"1"` (served entity `airplane-tail-v1`), apply the config
and run `serving/smoke_test.py --version 1`.

## App wireframe

Open [the interactive wireframe](app_wireframe/index.html) for the original static mock
of the photo submission and possible-departure flow; the Tailspot app implements it.
The mock uses fixed sample results and does not call the serving endpoint or Unity
Catalog. See [wireframe notes](app_wireframe/README.md).
