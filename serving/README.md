# Airline-tail Model Serving

Endpoint: **`airplane-tail-identifier`**

- Model: `dbx_ai_app.airplane_tail_id.airplane_tail_identifier`, immutable version **3**
  (alias `demo`; promoted from version 1 on 2026-10-27).
- Small CPU workload, 100% traffic to `airplane-tail-v3`, scale-to-zero enabled.
- Tail-focused image input; no automatic localization or unknown-airline detection.
- Deployment verified: all 16 dataset v2 held-out crops returned the same labels and
  all class probabilities as version 3's recorded CPU reference (14/16 correct on
  those crops). The endpoint reached `READY` / `NOT_UPDATING`. Version 1 was verified
  the same way on its 10 held-out crops (10/10) before the switch.
- Scale-to-zero reduces idle compute but adds cold-start delay to the next request.
  Serving compute incurs charges while active. For an interactive app, show loading
  and retry states; consider disabling scale-to-zero if warm response time is required.

## Deployment

From the repository root, with an authenticated profile:

```powershell
databricks serving-endpoints create --json @serving/endpoint.json --no-wait --profile YOUR_PROFILE
databricks serving-endpoints get airplane-tail-identifier --profile YOUR_PROFILE
```

Wait for **both** `state.ready = READY` and `state.config_update = NOT_UPDATING`.
The configuration pins a numeric version; moving the `demo` alias alone does not
update the endpoint. Deploy future versions through an explicit config update and
repeat the serving verification before using them in the app.

To change the version on the existing endpoint, edit `serving/endpoint.json` and apply
its `config` object with `databricks serving-endpoints update-config
airplane-tail-identifier --json @<config file>`. The previous version keeps serving
until the new one is ready. The app needs no change: it reads the served version from
the endpoint for each submission and for the Model performance tab.

## Request / response

Send an authenticated `POST` to
`<workspace-host>/serving-endpoints/airplane-tail-identifier/invocations`:

```json
{"dataframe_records": [{"image_base64": "<base64-encoded JPEG or PNG bytes>"}]}
```

The response has a `predictions` array. Each row contains `airline`, `confidence`
and `probabilities_json` (a JSON-encoded string mapping the five airlines to scores).
Encode actual image bytes without a data-URL prefix. Confidence is uncalibrated;
it is not a guarantee that the predicted airline is correct.

## Databricks App integration

Add this resource to the app entry in its future `databricks.yml`:

```yaml
resources:
  apps:
    airline_app:
      name: YOUR_APP_NAME
      source_code_path: ./app
      resources:
        - name: tail-model-endpoint
          serving_endpoint:
            name: airplane-tail-identifier
            permission: CAN_QUERY
```

Reference the **same resource name** in the app's `app.yaml`:

```yaml
env:
  - name: SERVING_ENDPOINT
    valueFrom: tail-model-endpoint
```

This grants the app's service principal permission to query this endpoint. The app
does not need to load model weights or receive training-data access. Call it from
the app backend using managed credentials, keeping authentication out of browser
code. `serving/client.py` shows the Python SDK request/response contract; use the
same JSON contract from a TypeScript backend. In an async Python route, wrap this
synchronous helper with `await asyncio.to_thread(identify_tail, image_bytes)`.
The helper accepts either an already-cropped image or original image bytes with
`crop_box=(left, top, right, bottom)` as normalized coordinates from 0 to 1 on
the EXIF-oriented image. It applies the user-selected crop before encoding and
transmitting, rounds its edges outward, and strips inherited image metadata. For
example, `identify_tail(photo_bytes, crop_box=(0.45, 0.1, 0.95, 0.7))` sends only
that region. The backend must validate the uploaded image and chosen box, and
must use the same orientation as the UI; this helper does not find the tail.
Orientation is read from the EXIF block only: like browsers, it ignores XMP-only
orientation, so the box refers to the image the user saw. 16-bit grayscale PNGs are
scaled to 8 bits rather than clipped. `prepare_tail_png` returns the exact PNG bytes
`identify_tail` transmits, and `query_tail_png` sends already prepared bytes. The
Tailspot app (`app/`) deploys a byte-identical copy as `app/serving_client.py`; a
repository test keeps the two files identical.
Its local
request-format and metadata-removal checks can be run with:

```powershell
uv run --with databricks-sdk --with pillow python -B -m unittest discover -s serving -p test_client.py -v
```

## End-to-end smoke test

Prepare the dataset the served version was trained on, then download that training
run's `result.json` from `/Volumes/.../training_data/runs/<run_id>/result.json`. For
version 3 (dataset v2, run `9d9fe8ed33194dce9cd9e4359408f5b9`):

```powershell
python training/prepare_data.py --annotations dataset/tail_annotations_v2.json
python serving/smoke_test.py --profile YOUR_PROFILE --dataset .local/dataset_v2 --reference .local/training_result_v3.json
```

`--version` defaults to the version in `serving/endpoint.json`. For version 1, use the
defaults `.local/dataset` and `.local/training_result.json` with `--version 1`. The
test checks that the endpoint serves only that version and that the dataset manifest
hash matches the reference run. It then sends one request per held-out image and
compares labels and all probabilities with the registered model's CPU reference. The
local report is ignored. These are deployment consistency checks on existing test
images, not new evidence of generalization.
