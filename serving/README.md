# Airline-tail Model Serving

Endpoint: **`airplane-tail-identifier`**

- Model: `dbx_ai_app.airplane_tail_id.airplane_tail_identifier`, immutable version **1**.
- Small CPU workload, 100% traffic to `airplane-tail-v1`, scale-to-zero enabled.
- Tail-focused image input; no automatic localization or unknown-airline detection.
- Deployment verified: ten held-out requests returned the same labels and all class
  probabilities as the registered version's CPU reference (10/10 correct on these
  existing test crops). The endpoint reached `READY` / `NOT_UPDATING`.
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
Its local
request-format and metadata-removal checks can be run with:

```powershell
uv run --with databricks-sdk --with pillow python -B -m unittest discover -s serving -p test_client.py -v
```

## End-to-end smoke test

After preparing `.local/dataset/` and downloading the successful training run's
`result.json` as `.local/training_result.json`:

```powershell
python serving/smoke_test.py --profile YOUR_PROFILE
```

This submits ten independent requests (one per original held-out tail image) and
compares labels and all probabilities with the registered model's CPU reference.
The local report is ignored. These are deployment consistency checks on existing
test images, not new evidence of generalization.
