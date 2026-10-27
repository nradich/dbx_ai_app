# Tailspot app — HTTP API contract

Used by `static/app.js` and implemented by `main.py` / `pipeline.py`. All responses are
JSON. Times are ISO-8601 UTC strings ending in `Z` (for example `2026-10-27T15:20:00Z`)
unless named `*_local`. The airport is fixed: SAN, time zone America/Los_Angeles.
Error bodies are always `{"detail": "<human readable message>"}`.

## GET /api/health
200 `{"status": "ok"}` — makes no Databricks calls.

## GET /api/config
200 (`Cache-Control: no-store`)
```json
{
  "airport_iata": "SAN",
  "airport_name": "San Diego International Airport",
  "time_zone": "America/Los_Angeles",
  "now_local": "2026-10-27T08:20",
  "max_upload_bytes": 20971520,
  "window_minutes_before": 60,
  "window_minutes_after": 10
}
```
`now_local` is the current SAN wall-clock time (minute precision). The schedule window
covers scheduled gate departures from `window_minutes_before` minutes before the sighting
to `window_minutes_after` minutes after it, inclusive.

## POST /api/submissions (multipart/form-data)
| Field | Required | Meaning |
| --- | --- | --- |
| `photo` | yes | JPEG or PNG file, at most 20 MiB and 40 megapixels |
| `crop_left`, `crop_top`, `crop_right`, `crop_bottom` | yes | Decimal fractions 0–1 of the EXIF-oriented (upright) image; left < right, top < bottom; the box must cover at least 2 pixels each way |
| `observed_local` | yes | `YYYY-MM-DDTHH:MM` (optional `:SS`), San Diego wall-clock sighting time, at most 10 minutes in the future. Repeated times when clocks go back resolve to the first occurrence; skipped times when clocks go forward are rejected |
| `image_width`, `image_height` | no (both or neither) | Upright pixel size the browser displayed; a mismatch with the server's EXIF-oriented size is rejected so the box is never applied to a differently rotated image |

Orientation comes from the EXIF block only (XMP-only orientation is ignored, as browsers do).
The upright original and the tail crop are both saved as metadata-free PNGs before the
response is sent. A crop whose longest side exceeds 1536 px is reduced to 1536 px; the
stored crop is exactly what the model receives.

Responses:
- **202** `{"submission_id": "<uuid4>", "status": "queued", "status_url": "/api/submissions/<uuid4>"}`
- 413 photo or request body too large
- 422 invalid or unsupported image, invalid tail box, orientation mismatch, invalid,
  nonexistent or future time, or a missing field
- 502 the photo could not be saved to storage
- 503 too many submissions in progress, or the app is restarting — try again shortly
- 500 unexpected error

## GET /api/submissions/{submission_id}
`Cache-Control: no-store`. 404 `{"detail": "Submission not found"}` for unknown or
malformed ids; 503 if the status cannot be read right now (the client retries).
After an app restart, finished submissions are read back from the submissions table.

200:
```json
{
  "submission_id": "uuid",
  "status": "queued | processing | succeeded | failed",
  "stage": "queued | waiting_for_model | looking_up_schedule | recording | done",
  "message": "Human readable progress or failure text",
  "inference_attempts": 2,
  "submitted_at": "2026-10-27T15:21:03Z",
  "observed_local": "2026-10-27T08:20",
  "observed_at": "2026-10-27T15:20:00Z",
  "airport_iata": "SAN",
  "prediction": null,
  "model": null,
  "candidates": [],
  "schedule_lookup": "pending | ok | failed | skipped",
  "storage": {
    "original_path": "/Volumes/.../original.png",
    "crop_path": "/Volumes/.../tail_crop.png",
    "crop_width": 512, "crop_height": 640, "crop_downscaled": false
  },
  "recorded": false,
  "error": null
}
```
- `prediction`, once the model answers:
  ```json
  {"airline": "southwest", "airline_name": "Southwest Airlines", "iata_code": "WN",
   "confidence": 0.93,
   "probabilities": {"alaska": 0.01, "american": 0.02, "delta": 0.01, "southwest": 0.93, "united": 0.03}}
  ```
  `airline_name` and `iata_code` are null if the label is not in `demo_airlines`.
  Confidence is uncalibrated.
- `model`: null while queued, then always
  `{"endpoint": "airplane-tail-identifier", "served_entity": "airplane-tail-v1",
  "model_name": "dbx_ai_app.airplane_tail_id.airplane_tail_identifier", "model_version": "1"}`
  (fields other than `endpoint` may be null).
- `candidates`, closest first (empty unless `schedule_lookup` is `ok`):
  ```json
  [{"flight_number": "WN1002", "destination_iata": "PHX", "destination_city": "Phoenix",
    "day_of_week": "TUE", "local_date": "2026-10-27", "scheduled_departure_local": "08:30:00",
    "scheduled_departure_utc": "2026-10-27T15:30:00Z",
    "minutes_after_scheduled_gate_departure": -10.0}]
  ```
  `minutes_after_scheduled_gate_departure` is the sighting minus the scheduled departure:
  negative means the departure is scheduled that many minutes after the sighting.
- `schedule_lookup`: `ok` (candidates may still be empty), `failed` (the lookup did not
  run; the prediction is still shown and `error` explains), `skipped` (no prediction),
  `pending` (not reached yet).
- `status` `succeeded` means the model answered. `error` may still be set — for example
  when departures could not be loaded or the record could not be saved (`recorded` false).
- `status` `failed` means no prediction; `error` explains (model timeout or failure).

The client polls every 3 seconds. A model cold start typically takes about 90 seconds
(`stage` `waiting_for_model`, `inference_attempts` increases on retries). Each Databricks
call is time-bounded, and no new inference attempt starts after 10 minutes, so every
submission reaches `succeeded` or `failed`.

## Model performance endpoints

Read-only; implemented by `performance.py` (shared contract: `docs/model_performance.md`).
Every response, including errors, has `Cache-Control: no-store`. The server caches results
briefly (summary and predictions 30 s, training runs 10 s, run history 10 s while running and
300 s once finished, submissions 15 s), so new data can take that long to appear. Errors:
404 / 422 as listed below, and 503 `{"detail": "..."}` when the SQL warehouse or MLflow
cannot be read right now (failures are not cached; retry later). Numbers are JSON numbers;
missing or non-finite values are `null`. Model versions are strings.

### GET /api/performance/summary
The latest published comparison (the `eval_run_id` with the newest `created_at` in
`model_eval_runs`):
```json
{"available": true,
 "eval_run_id": "...", "evaluated_at": "2026-10-27T20:00:00Z",
 "serving": {"endpoint": "airplane-tail-identifier", "model_version": "1"},
 "models": [{"model_version": "1", "recipe": "v1", "dataset_version": "v1",
             "training_run_id": "...", "val_accuracy": 1.0, "val_loss": 0.29,
             "aliases": ["demo"], "serving": true, "selected": false}],
 "eval_sets": [{"eval_set": "v1_test", "label": "v1 test crops", "description": "...",
                "variants": ["tail_crop"]}],
 "metrics": [{"model_version": "1", "eval_set": "v1_test", "variant": "tail_crop",
              "metric": "accuracy", "value": 1.0, "n": 10}]}
```
- `available` is false, with `eval_run_id`/`evaluated_at` null and empty `models`,
  `eval_sets` and `metrics`, when no comparison has been published.
- `serving` is read live from the serving endpoint (`model_version` null if it cannot be read
  or the endpoint serves more than one version); `models[].serving` records what was served
  when the comparison ran.
- `models` are ordered by version number. `eval_sets` lists, in a fixed order (`v2_val`,
  `v1_test`, `new_holdout`, `v2_test`, `full_plane_stress`, `context_sensitivity`,
  `app_reviewed`), only sets with metrics in this comparison, with a plain-English `label` and
  `description` (including caveats such as the full-plane contamination note) and the variants
  present (`tail_crop`; `whole_image` then `tail_crop`; `context_1.0` ... `context_3.0` by scale,
  then `context_full`). `metrics` is long format, one row per model version, set, variant and
  metric (names as in `model_eval_metrics`); `n` is the number of images scored.

### GET /api/performance/predictions
Query parameters: `eval_set` and `variant` (for example `?eval_set=v1_test&variant=tail_crop`),
and optionally `eval_run_id` (the summary's `eval_run_id`, so the drill-down matches the
summary even if a newer comparison is published in between).
```json
{"eval_run_id": "...",
 "rows": [{"model_version": "1", "item_id": "delta_tail_3", "actual": "delta",
           "predicted": "delta", "confidence": 0.93, "true_prob": 0.93, "correct": true}]}
```
Rows of the given comparison (default: the latest), ordered by model version then `item_id`;
no image paths. Without `eval_run_id`, `eval_run_id` is null and `rows` empty when nothing is
published; with it, `eval_run_id` echoes the parameter and `rows` is empty if that comparison
has no rows for the set and variant. 422 if `eval_set` or `variant` is missing or unknown
(`variant` must belong to the set; `context_sensitivity` accepts `context_full` and
`context_<scale>` such as `context_1.5`) or `eval_run_id` is not 32 lowercase hex characters.

### GET /api/performance/training-runs
```json
{"experiment_id": "350719556651226",
 "runs": [{"run_id": "<32 hex>", "run_name": "...", "status": "RUNNING",
           "start_time": "2026-10-27T19:00:00Z", "end_time": null, "recipe": "improved",
           "dataset_version": "v2", "registered_model_version": null,
           "best_val_accuracy": 0.92, "test_accuracy": null, "test_macro_f1": null,
           "epochs_completed": 7}]}
```
At most 20 runs of the training experiment, newest first, read live (includes running
runs). `status` is `RUNNING`, `SCHEDULED`, `FINISHED`, `FAILED` or `KILLED`. `recipe` and
`dataset_version` are `"v1"` for runs logged before those tags existed.
`best_val_accuracy` is the run's `best_val_accuracy` metric; only while the run is `RUNNING`
and that metric is not yet logged does it fall back to the latest `val_accuracy`, otherwise it
is null. `epochs_completed` is the step of the latest `val_accuracy`.
`registered_model_version` is set once training registered a model version. If the app has
no training experiment configured: `{"experiment_id": null, "runs": []}`.

### GET /api/performance/training-runs/{run_id}/history
```json
{"run_id": "<32 hex>", "status": "FINISHED",
 "metrics": {"train_loss": [[1, 1.21], [2, 0.84]], "train_accuracy": [], "val_loss": [],
             "val_accuracy": [], "val_macro_f1": [], "lr": []}}
```
Per-epoch `[step, value]` pairs sorted by step (step = epoch; the latest value wins if a step
was logged twice); a metric that was never logged is `[]`. 404 `{"detail": "Training run not
found."}` if the ID is not 32 lowercase hex characters, the run does not exist or is
deleted, it belongs to another experiment, or no training experiment is configured.

### GET /api/performance/submissions
```json
{"total": 9, "succeeded": 7, "failed": 2, "reviewed": 4, "reviewed_correct": 3,
 "by_predicted_airline": [{"airline": "delta", "count": 5}],
 "confidence_buckets": [{"lower": 0.0, "upper": 0.2, "count": 0},
                        {"lower": 0.2, "upper": 0.4, "count": 3},
                        {"lower": 0.4, "upper": 0.6, "count": 0},
                        {"lower": 0.6, "upper": 0.8, "count": 0},
                        {"lower": 0.8, "upper": 1.0, "count": 4}],
 "recent": [{"submission_id": "uuid", "submitted_at": "2026-10-27T15:21:03Z",
             "predicted_airline": "delta", "confidence": 0.91, "review_label": null,
             "model_version": "2"}]}
```
From `photo_submissions`. `reviewed` counts rows with a `review_label`; `reviewed_correct`
those with a non-null `predicted_airline` equal to the `review_label` (a reviewed failed
submission counts as reviewed but not correct). `by_predicted_airline` is ordered by
count (descending) then airline; failed submissions have no prediction and are not listed.
`confidence_buckets` always has five buckets: a value on an edge goes to the upper bucket and
1.0 to the last. `recent` is the 10 newest submissions; storage paths are never returned.
Predictions and confidences are uncalibrated model outputs, not confirmed labels.
