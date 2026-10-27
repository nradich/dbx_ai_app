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
