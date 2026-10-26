# Tailspot app wireframe

Open `index.html` in a browser to review an interactive, local-only prototype.
Select a JPEG/PNG to preview it and use **Show sample result** to switch between
empty and result states. The sample is fixed: neither the photo nor the entered
time is sent anywhere or affects the result. The sample flights correspond to the
synthetic Tuesday 08:20 example in `schedule/lookup_example.sql`.

This is a design mock, not a deployable Databricks App. There is no inference,
photo cropping, authentication, or SQL connection. Future implementation should
accept a user-selected tail crop, invoke `serving/client.py` from the backend with
managed credentials, and query Unity Catalog using an observation instant and a
parameterized recurring-schedule lookup. The backend must handle missing/invalid
images, endpoint cold starts and errors, and zero/multiple schedule candidates.
See `serving/README.md`, `schedule/README.md`, and `full_plane_tests/RESULTS.md`.
The model does not reliably handle full-plane photos or reject unknown airlines;
its confidence is uncalibrated. The current schedule is synthetic.

## Submitted photos and future retraining

The future app should save submitted images through its backend to a dedicated
Unity Catalog Volume with appropriate app write and reviewer read access. Preserve
the submitted image and the exact tail crop used for inference as separate,
linked artifacts when cropping is supported; the current model needs a tail-focused
image. Record a stable submission ID, storage paths, submission time, model
version, predicted airline and confidence alongside them. Avoid embedding image
bytes in the schedule tables. Define retention and user-facing disclosure before
enabling collection, and validate file types and size before storage.

Predictions are not training labels: a reviewer must confirm or correct the airline
and approve images for a separately versioned training dataset. Track image hashes
and source/submission groups so duplicates and related crops stay in the same split.
This is a proposed future workflow; this wireframe does not store photos or create
a Volume, metadata table, or retraining pipeline.
