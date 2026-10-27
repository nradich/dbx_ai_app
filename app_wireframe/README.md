# Tailspot app wireframe

Open `index.html` in a browser to review an interactive, local-only prototype.
Select a JPEG/PNG, drag a box around the tail (or enter its edges as percentages),
then choose **Preview tail crop**. The crop can also be downloaded locally as PNG.
Use **Show sample result** to switch between
empty and result states. The sample is fixed: neither the photo nor the entered
time is sent anywhere or affects the result. The sample flights correspond to the
synthetic Tuesday 08:20 example in `schedule/lookup_example.sql`.

This is a design mock, not a deployable Databricks App. The working implementation
is the Tailspot app in [`app/`](../app/README.md), which follows the plan below and
the storage proposal in the next section. The selection and crop in this mock
run locally in the browser, but there is no inference, authentication, or SQL
connection. Future implementation should send the original bytes and normalized
crop coordinates to the backend, invoke `serving/client.py` with managed
credentials, and query Unity Catalog using an observation instant and a
parameterized recurring-schedule lookup. Coordinates are `[left, top, right,
bottom]` fractions of the EXIF-oriented image. The backend transposes the image,
validates the box and crops before sending image bytes to the endpoint. The
browser's crop preview uses the same outward-rounded pixel boundaries. The
backend must handle missing/invalid
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
This wireframe does not store photos. The Tailspot app now implements the storage
part: the `submitted_photos` Volume, the `photo_submissions` table (with empty
`review_label`/`reviewed_at` columns for review) and image hashes. Review tooling,
a retention policy and a retraining pipeline are still future work.
