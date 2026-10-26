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
