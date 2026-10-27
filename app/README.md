# Tailspot Databricks App

A FastAPI backend and a static browser UI (adapted from `app_wireframe/`). The user
chooses a JPEG or PNG photo, outlines the aircraft tail, and enters the San Diego
time of the sighting. The app then:

1. Validates the photo, turns it upright using its EXIF orientation, and saves two
   metadata-free PNGs to the Unity Catalog Volume
   `dbx_ai_app.airplane_tail_id.submitted_photos`:
   `YYYY/MM/DD/<submission_id>/original.png` (whole photo) and `tail_crop.png` (the exact
   bytes sent to the model; crops longer than 1536 px are reduced to 1536 px).
2. Queries the `airplane-tail-identifier` serving endpoint in the background, retrying
   while it starts from zero (a cold start took about 90 seconds).
3. Looks up the synthetic weekly SAN schedule (`demo_airlines`, `demo_weekly_departures`)
   with bound parameters, then lists scheduled gate departures from 60 minutes before
   to 10 minutes after the sighting. Matching uses the same recurring local-time logic
   as `schedule/lookup_example.sql` and `tests/test_schedule_fixture.py`.
4. Records one row per submission in `dbx_ai_app.airplane_tail_id.photo_submissions`
   (an idempotent `MERGE`): hashes, sizes, crop box, sighting time, model version,
   prediction, probabilities and the departures shown. No user identity is stored.

The browser polls `GET /api/submissions/{id}`; see [API.md](API.md) for the contract.
The demo limits from the main README still apply: the model expects a tail crop, has no
unknown-airline rejection, and its confidence is uncalibrated. The schedule is
synthetic, and a sighting cannot confirm a particular flight.

## Model performance tab

The **Model performance** tab (`#performance`) is read-only:
- **Latest comparison:** read from `model_eval_runs`, `model_eval_metrics` and
  `model_eval_predictions`, published by `evaluation/publish.py`. It shows the model
  versions, a results matrix with n, context-sensitivity charts, and a per-set
  drill-down with confusion matrices and every prediction.
- **Training runs:** read live from the MLflow experiment `airplane-tail-identifier`.
  Running runs refresh automatically.
- **App submission statistics:** read from `photo_submissions`.

The app reads the experiment ID from the `training-experiment` resource
(`TAILSPOT_TRAINING_EXPERIMENT_ID`). The bundle variable `training_experiment_id`
defaults to this workspace's experiment; override it in other workspaces. The tab
never changes aliases or the serving endpoint.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI routes, upload size limit, static files |
| `pipeline.py` | Validation, storage, inference retries, schedule lookup, recording |
| `performance.py` | Read-only model performance endpoints (UC tables, MLflow, submissions) |
| `schedule.py` | Sighting-time parsing and the departure window |
| `serving_client.py` | Byte-identical copy of `serving/client.py` (enforced by a test) |
| `static/` | Browser UI (`app.js` Identify tab, `performance.js` Model performance tab) |
| `app.yaml`, `requirements.txt` | Databricks Apps runtime configuration |

## Setup and deployment

From the repository root, with an authenticated profile:

1. Load the schedule tables as described in `schedule/README.md`.
2. Run `app_setup/create_storage.sql` to create the Volume and submissions table, and
   `app_setup/create_eval_tables.sql` for the model performance tables. The bundle
   declares those tables as resources, so they must exist before deployment.
3. Deploy the bundle (`databricks.yml`) and start the app:
   ```powershell
   databricks bundle validate -p YOUR_PROFILE
   databricks bundle deploy -p YOUR_PROFILE
   databricks bundle run tailspot -p YOUR_PROFILE
   ```
   The bundle declares the app's resources: `CAN_QUERY` on the serving endpoint,
   `CAN_USE` on the SQL warehouse (looked up by name), and Unity Catalog access to
   the tables and Volume. On deployment, Databricks granted the app's service
   principal everything it needs: `USE CATALOG`, `USE SCHEMA`, `SELECT` on the
   schedule tables, `SELECT` and `MODIFY` on `photo_submissions`, and `READ VOLUME`
   and `WRITE VOLUME` on `submitted_photos`. It also grants `SELECT` on the three
   `model_eval_*` tables and `CAN_READ` on the training experiment.
4. Optional: `app_setup/grant_app_access.sql` lists the same grants explicitly, for
   deployments without the bundle. Substitute the `service_principal_client_id`
   shown by `databricks apps get tailspot` before running it.
5. Check effective access with
   `databricks grants get-effective table dbx_ai_app.airplane_tail_id.photo_submissions --principal <client id>`.

## Local development

Use Python 3.11 and the pinned `requirements.txt`. Authenticate with a CLI profile,
then run the app from this folder:

```powershell
$env:DATABRICKS_CONFIG_PROFILE = 'YOUR_PROFILE'
$env:DATABRICKS_WAREHOUSE_ID = '<warehouse id>'
python main.py
```

Tests run without Databricks credentials:
`python -m unittest discover -s tests -p "test_app_*.py" -v` from the repository root.

## Stored photos

Submissions are kept for later review and possible retraining, as proposed in
`app_wireframe/README.md`. Predictions are not training labels. After a result, the
app asks **"Did we get the airline right?"**: the user answers yes, or no and picks the
actual airline (or "another airline"). The answer is stored in `review_label` with
`review_source = 'app_user'`. A label set by a reviewer outside the app has
`review_source = 'reviewer'` and is never overwritten by the app. App answers feed the
reviewed-accuracy figures on the Model performance tab. Confirm them before using a
photo for training or evaluation. Tables created before this column existed need
`app_setup/add_review_source.sql` once. Define retention and user-facing disclosure
before wider use. The UI notes that the photo, tail outline and time are saved with
the result.
