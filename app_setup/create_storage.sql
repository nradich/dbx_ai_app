-- Storage for photos submitted through the Tailspot Databricks App.
-- Requires USE CATALOG, USE SCHEMA, CREATE VOLUME and CREATE TABLE. Nothing is
-- dropped or replaced; rerunning is safe.

CREATE VOLUME IF NOT EXISTS dbx_ai_app.airplane_tail_id.submitted_photos
COMMENT 'Tailspot app submissions: upright metadata-free originals and the exact tail crops sent to the model.';

CREATE TABLE IF NOT EXISTS dbx_ai_app.airplane_tail_id.photo_submissions (
  submission_id STRING NOT NULL COMMENT 'Random UUID, also the submission folder name in the Volume',
  submitted_at TIMESTAMP NOT NULL COMMENT 'When the app accepted the upload',
  completed_at TIMESTAMP COMMENT 'When processing finished',
  status STRING NOT NULL COMMENT 'succeeded or failed',
  error STRING COMMENT 'Why the submission failed, or a warning on success when departures could not be loaded',
  airport_iata STRING NOT NULL COMMENT 'Airport of the sighting. The demo schedule covers SAN only',
  observed_local STRING NOT NULL COMMENT 'Sighting wall-clock time, yyyy-MM-ddTHH:mm[:ss], America/Los_Angeles',
  observed_at TIMESTAMP NOT NULL COMMENT 'Sighting instant derived from observed_local',
  upload_format STRING NOT NULL COMMENT 'Decoded upload format (JPEG, MPO or PNG). Upload bytes are not stored',
  upload_bytes BIGINT NOT NULL COMMENT 'Size of the uploaded file',
  upload_sha256 STRING NOT NULL COMMENT 'SHA-256 of the uploaded file bytes',
  original_path STRING NOT NULL COMMENT 'Upright, metadata-free PNG of the full upload',
  original_sha256 STRING NOT NULL,
  original_width INT NOT NULL,
  original_height INT NOT NULL,
  crop_left DOUBLE NOT NULL COMMENT 'Normalized tail box on the upright original, 0 to 1',
  crop_top DOUBLE NOT NULL,
  crop_right DOUBLE NOT NULL,
  crop_bottom DOUBLE NOT NULL,
  crop_path STRING NOT NULL COMMENT 'Exact PNG bytes sent to the model',
  crop_sha256 STRING NOT NULL,
  crop_width INT NOT NULL,
  crop_height INT NOT NULL,
  crop_downscaled BOOLEAN NOT NULL COMMENT 'True if the crop was reduced to the app maximum side before inference',
  serving_endpoint STRING NOT NULL,
  served_entity STRING COMMENT 'Served entity that answered, when reported',
  model_name STRING,
  model_version STRING,
  inference_attempts INT NOT NULL COMMENT 'Serving requests made, including cold-start retries',
  inference_seconds DOUBLE COMMENT 'Seconds from first request to a response or final failure',
  predicted_airline STRING COMMENT 'Model output label, not a confirmed label',
  confidence DOUBLE COMMENT 'Uncalibrated winning probability',
  probabilities_json STRING COMMENT 'JSON object of all class probabilities',
  candidate_count INT COMMENT 'Number of possible departures shown. NULL if the lookup failed or was skipped',
  candidates_json STRING COMMENT 'JSON array of the possible departures shown. NULL if the lookup failed or was skipped',
  review_label STRING COMMENT 'Reviewer-confirmed airline, NULL until reviewed',
  reviewed_at TIMESTAMP COMMENT 'When review_label was set',
  review_source STRING COMMENT 'Who set review_label: app_user (feedback in the app) or reviewer (NULL on older reviewed rows means reviewer)'
) USING DELTA
COMMENT 'One row per Tailspot app submission. Predictions are not training labels. No user identity is stored.';
