-- Unity Catalog access for the Tailspot app's service principal.
-- Deploying with databricks.yml already grants all of these through the app's declared
-- resources (verified on deployment). Use this script for deployments without the
-- bundle, or to make the grants explicit. Replace every APP_SP_CLIENT_ID with the app's
-- service principal client ID, shown as service_principal_client_id by:
-- databricks apps get tailspot
-- SELECT on photo_submissions supports status after a restart (and MERGE reads its
-- target). READ VOLUME allows cleanup of a partially stored submission. Rerunning is safe.

GRANT USE CATALOG ON CATALOG dbx_ai_app TO `APP_SP_CLIENT_ID`;
GRANT USE SCHEMA ON SCHEMA dbx_ai_app.airplane_tail_id TO `APP_SP_CLIENT_ID`;
GRANT SELECT ON TABLE dbx_ai_app.airplane_tail_id.demo_airlines TO `APP_SP_CLIENT_ID`;
GRANT SELECT ON TABLE dbx_ai_app.airplane_tail_id.demo_weekly_departures TO `APP_SP_CLIENT_ID`;
GRANT SELECT, MODIFY ON TABLE dbx_ai_app.airplane_tail_id.photo_submissions TO `APP_SP_CLIENT_ID`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME dbx_ai_app.airplane_tail_id.submitted_photos TO `APP_SP_CLIENT_ID`;
-- Model performance tab (tables from app_setup/create_eval_tables.sql). The app also needs CAN_READ on
-- the MLflow training experiment airplane-tail-identifier (an app resource in databricks.yml).
GRANT SELECT ON TABLE dbx_ai_app.airplane_tail_id.model_eval_runs TO `APP_SP_CLIENT_ID`;
GRANT SELECT ON TABLE dbx_ai_app.airplane_tail_id.model_eval_metrics TO `APP_SP_CLIENT_ID`;
GRANT SELECT ON TABLE dbx_ai_app.airplane_tail_id.model_eval_predictions TO `APP_SP_CLIENT_ID`;
