-- Model evaluation results published from evaluation/compare.py by evaluation/publish.py.
-- Read by the Tailspot app's Model performance tab and notebooks/04_model_performance.py.
-- One comparison (eval_run_id = the comparison's MLflow run ID) scores several registered
-- model versions on the same evaluation sets. Rerunning is safe; nothing is dropped.

CREATE TABLE IF NOT EXISTS dbx_ai_app.airplane_tail_id.model_eval_runs (
  eval_run_id STRING NOT NULL COMMENT 'MLflow run ID of the comparison (experiment airplane-tail-evaluation)',
  created_at TIMESTAMP NOT NULL COMMENT 'When the comparison finished',
  model_name STRING NOT NULL,
  model_version STRING NOT NULL,
  model_uri STRING NOT NULL,
  training_run_id STRING COMMENT 'MLflow training run that produced the version',
  recipe STRING COMMENT 'Training recipe: v1 or improved',
  dataset_version STRING COMMENT 'Training dataset version, e.g. v1 or v2',
  val_accuracy DOUBLE COMMENT 'Validation accuracy recorded by the training run (selection metric)',
  val_loss DOUBLE COMMENT 'Validation loss recorded by the training run (selection tie-break)',
  aliases STRING COMMENT 'JSON array of registry aliases on the version when published',
  serving BOOLEAN COMMENT 'True if this version matched the serving_endpoint_version parameter given to the comparison',
  selected BOOLEAN COMMENT 'True for the candidate chosen by validation in this comparison',
  child_run_id STRING COMMENT 'Nested MLflow run holding this version''s evaluation metrics'
) USING DELTA
COMMENT 'One row per model version scored in a comparison.';

CREATE TABLE IF NOT EXISTS dbx_ai_app.airplane_tail_id.model_eval_metrics (
  eval_run_id STRING NOT NULL,
  model_version STRING NOT NULL,
  eval_set STRING NOT NULL COMMENT 'v2_val, v1_test, new_holdout, v2_test, full_plane_stress, context_sensitivity or app_reviewed',
  variant STRING NOT NULL COMMENT 'tail_crop, whole_image, or context_<scale> (context_1.0 ... context_full)',
  metric STRING NOT NULL COMMENT 'accuracy, macro_f1, nll, brier, ece, mean_confidence, mean_true_prob, share_pred_<class>, f1_<class>, precision_<class>, recall_<class>, support_<class>',
  value DOUBLE,
  n INT NOT NULL COMMENT 'Images scored in this set and variant'
) USING DELTA
COMMENT 'Long-format evaluation metrics per comparison, model version, set and variant.';

CREATE TABLE IF NOT EXISTS dbx_ai_app.airplane_tail_id.model_eval_predictions (
  eval_run_id STRING NOT NULL,
  model_version STRING NOT NULL,
  eval_set STRING NOT NULL,
  variant STRING NOT NULL,
  item_id STRING NOT NULL COMMENT 'Image ID within the set',
  actual STRING NOT NULL,
  predicted STRING NOT NULL,
  confidence DOUBLE NOT NULL COMMENT 'Uncalibrated winning probability',
  true_prob DOUBLE NOT NULL COMMENT 'Probability assigned to the actual airline',
  probabilities_json STRING NOT NULL,
  correct BOOLEAN NOT NULL,
  source STRING COMMENT 'Repository-relative or Volume path of the scored image source'
) USING DELTA
COMMENT 'Every prediction behind model_eval_metrics.';
