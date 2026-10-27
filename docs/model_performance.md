# Model performance: contracts for dataset v2, training, evaluation and display

Working contract shared by `training/`, `evaluation/`, `app/` and `notebooks/04_model_performance.py`.

## Dataset v2 (`training/prepare_data.py --annotations dataset/tail_annotations_v2.json`)

Annotation file: `dataset/tail_annotations_v2.json` (version 2). Each record has `source_path`
(repository-relative, under one of `source_roots`), `airline`, `bbox` (normalized
left, top, right, bottom on the EXIF-oriented source), `group`, `split` (explicit), `kind`
(`tail-photo`, `tail-crop` or `full-plane`) and `box_source`. Every image under the source roots
must be annotated exactly once. A group never spans splits. v1 records keep their v1 bbox and split.

Output (`.local/dataset_v2`, uploaded to `/Volumes/dbx_ai_app/airplane_tail_id/training_data/dataset_v2`):

```
manifest.json
{train,val,test}/<airline>/<id>.png          tail crop (as v1)
context/{train,val,test}/<airline>/<id>.png  tail box expanded 2.5x about its centre, clamped to the image
originals/{train,val,test}/<airline>/<id>.png  upright metadata-free source, full-plane records only
sources/<root>/<airline>/SOURCES.md
review/<airline>/review.jpg                  contact sheet, sized for any count
```

`manifest.json` (version 2) keeps every v1 field and adds:
- top level: `dataset_version` (`"v2"`), `parent_manifest_sha256`, `source_roots`,
  `split_policy` (`{"method": "explicit", "note": ...}`), `counts` (`{class: {train, val, test}}`),
  `context_scale` (2.5), `screen_checks` (`{"against": [...], "findings": []}`).
- per record: `source_root`, `kind`, `box_source`, `context_path`, `context_bbox` (tail box
  normalized within the context image), `context_sha256`, and for full-plane records
  `original_path`, `original_sha256` (else null). IDs: `<file stem>` (unique across roots).

## Training (`training/train.py`, AIR experiment `airplane-tail-identifier`)

New parameters (defaults keep v1 behaviour): `recipe` (`v1` | `improved`), `alias` (`""` = set no
alias), `latest_pointer` (`""` = do not write), plus improved-recipe hyperparameters. The run logs:
- params: all parameters, `dataset_version`, `recipe`, `manifest_sha256`, split sizes
- tags: `dataset_version`, `recipe`, `parent_manifest_sha256`, `task`, `checkpoint_selection`
- per-epoch metrics (step = epoch): `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`,
  `val_macro_f1`, `lr`, `epoch_seconds`
- final metrics: `best_epoch`, `best_val_accuracy`, `best_val_loss`, `test_accuracy`,
  `test_macro_f1`, `test_loss`, `test_ece`, `test_brier`, `test_f1_<class>`, baselines, timings
- artifacts: dataset manifest/sources/review, `eval/test_predictions.json` (table),
  `eval/test_per_class.json` (table), `figures/learning_curves.png`, `figures/confusion_matrix.png`,
  `figures/reliability.png` (when matplotlib is available), `result.json`, `environment.json`
- registered model version tags: `recipe`, `dataset_version`, `val_accuracy`, `val_loss`,
  `training_run_id`, `reload_test`

## Evaluation (`evaluation/compare.py`, AIR experiment `airplane-tail-evaluation`)

One parent run per comparison, one nested child run per model version. Child metric keys:
`<eval_set>/<variant>/<metric>`. Eval sets and variants:

| eval_set | images | variants |
| --- | --- | --- |
| `v2_val` | v2 validation crops (13) | `tail_crop` |
| `v1_test` | the 10 v1 test crops | `tail_crop` |
| `new_holdout` | v2 test records not in v1 (6) | `tail_crop` |
| `v2_test` | all v2 test crops (16) | `tail_crop` |
| `full_plane_stress` | `full_plane_tests` (10) | `whole_image`, `tail_crop` |
| `context_sensitivity` | sources with a full image and a tail box: full_plane_tests (10), v2 test full-plane originals, reviewed app submissions | `context_1.0`, `context_1.5`, `context_2.0`, `context_3.0`, `context_full` |
| `app_reviewed` | reviewed app submissions (stored tail crop) | `tail_crop` |

Metrics per set and variant: `accuracy`, `macro_f1`, `nll`, `brier`, `ece` (5 equal-width
confidence bins), `mean_confidence`, `mean_true_prob`, `share_pred_<class>`, and per class
`f1_`, `precision_`, `recall_`, `support_<class>`. Context crops: box scaled about its centre,
clamped to the image; `context_full` is the whole image.

Outputs in `/Volumes/dbx_ai_app/airplane_tail_id/training_data/evaluations/<eval_run_id>/`:
`result.json`, `runs.jsonl`, `metrics.jsonl`, `predictions.jsonl` (rows match the tables in
`app_setup/create_eval_tables.sql`). `evaluation/publish.py` loads them into
`model_eval_runs`, `model_eval_metrics` and `model_eval_predictions` (replacing rows for that
`eval_run_id`).

## App performance API (read-only, `Cache-Control: no-store`)

- `GET /api/performance/summary`
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
  `available` is false (other fields empty) when no comparison has been published.
- `GET /api/performance/predictions?eval_set=<set>&variant=<variant>` → `{"eval_run_id", "rows":
  [{model_version, item_id, actual, predicted, confidence, true_prob, correct}]}` for the latest
  comparison (422 for unknown set/variant names).
- `GET /api/performance/training-runs` → `{"experiment_id", "runs": [{run_id, run_name, status,
  start_time, end_time, recipe, dataset_version, registered_model_version, best_val_accuracy,
  test_accuracy, test_macro_f1, epochs_completed}]}` newest first, at most 20, from the training
  experiment (live, includes RUNNING runs).
- `GET /api/performance/training-runs/{run_id}/history` → `{"run_id", "status", "metrics":
  {"train_loss": [[step, value], ...], "train_accuracy": ..., "val_loss": ..., "val_accuracy": ...,
  "val_macro_f1": ..., "lr": ...}}` (404 if the run is not in the training experiment).
- `GET /api/performance/submissions` → `{"total", "succeeded", "failed", "reviewed",
  "reviewed_correct", "by_predicted_airline": [{airline, count}], "confidence_buckets":
  [{"lower": 0.2, "upper": 0.4, "count": 3}], "by_model_version": [{model_version,
  submissions, reviewed, correct}], "recent": [{submission_id, submitted_at, status,
  predicted_airline, confidence, review_label, review_source, correct, model_version,
  has_photo}]}` (recent: last 10, no paths).
- `GET /api/submissions/{id}/photo?kind=thumb|crop|original` → `image/png` of a stored
  submission photo (thumbnail ≤ 256 px, exact model input, or upright original).
