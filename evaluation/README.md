# Reproduce the external photo evaluation

From the repository root, with an authenticated profile:

```powershell
python full_plane_tests/download_photos.py validate
databricks fs cp -r full_plane_tests dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/full_plane_tests_v1 --profile YOUR_PROFILE
```

From the `evaluation/` directory:

```powershell
air run -f workload.yaml --dry-run -p YOUR_PROFILE
air run -f workload.yaml -p YOUR_PROFILE
```

This uses the matching managed AI environment on one A10. The registered wrapper
performs CPU inference; this small job uses AIR for environment consistency. It
does not retrain or alter the model. Version 1 is pinned; photos and boxes are
frozen before prediction. All 20 predictions (two views of ten sources) and metrics
are logged to MLflow and written to a run-specific Volume `result.json`.

Import `notebooks/03_full_plane_evaluation.py` as a Databricks source notebook.
Set `result_path` to that result file to display every input/prediction. The supplied
`validate_notebook.json` replays the completed evaluation on serverless CPU compute.
`export_results.py` copies portable measured results without operational paths or
workspace identities to `full_plane_tests/results.json`.

# Compare model versions (dataset v2 and later)

`compare.py` scores several registered versions on the same evaluation sets and
logs one MLflow parent run (experiment `airplane-tail-evaluation`), with one nested
run per version. Metric keys have the form `<eval_set>/<variant>/<metric>`. The sets,
variants and metrics are defined in `docs/model_performance.md`. The job verifies
every image hash, prepares images exactly like the app (EXIF upright, outward-rounded
crops, metadata-free PNG, 1536 px limit), and selects the best new version using
validation data only. It never changes registry aliases or the serving endpoint.

From the `evaluation/` directory, after listing the versions and any reviewed app
submissions in `workload_compare.yaml`:

```powershell
air run -f workload_compare.yaml --dry-run -p YOUR_PROFILE
air run -f workload_compare.yaml -p YOUR_PROFILE
```

The job writes `result.json` and `runs.jsonl` / `metrics.jsonl` / `predictions.jsonl`
to `/Volumes/dbx_ai_app/airplane_tail_id/training_data/evaluations/<eval_run_id>/`.
Create the tables once with `app_setup/create_eval_tables.sql`, then publish from the
repository root:

```powershell
python evaluation/publish.py --eval-run-id <eval_run_id> --profile YOUR_PROFILE --warehouse-id <warehouse id>
```

Publishing replaces that comparison's rows and verifies the row counts. Results
appear in the Tailspot app's **Model performance** tab and in
`notebooks/04_model_performance.py` (`notebooks/validate_performance.json` runs it on
serverless). Promotion is a manual decision: move the `demo` alias, update
`serving/endpoint.json`, redeploy the endpoint and rerun the serving smoke test.
