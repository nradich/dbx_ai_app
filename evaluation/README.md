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
