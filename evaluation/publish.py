"""Load one comparison's JSONL outputs (evaluation/compare.py) into Unity Catalog tables.

    python evaluation/publish.py --eval-run-id <32-hex MLflow run ID> --profile dev_kit \
        --warehouse-id <id> [--volume-root /Volumes/.../evaluations] [--schema catalog.schema]

Steps (all statements run on the SQL warehouse via the Statement Execution API, polled to
completion; any failure stops the publish):
1. Download runs/metrics/predictions JSONL from the Volume; check every row has exactly the
   table's columns and this eval_run_id; count rows.
2. Check read_files() sees the same row count with no foreign eval_run_id.
3. DELETE + INSERT per table in the order: hide (DELETE runs), replace predictions,
   replace metrics, reveal (INSERT runs) - so readers that start from model_eval_runs never
   see a run whose metrics are missing. Rerunning is idempotent.
4. Verify each table holds exactly the JSONL row count for this eval_run_id.

The eval_run_id is bound as a named parameter; paths and table names are inlined only after
strict regex validation (read_files needs a literal path).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_metrics as em  # noqa: E402

DEFAULT_VOLUME_ROOT = "/Volumes/dbx_ai_app/airplane_tail_id/training_data/evaluations"
DEFAULT_SCHEMA = "dbx_ai_app.airplane_tail_id"
EVAL_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
VOLUME_ROOT = re.compile(r"^/Volumes(/[A-Za-z0-9_]+){3}(/[A-Za-z0-9_-]+)*$")
SCHEMA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
WAREHOUSE_ID = re.compile(r"^[A-Za-z0-9]{1,64}$")
PROFILE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"}

# (table, file, [(column, SQL type)]) - must match app_setup/create_eval_tables.sql.
TABLES = {
    "model_eval_runs": ("runs.jsonl", list(zip(em.RUN_COLUMNS, (
        "STRING", "TIMESTAMP", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
        "DOUBLE", "DOUBLE", "STRING", "BOOLEAN", "BOOLEAN", "STRING")))),
    "model_eval_metrics": ("metrics.jsonl", list(zip(em.METRIC_COLUMNS, (
        "STRING", "STRING", "STRING", "STRING", "STRING", "DOUBLE", "INT")))),
    "model_eval_predictions": ("predictions.jsonl", list(zip(em.PREDICTION_COLUMNS, (
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "DOUBLE", "DOUBLE",
        "STRING", "BOOLEAN", "STRING")))),
}
# Replacement order: hide the run, replace its detail rows, then reveal the run.
STEPS = (("delete", "model_eval_runs"),
         ("delete", "model_eval_predictions"), ("insert", "model_eval_predictions"),
         ("delete", "model_eval_metrics"), ("insert", "model_eval_metrics"),
         ("insert", "model_eval_runs"))


class PublishError(RuntimeError):
    pass


def validate(eval_run_id, volume_root, schema, warehouse_id=None):
    if not isinstance(eval_run_id, str) or not EVAL_RUN_ID.match(eval_run_id):
        raise PublishError("eval run ID must be 32 lowercase hex characters")
    if not isinstance(volume_root, str) or not VOLUME_ROOT.match(volume_root):
        raise PublishError("volume root must look like /Volumes/<catalog>/<schema>/<volume>[/<dir>...]")
    if not isinstance(schema, str) or not SCHEMA.match(schema):
        raise PublishError("schema must be <catalog>.<schema>")
    if warehouse_id is not None and (not isinstance(warehouse_id, str) or not WAREHOUSE_ID.match(warehouse_id)):
        raise PublishError("warehouse ID must be alphanumeric")


def file_path(volume_root, eval_run_id, table):
    return f"{volume_root}/{eval_run_id}/{TABLES[table][0]}"


def _read_files(volume_root, eval_run_id, table):
    columns = TABLES[table][1]
    read_schema = ", ".join(f"{name} {'STRING' if kind == 'TIMESTAMP' else kind}" for name, kind in columns)
    return (f"read_files('{file_path(volume_root, eval_run_id, table)}', format => 'json', "
            f"schema => '{read_schema}')")


def statements(eval_run_id, volume_root, schema):
    """Ordered [(kind, table, sql)]; every statement binds :eval_run_id. Validates inputs."""
    validate(eval_run_id, volume_root, schema)
    result = []
    for table in TABLES:
        result.append(("source_count", table,
                       f"SELECT count(*) AS n, count_if(eval_run_id IS NULL OR eval_run_id <> :eval_run_id) "
                       f"AS foreign_rows FROM {_read_files(volume_root, eval_run_id, table)}"))
    for kind, table in STEPS:
        target = f"{schema}.{table}"
        if kind == "delete":
            sql = f"DELETE FROM {target} WHERE eval_run_id = :eval_run_id"
        else:
            columns = TABLES[table][1]
            names = ", ".join(name for name, _ in columns)
            casts = ", ".join(f"CAST({name} AS {sql_type}) AS {name}" for name, sql_type in columns)
            sql = (f"INSERT INTO {target} ({names}) SELECT {casts} FROM "
                   f"{_read_files(volume_root, eval_run_id, table)} WHERE eval_run_id = :eval_run_id")
        result.append((kind, table, sql))
    for table in TABLES:
        result.append(("verify", table,
                       f"SELECT count(*) AS n FROM {schema}.{table} WHERE eval_run_id = :eval_run_id"))
    for _, _, sql in result:
        if ";" in sql:
            raise PublishError("statement must not contain ';'")
    return result


def _state(response):
    state = getattr(getattr(response, "status", None), "state", None)
    return getattr(state, "value", state)


def execute(client, warehouse_id, sql, eval_run_id, timeout_s=900, poll_s=2.0, sleep=time.sleep,
            clock=time.monotonic):
    """Run one statement and poll until terminal; return result rows (list of lists of str)."""
    from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout, StatementParameterListItem

    api = client.statement_execution
    response = api.execute_statement(
        statement=sql, warehouse_id=warehouse_id, wait_timeout="30s",
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
        parameters=[StatementParameterListItem(name="eval_run_id", value=eval_run_id, type="STRING")])
    deadline = clock() + timeout_s
    while _state(response) not in TERMINAL:
        if clock() > deadline:
            api.cancel_execution(response.statement_id)
            raise PublishError(f"statement timed out after {timeout_s}s: {sql[:120]}")
        sleep(poll_s)
        response = api.get_statement(response.statement_id)
    if _state(response) != "SUCCEEDED":
        error = getattr(response.status, "error", None)
        message = getattr(error, "message", None) or _state(response)
        raise PublishError(f"statement failed ({message}): {sql[:200]}")
    result = getattr(response, "result", None)
    return (getattr(result, "data_array", None) or []) if result is not None else []


def check_jsonl(text, table, eval_run_id):
    """Row count of a JSONL file after checking keys and eval_run_id of every row."""
    columns = {name for name, _ in TABLES[table][1]}
    count = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != columns:
            raise PublishError(f"{TABLES[table][0]} line {number}: columns differ from {table}")
        if row["eval_run_id"] != eval_run_id:
            raise PublishError(f"{TABLES[table][0]} line {number}: eval_run_id differs")
        count += 1
    if count == 0:
        raise PublishError(f"{TABLES[table][0]} is empty")
    return count


def publish(client, eval_run_id, warehouse_id, volume_root=DEFAULT_VOLUME_ROOT, schema=DEFAULT_SCHEMA,
            log=print, **execute_options):
    validate(eval_run_id, volume_root, schema, warehouse_id)
    plan = statements(eval_run_id, volume_root, schema)
    expected = {}
    for table in TABLES:
        download = client.files.download(file_path(volume_root, eval_run_id, table))
        expected[table] = check_jsonl(download.contents.read().decode("utf-8"), table, eval_run_id)
    summary = {"eval_run_id": eval_run_id, "schema": schema, "tables": {}}
    for kind, table, sql in plan:
        rows = execute(client, warehouse_id, sql, eval_run_id, **execute_options)
        if kind == "source_count":
            n, foreign = int(rows[0][0]), int(rows[0][1])
            if n != expected[table] or foreign:
                raise PublishError(f"read_files sees {n} rows ({foreign} foreign) for {table}; "
                                   f"file has {expected[table]}")
        elif kind == "verify":
            n = int(rows[0][0])
            if n != expected[table]:
                raise PublishError(f"{schema}.{table} holds {n} rows for {eval_run_id}, expected {expected[table]}")
            summary["tables"][table] = n
        log(f"{kind:12s} {table}: ok")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--volume-root", default=DEFAULT_VOLUME_ROOT)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)
    try:
        validate(args.eval_run_id, args.volume_root, args.schema, args.warehouse_id)
        if not PROFILE.match(args.profile):
            raise PublishError("profile name has unexpected characters")
        from databricks.sdk import WorkspaceClient

        summary = publish(WorkspaceClient(profile=args.profile), args.eval_run_id, args.warehouse_id,
                          args.volume_root, args.schema)
    except PublishError as exc:
        parser.exit(1, f"Publish failed: {exc}\n")
    print("PUBLISH_RESULT=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
