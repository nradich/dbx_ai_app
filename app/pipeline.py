"""Tailspot submission processing: validate, store, identify, look up and record.

Databricks access goes through `WorkspaceClient`-shaped objects so the logic can
be tested with fakes. User input never enters SQL text: every value is a bound
parameter, and only the validated schema name is interpolated as an identifier.

Timing: each Databricks call is bounded by client timeouts, inference retries stop
starting new attempts after INFERENCE_DEADLINE_S, and SQL polling is capped, so a
job always reaches a terminal state well within the browser's polling window.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
from typing import Callable
import uuid

from PIL import Image

import schedule
import serving_client

log = logging.getLogger("tailspot")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
CROP_MAX_SIDE = 1536
ALLOWED_FORMATS = frozenset({"JPEG", "MPO", "PNG"})
INFERENCE_DEADLINE_S = 600       # no new inference attempt starts after this
BACKOFF_S = (5, 10, 15, 20, 30)
MIN_ATTEMPT_WINDOW_S = 10
SQL_WAIT_S = 120                 # polling cap for background statements
STATUS_SQL_WAIT = "10s"          # status fallback never polls beyond this
WORKERS = 4
MAX_IN_FLIGHT = WORKERS          # no queueing behind long-running jobs
PREPARE_CONCURRENCY = 2          # bounds peak memory for large photos
STORE_LIMIT = 1000
NOT_FOUND_CACHE_S = 30
MERGE_ATTEMPTS = 3

TIMEOUT_MESSAGE = ("The model did not respond in time. It may still be starting; "
                   "please try again in a few minutes.")
MODEL_FAILED_MESSAGE = "The model could not process this photo."
MODEL_UNAVAILABLE_MESSAGE = "The model is unavailable right now. Please try again later."
SCHEDULE_FAILED_MESSAGE = "Possible departures could not be loaded."
NOT_RECORDED_MESSAGE = "The result could not be saved."
UNEXPECTED_MESSAGE = "Something went wrong while processing this photo."
RESTARTED_MESSAGE = "The app restarted before this photo was processed. Please submit it again."
FIRST_ATTEMPT_MESSAGE = ("Identifying the airline. If the model has been idle, "
                         "starting it can take a few minutes.")
RETRY_MESSAGE = "Waking the model. The first request after a quiet period can take a few minutes."

_SCHEMA_PATTERN = re.compile(r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+")
_VOLUME_PATTERN = re.compile(r"/Volumes/[A-Za-z0-9_]+/[A-Za-z0-9_]+/[A-Za-z0-9_]+")

# Columns of photo_submissions (app_setup/create_storage.sql) and their SQL types.
SUBMISSION_COLUMNS = (
    ("submission_id", "STRING"), ("submitted_at", "TIMESTAMP"), ("completed_at", "TIMESTAMP"),
    ("status", "STRING"), ("error", "STRING"), ("airport_iata", "STRING"),
    ("observed_local", "STRING"), ("observed_at", "TIMESTAMP"), ("upload_format", "STRING"),
    ("upload_bytes", "BIGINT"), ("upload_sha256", "STRING"), ("original_path", "STRING"),
    ("original_sha256", "STRING"), ("original_width", "INT"), ("original_height", "INT"),
    ("crop_left", "DOUBLE"), ("crop_top", "DOUBLE"), ("crop_right", "DOUBLE"),
    ("crop_bottom", "DOUBLE"), ("crop_path", "STRING"), ("crop_sha256", "STRING"),
    ("crop_width", "INT"), ("crop_height", "INT"), ("crop_downscaled", "BOOLEAN"),
    ("serving_endpoint", "STRING"), ("served_entity", "STRING"), ("model_name", "STRING"),
    ("model_version", "STRING"), ("inference_attempts", "INT"), ("inference_seconds", "DOUBLE"),
    ("predicted_airline", "STRING"), ("confidence", "DOUBLE"), ("probabilities_json", "STRING"),
    ("candidate_count", "INT"), ("candidates_json", "STRING"), ("review_label", "STRING"),
    ("reviewed_at", "TIMESTAMP"),
)


class UserError(Exception):
    """A problem to report to the browser with an HTTP status and a readable message."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class InferenceError(Exception):
    """Inference failed; `message` is safe to show to users."""

    def __init__(self, message: str, attempts: int, seconds: float):
        super().__init__(message)
        self.message = message
        self.attempts = attempts
        self.seconds = seconds


@dataclass(frozen=True)
class Settings:
    """Deployment configuration, normally from app.yaml environment variables."""

    serving_endpoint: str = serving_client.DEFAULT_ENDPOINT
    warehouse_id: str | None = None
    schema: str = "dbx_ai_app.airplane_tail_id"
    volume_path: str = "/Volumes/dbx_ai_app/airplane_tail_id/submitted_photos"

    def __post_init__(self):
        if not _SCHEMA_PATTERN.fullmatch(self.schema):
            raise ValueError("TAILSPOT_SCHEMA must look like catalog.schema")
        if not _VOLUME_PATTERN.fullmatch(self.volume_path):
            raise ValueError("TAILSPOT_VOLUME_PATH must look like /Volumes/catalog/schema/volume")
        if not self.serving_endpoint:
            raise ValueError("SERVING_ENDPOINT must not be empty")

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(serving_endpoint=os.environ.get("SERVING_ENDPOINT") or defaults.serving_endpoint,
                   warehouse_id=os.environ.get("DATABRICKS_WAREHOUSE_ID") or None,
                   schema=os.environ.get("TAILSPOT_SCHEMA") or defaults.schema,
                   volume_path=os.environ.get("TAILSPOT_VOLUME_PATH") or defaults.volume_path)

    def table(self, name: str) -> str:
        """Backtick-quoted three-part table name."""
        return ".".join(f"`{part}`" for part in (*self.schema.split("."), name))


@dataclass(frozen=True)
class PreparedUpload:
    """Validated upload and the two PNG artifacts derived from it."""

    upload_format: str
    upload_bytes: int
    upload_sha256: str
    crop_box: tuple
    original_png: bytes
    original_sha256: str
    original_width: int
    original_height: int
    crop_png: bytes
    crop_sha256: str
    crop_width: int
    crop_height: int
    crop_downscaled: bool


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_crop_box(left, top, right, bottom) -> tuple:
    """Parse the four form values; range checks happen against the decoded image."""
    try:
        box = tuple(float(value) for value in (left, top, right, bottom))
    except (TypeError, ValueError):
        raise UserError(422, "The tail box is invalid. Draw a box around the tail.") from None
    if not all(math.isfinite(value) for value in box):
        raise UserError(422, "The tail box is invalid. Draw a box around the tail.")
    return box


def parse_seen_size(width, height) -> tuple[int, int] | None:
    """Optional upright image size the browser displayed; both or neither must be given."""
    if width in (None, "") and height in (None, ""):
        return None
    try:
        size = int(width), int(height)
    except (TypeError, ValueError):
        raise UserError(422, "The photo size sent by the browser is invalid.") from None
    if size[0] <= 0 or size[1] <= 0:
        raise UserError(422, "The photo size sent by the browser is invalid.")
    return size


def prepare_upload(data: bytes, crop_box: tuple,
                   seen_size: tuple[int, int] | None = None) -> PreparedUpload:
    """Validate a JPEG/PNG upload and build the upright original and the model's tail crop.

    `seen_size` is the upright size the browser displayed; a mismatch means the
    browser and server disagree about orientation, so the box would be misplaced.
    """
    if not data:
        raise UserError(422, "Choose a photo to upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UserError(413, "The photo is larger than 20 MB. Choose a smaller photo.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            upload_format, (width, height) = probe.format, probe.size
    except Image.DecompressionBombError:
        raise UserError(422, "The photo has too many pixels. Choose a smaller photo.") from None
    except Exception:
        raise UserError(422, "Upload a JPEG or PNG photo.") from None
    if upload_format not in ALLOWED_FORMATS:
        raise UserError(422, "Upload a JPEG or PNG photo.")
    if width * height > MAX_PIXELS:
        raise UserError(422, "The photo has too many pixels. Choose a smaller photo.")
    try:
        rgb = serving_client.oriented_rgb(data)
    except Exception:
        raise UserError(422, "The photo could not be decoded. Choose another JPEG or PNG.") from None
    if seen_size is not None and tuple(seen_size) != rgb.size:
        raise UserError(422, "Your browser shows this photo with a different orientation than the "
                             "server reads. Try another photo, or save it again as a JPEG.")
    try:
        left, top, right, bottom = serving_client.crop_pixels(rgb.size, crop_box)
    except ValueError:
        raise UserError(422, "The tail box must lie within the photo and be at least "
                             "two pixels wide and tall.") from None
    crop = serving_client.tail_crop(rgb, crop_box, CROP_MAX_SIDE)
    crop_png = serving_client.png_bytes(crop)
    original_png = serving_client.png_bytes(rgb)
    return PreparedUpload(
        upload_format=upload_format, upload_bytes=len(data), upload_sha256=sha256(data),
        crop_box=tuple(crop_box), original_png=original_png, original_sha256=sha256(original_png),
        original_width=rgb.width, original_height=rgb.height, crop_png=crop_png,
        crop_sha256=sha256(crop_png), crop_width=crop.width, crop_height=crop.height,
        crop_downscaled=crop.size != (right - left, bottom - top))


def storage_paths(settings: Settings, submission_id: str, submitted_at: datetime) -> tuple[str, str]:
    """Volume paths for the upright original and the tail crop, grouped by UTC date."""
    folder = f"{settings.volume_path}/{submitted_at.astimezone(timezone.utc):%Y/%m/%d}/{submission_id}"
    return f"{folder}/original.png", f"{folder}/tail_crop.png"


def execute_sql(w, warehouse_id: str | None, statement: str, params: dict | None = None, *,
                wait_timeout: str = "50s", max_wait_s: float = SQL_WAIT_S,
                sleep: Callable[[float], None] = time.sleep,
                monotonic: Callable[[], float] = time.monotonic) -> list[dict]:
    """Run one parameterized statement on a SQL warehouse and return rows as dicts of strings."""
    from databricks.sdk.service.sql import StatementParameterListItem, StatementState

    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not configured")
    parameters = [StatementParameterListItem(name=name, value=None if value is None else str(value))
                  for name, value in (params or {}).items()]
    api = w.statement_execution
    response = api.execute_statement(statement=statement, warehouse_id=warehouse_id,
                                     wait_timeout=wait_timeout, parameters=parameters)
    deadline = monotonic() + max_wait_s
    while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if monotonic() >= deadline:
            try:
                api.cancel_execution(response.statement_id)
            except Exception:
                log.warning("Could not cancel statement %s", response.statement_id, exc_info=True)
            raise TimeoutError("The SQL statement did not finish in time")
        sleep(2)
        response = api.get_statement(response.statement_id)
    if response.status.state != StatementState.SUCCEEDED:
        error = response.status.error
        raise RuntimeError(f"SQL statement {response.status.state.value}: "
                           f"{getattr(error, 'message', None) or error}")
    result = response.result
    if result is None or response.manifest is None or response.manifest.schema is None:
        return []
    rows = list(result.data_array or [])
    while result.next_chunk_index is not None:
        result = api.get_statement_result_chunk_n(response.statement_id, result.next_chunk_index)
        rows.extend(result.data_array or [])
    columns = [column.name for column in response.manifest.schema.columns]
    return [dict(zip(columns, row)) for row in rows]


def _parse_prediction(response) -> dict:
    predictions = getattr(response, "predictions", None)
    if not predictions:
        raise ValueError("The endpoint returned no predictions")
    row = predictions[0]
    airline, confidence = row["airline"], row["confidence"]
    probabilities = json.loads(row["probabilities_json"])

    def numeric(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    if not isinstance(airline, str) or not airline or not numeric(confidence):
        raise ValueError("The endpoint returned an invalid prediction")
    if (not isinstance(probabilities, dict)
            or not all(isinstance(k, str) and numeric(v) for k, v in probabilities.items())):
        raise ValueError("The endpoint returned invalid probabilities")
    return {"airline": airline, "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in probabilities.items()}}


def _failure_message(exc: Exception) -> str | None:
    """User message for a non-retryable inference failure, or None if it may be retried."""
    from databricks.sdk import errors

    if isinstance(exc, (errors.PermissionDenied, errors.Unauthenticated, errors.NotFound)):
        return MODEL_UNAVAILABLE_MESSAGE
    if isinstance(exc, (errors.BadRequest, ValueError, KeyError, TypeError)):
        return MODEL_FAILED_MESSAGE
    return None


def infer(w, endpoint: str, crop_png: bytes, *, on_attempt: Callable[[int], None] = lambda n: None,
          sleep: Callable[[float], None] = time.sleep,
          monotonic: Callable[[], float] = time.monotonic) -> tuple[dict, str | None, int, float]:
    """Query the endpoint, retrying cold-start and transient failures until the deadline.

    Returns (prediction, served_model_name, attempts, seconds). Raises InferenceError.
    """
    record = [{"image_base64": base64.b64encode(crop_png).decode("ascii")}]
    started = monotonic()
    attempt = 0
    while True:
        attempt += 1
        on_attempt(attempt)
        try:
            response = w.serving_endpoints.query(name=endpoint, dataframe_records=record)
            prediction = _parse_prediction(response)
            return (prediction, getattr(response, "served_model_name", None), attempt,
                    monotonic() - started)
        except Exception as exc:
            message = _failure_message(exc)
            if message is not None:
                log.exception("Non-retryable inference failure on attempt %d", attempt)
                raise InferenceError(message, attempt, monotonic() - started) from None
            log.warning("Inference attempt %d failed; retrying if time allows", attempt,
                        exc_info=True)
        elapsed = monotonic() - started
        delay = BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)]
        if elapsed + delay + MIN_ATTEMPT_WINDOW_S > INFERENCE_DEADLINE_S:
            raise InferenceError(TIMEOUT_MESSAGE, attempt, elapsed)
        sleep(delay)


def empty_model(endpoint: str) -> dict:
    return {"endpoint": endpoint, "served_entity": None, "model_name": None, "model_version": None}


def model_metadata(w, endpoint: str, served_name: str | None) -> dict:
    """Describe the served model that answered; never raises."""
    meta = empty_model(endpoint)
    meta["served_entity"] = served_name
    try:
        config = w.serving_endpoints.get(endpoint).config
        entities = list((config.served_entities if config else None) or [])
        if served_name:
            entities = [entity for entity in entities if entity.name == served_name]
        if len(entities) == 1:
            entity = entities[0]
            meta.update(served_entity=entity.name, model_name=entity.entity_name,
                        model_version=entity.entity_version)
    except Exception:
        log.warning("Could not read serving endpoint metadata", exc_info=True)
    return meta


def lookup_schedule(w, settings: Settings, airline: str, observed_at: datetime, **sql_kwargs):
    """Return (airline_name, iata_code, candidates) for a predicted airline."""
    rows = execute_sql(w, settings.warehouse_id, (
        "SELECT a.airline_name, a.iata_code, d.departure_id, d.flight_number, "
        "d.destination_iata, d.destination_city, d.day_of_week, d.scheduled_departure_local "
        f"FROM {settings.table('demo_airlines')} a "
        f"LEFT JOIN {settings.table('demo_weekly_departures')} d "
        "ON d.airline_id = a.airline_id AND d.origin_iata = :origin "
        "WHERE a.airline_id = :airline"), {"origin": schedule.AIRPORT_IATA, "airline": airline},
        **sql_kwargs)
    if not rows:
        return None, None, []
    departures = [row for row in rows if row.get("departure_id") is not None]
    return rows[0].get("airline_name"), rows[0].get("iata_code"), schedule.match_departures(
        departures, observed_at)


def insert_submission(w, settings: Settings, values: dict, **sql_kwargs) -> None:
    """Record one photo_submissions row; idempotent if the statement is retried."""
    names = [name for name, _ in SUBMISSION_COLUMNS]
    if set(values) != set(names):
        raise ValueError(f"Submission row columns differ: {sorted(set(values) ^ set(names))}")
    params = {}
    for name, sql_type in SUBMISSION_COLUMNS:
        value = values[name]
        if value is not None and sql_type == "TIMESTAMP":
            value = value.astimezone(timezone.utc).isoformat()
        elif value is not None and sql_type == "BOOLEAN":
            value = "true" if value else "false"
        elif value is not None and sql_type == "DOUBLE":
            value = repr(float(value))
        elif value is not None and sql_type in ("INT", "BIGINT"):
            value = str(int(value))
        params[name] = value
    source = ", ".join(f"CAST(:{name} AS {sql_type}) AS {name}" for name, sql_type in SUBMISSION_COLUMNS)
    statement = (f"MERGE INTO {settings.table('photo_submissions')} AS t "
                 f"USING (SELECT {source}) AS s ON t.submission_id = s.submission_id "
                 "WHEN NOT MATCHED THEN INSERT *")
    sleep = sql_kwargs.get("sleep", time.sleep)
    for attempt in range(1, MERGE_ATTEMPTS + 1):
        try:
            execute_sql(w, settings.warehouse_id, statement, params, **sql_kwargs)
            return
        except RuntimeError as exc:
            # The MERGE is idempotent, so a concurrent-write conflict can simply be retried.
            if attempt == MERGE_ATTEMPTS or "CONCURRENT" not in str(exc).upper():
                raise
            log.warning("Concurrent write while recording; retrying (%d)", attempt)
            sleep(2 * attempt)


def _epoch_to_iso(value) -> str | None:
    return None if value is None else schedule.iso_z(
        datetime.fromtimestamp(int(value), tz=timezone.utc))


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def _default_client(purpose: str):
    """Databricks client using the app's service principal (or local CLI auth)."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.config import Config

    if purpose == "files":
        # Keep uploads well inside the 120 s app proxy limit for the POST request.
        return WorkspaceClient(config=Config(http_timeout_seconds=45, retry_timeout_seconds=30))
    # A cold serving endpoint held a request open for about 90 s; `infer` owns retries.
    return WorkspaceClient(config=Config(http_timeout_seconds=150, retry_timeout_seconds=20))


class SubmissionService:
    """Accepts submissions, processes them in background threads and reports status."""

    def __init__(self, settings: Settings | None = None,
                 client_factory: Callable[[str], object] | None = None, *,
                 executor=None, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.settings = settings or Settings.from_env()
        self._client_factory = client_factory or _default_client
        self._clients: dict[str, object] = {}
        self._executor = executor or ThreadPoolExecutor(max_workers=WORKERS,
                                                        thread_name_prefix="tailspot")
        self._sleep, self._monotonic, self._now = sleep, monotonic, now
        self._lock = threading.Lock()
        self._prepare_slots = threading.BoundedSemaphore(PREPARE_CONCURRENCY)
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._not_found: dict[str, float] = {}
        self._active = 0

    # Databricks clients ------------------------------------------------------
    def client(self, purpose: str = "api"):
        """Create Databricks clients on first use (never at import or startup)."""
        if purpose not in self._clients:
            created = self._client_factory(purpose)
            with self._lock:
                self._clients.setdefault(purpose, created)
        return self._clients[purpose]

    def _sql(self, **overrides) -> dict:
        return {"sleep": self._sleep, "monotonic": self._monotonic, **overrides}

    # Public API --------------------------------------------------------------
    def submit(self, data: bytes, crop_box: tuple, observed_local: str,
               seen_size: tuple[int, int] | None = None) -> dict:
        """Validate and store a submission, then queue background processing."""
        try:
            observed_text, observed_at = schedule.parse_observed_local(observed_local, self._now())
        except ValueError as exc:
            raise UserError(422, str(exc)) from None
        with self._lock:
            if self._active >= MAX_IN_FLIGHT:
                raise UserError(503, "Tailspot is busy with other photos. Please try again shortly.")
            self._active += 1
        queued = False
        try:
            submission_id = str(uuid.uuid4())
            with self._prepare_slots:
                prepared = prepare_upload(data, crop_box, seen_size)
                submitted_at = self._now()
                original_path, crop_path = storage_paths(self.settings, submission_id, submitted_at)
                self._store_files(submission_id, prepared, original_path, crop_path)
            prepared = replace(prepared, original_png=b"")  # only the crop is needed from here
            public = {
                "submission_id": submission_id, "status": "queued", "stage": "queued",
                "message": "Photo saved. Waiting to start.", "inference_attempts": 0,
                "submitted_at": schedule.iso_z(submitted_at), "observed_local": observed_text,
                "observed_at": schedule.iso_z(observed_at), "airport_iata": schedule.AIRPORT_IATA,
                "prediction": None, "model": None, "candidates": [], "schedule_lookup": "pending",
                "storage": {"original_path": original_path, "crop_path": crop_path,
                            "crop_width": prepared.crop_width, "crop_height": prepared.crop_height,
                            "crop_downscaled": prepared.crop_downscaled},
                "recorded": False, "error": None,
            }
            private = {"prepared": prepared, "submitted_at": submitted_at,
                       "observed_at": observed_at, "recorded": False}
            with self._lock:
                self._jobs[submission_id] = {"public": public, "private": private, "started": False}
                self._evict_locked()
            try:
                self._executor.submit(self._run, submission_id)
            except Exception:
                log.exception("Could not queue submission %s", submission_id)
                with self._lock:
                    self._jobs.pop(submission_id, None)
                raise UserError(503, "Tailspot is restarting. Please try again shortly.") from None
            queued = True
            log.info("Accepted submission %s", submission_id)
            return {"submission_id": submission_id, "status": "queued",
                    "status_url": f"/api/submissions/{submission_id}"}
        finally:
            if not queued:
                with self._lock:
                    self._active -= 1

    def get(self, submission_id: str) -> dict | None:
        """Return the submission status, falling back to the table after a restart."""
        try:
            submission_id = str(uuid.UUID(submission_id))
        except (TypeError, ValueError, AttributeError):
            return None
        with self._lock:
            job = self._jobs.get(submission_id)
            if job is not None:
                return copy.deepcopy(job["public"])
            missing_since = self._not_found.get(submission_id)
            if missing_since is not None and self._monotonic() - missing_since < NOT_FOUND_CACHE_S:
                return None
        status = self._load_recorded(submission_id)
        with self._lock:
            if status is None:
                self._not_found[submission_id] = self._monotonic()
                if len(self._not_found) > STORE_LIMIT:
                    self._not_found.pop(next(iter(self._not_found)))
            elif submission_id not in self._jobs:
                self._jobs[submission_id] = {"public": copy.deepcopy(status), "private": None,
                                             "started": True}
                self._evict_locked()
        return status

    def shutdown(self) -> None:
        """Stop accepting work; jobs that never started are reported as failed."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for job in self._jobs.values():
                if not job["started"] and job["public"]["status"] == "queued":
                    job["public"].update(status="failed", stage="done", message=RESTARTED_MESSAGE,
                                         error=RESTARTED_MESSAGE)

    # Storage -----------------------------------------------------------------
    def _store_files(self, submission_id: str, prepared: PreparedUpload, original_path: str,
                     crop_path: str) -> None:
        try:
            files = self.client("files").files
            files.upload(original_path, io.BytesIO(prepared.original_png), overwrite=True)
        except Exception:
            log.exception("Could not store the original for submission %s", submission_id)
            raise UserError(502, "The photo could not be saved. Please try again.") from None
        try:
            files.upload(crop_path, io.BytesIO(prepared.crop_png), overwrite=True)
        except Exception:
            log.exception("Could not store the tail crop for submission %s", submission_id)
            try:
                files.delete(original_path)
            except Exception:
                log.warning("Could not remove orphaned original %s", original_path, exc_info=True)
            raise UserError(502, "The photo could not be saved. Please try again.") from None

    # Background processing ---------------------------------------------------
    def _update(self, submission_id: str, **changes) -> None:
        with self._lock:
            self._jobs[submission_id]["public"].update(copy.deepcopy(changes))

    def _evict_locked(self) -> None:
        excess = len(self._jobs) - STORE_LIMIT
        if excess <= 0:
            return
        finished = [key for key, job in self._jobs.items()
                    if job["public"]["status"] in ("succeeded", "failed")]
        for key in finished[:excess]:
            del self._jobs[key]

    def _run(self, submission_id: str) -> None:
        with self._lock:
            self._jobs[submission_id]["started"] = True
        try:
            self._process(submission_id)
        except Exception:
            log.exception("Unexpected failure processing submission %s", submission_id)
            self._fail_unexpectedly(submission_id)
        finally:
            with self._lock:
                self._active -= 1
                job = self._jobs.get(submission_id)  # may already be evicted once finished
                if job is not None:
                    job["private"] = None  # release image bytes

    def _fail_unexpectedly(self, submission_id: str) -> None:
        with self._lock:
            job = self._jobs[submission_id]
            private, public = job["private"], copy.deepcopy(job["public"])
        recorded = public["recorded"] or bool(private and private["recorded"])
        if private is not None and not recorded:
            model = public["model"] or empty_model(self.settings.serving_endpoint)
            recorded = self._record(submission_id, private, "failed", UNEXPECTED_MESSAGE, model,
                                    None, None, public["inference_attempts"], None)
        self._update(submission_id, status="failed", stage="done", recorded=recorded,
                     message=UNEXPECTED_MESSAGE,
                     error=UNEXPECTED_MESSAGE if recorded else
                     f"{UNEXPECTED_MESSAGE} {NOT_RECORDED_MESSAGE}")

    def _process(self, submission_id: str) -> None:
        with self._lock:
            private = self._jobs[submission_id]["private"]
        prepared: PreparedUpload = private["prepared"]
        w = self.client()
        endpoint = self.settings.serving_endpoint
        self._update(submission_id, status="processing", stage="waiting_for_model",
                     message=FIRST_ATTEMPT_MESSAGE, model=empty_model(endpoint))

        def on_attempt(attempt: int) -> None:
            self._update(submission_id, inference_attempts=attempt,
                         message=FIRST_ATTEMPT_MESSAGE if attempt == 1 else RETRY_MESSAGE)

        prediction, model, candidates, error = None, empty_model(endpoint), None, None
        lookup = "skipped"
        try:
            raw, served_name, attempts, seconds = infer(
                w, endpoint, prepared.crop_png, on_attempt=on_attempt,
                sleep=self._sleep, monotonic=self._monotonic)
        except InferenceError as exc:
            attempts, seconds, error = exc.attempts, exc.seconds, exc.message
        else:
            model = model_metadata(w, endpoint, served_name)
            prediction = {"airline": raw["airline"], "airline_name": None, "iata_code": None,
                          "confidence": raw["confidence"], "probabilities": raw["probabilities"]}
            self._update(submission_id, stage="looking_up_schedule", model=model,
                         prediction=prediction, message="Checking the San Diego schedule.")
            try:
                name, code, candidates = lookup_schedule(
                    w, self.settings, raw["airline"], private["observed_at"], **self._sql())
                prediction.update(airline_name=name, iata_code=code)
                lookup = "ok"
            except Exception:
                log.exception("Schedule lookup failed for submission %s", submission_id)
                candidates, error, lookup = None, SCHEDULE_FAILED_MESSAGE, "failed"
        status = "succeeded" if prediction is not None else "failed"
        self._update(submission_id, stage="recording", message="Saving the result.",
                     inference_attempts=attempts, prediction=prediction, model=model,
                     candidates=candidates or [], schedule_lookup=lookup)
        recorded = self._record(submission_id, private, status, error, model, prediction,
                                candidates, attempts, seconds)
        if not recorded:
            error = f"{error} {NOT_RECORDED_MESSAGE}" if error else NOT_RECORDED_MESSAGE
        self._update(submission_id, status=status, stage="done", recorded=recorded, error=error,
                     message=error if status == "failed" else "Done.")
        log.info("Submission %s %s (recorded=%s)", submission_id, status, recorded)

    def _record(self, submission_id, private, status, error, model, prediction, candidates,
                attempts, seconds) -> bool:
        """Write the table row; `candidates` is None when the lookup failed or was skipped.

        Never raises: returns whether the row was written.
        """
        try:
            values = self._row_values(submission_id, private, status, error, model, prediction,
                                      candidates, attempts, seconds)
            insert_submission(self.client(), self.settings, values, **self._sql())
        except Exception:
            log.exception("Could not record submission %s", submission_id)
            return False
        private["recorded"] = True
        return True

    def _row_values(self, submission_id, private, status, error, model, prediction, candidates,
                    attempts, seconds) -> dict:
        prepared: PreparedUpload = private["prepared"]
        with self._lock:
            public = copy.deepcopy(self._jobs[submission_id]["public"])
        left, top, right, bottom = prepared.crop_box
        return {
            "submission_id": submission_id, "submitted_at": private["submitted_at"],
            "completed_at": self._now(), "status": status, "error": error,
            "airport_iata": schedule.AIRPORT_IATA, "observed_local": public["observed_local"],
            "observed_at": private["observed_at"], "upload_format": prepared.upload_format,
            "upload_bytes": prepared.upload_bytes, "upload_sha256": prepared.upload_sha256,
            "original_path": public["storage"]["original_path"],
            "original_sha256": prepared.original_sha256,
            "original_width": prepared.original_width, "original_height": prepared.original_height,
            "crop_left": left, "crop_top": top, "crop_right": right, "crop_bottom": bottom,
            "crop_path": public["storage"]["crop_path"], "crop_sha256": prepared.crop_sha256,
            "crop_width": prepared.crop_width, "crop_height": prepared.crop_height,
            "crop_downscaled": prepared.crop_downscaled,
            "serving_endpoint": self.settings.serving_endpoint,
            "served_entity": model.get("served_entity"), "model_name": model.get("model_name"),
            "model_version": model.get("model_version"), "inference_attempts": attempts,
            "inference_seconds": seconds,
            "predicted_airline": prediction["airline"] if prediction else None,
            "confidence": prediction["confidence"] if prediction else None,
            "probabilities_json": (json.dumps(prediction["probabilities"], sort_keys=True)
                                   if prediction else None),
            "candidate_count": None if candidates is None else len(candidates),
            "candidates_json": None if candidates is None else json.dumps(candidates, sort_keys=True),
            "review_label": None, "reviewed_at": None,
        }

    # Status after restart ----------------------------------------------------
    def _load_recorded(self, submission_id: str) -> dict | None:
        rows = execute_sql(self.client(), self.settings.warehouse_id, (
            "SELECT s.submission_id, s.status, s.error, s.airport_iata, s.observed_local, "
            "unix_timestamp(s.submitted_at) AS submitted_epoch, "
            "unix_timestamp(s.observed_at) AS observed_epoch, s.original_path, s.crop_path, "
            "s.crop_width, s.crop_height, s.crop_downscaled, s.serving_endpoint, s.served_entity, "
            "s.model_name, s.model_version, s.inference_attempts, s.predicted_airline, "
            "s.confidence, s.probabilities_json, s.candidates_json, a.airline_name, a.iata_code "
            f"FROM {self.settings.table('photo_submissions')} s "
            f"LEFT JOIN {self.settings.table('demo_airlines')} a "
            "ON a.airline_id = s.predicted_airline "
            "WHERE s.submission_id = :submission_id LIMIT 1"),
            {"submission_id": submission_id},
            **self._sql(wait_timeout=STATUS_SQL_WAIT, max_wait_s=0))
        if not rows:
            return None
        row = rows[0]
        prediction = None
        if row.get("predicted_airline") is not None:
            prediction = {"airline": row["predicted_airline"], "airline_name": row.get("airline_name"),
                          "iata_code": row.get("iata_code"),
                          "confidence": _optional_float(row.get("confidence")),
                          "probabilities": json.loads(row.get("probabilities_json") or "{}")}
        candidates_json = row.get("candidates_json")
        lookup = "skipped" if prediction is None else ("failed" if candidates_json is None else "ok")
        failed = row.get("status") == "failed"
        return {
            "submission_id": row["submission_id"], "status": row.get("status"), "stage": "done",
            "message": row.get("error") if failed else "Done.",
            "inference_attempts": _optional_int(row.get("inference_attempts")) or 0,
            "submitted_at": _epoch_to_iso(row.get("submitted_epoch")),
            "observed_local": row.get("observed_local"),
            "observed_at": _epoch_to_iso(row.get("observed_epoch")),
            "airport_iata": row.get("airport_iata"), "prediction": prediction,
            "model": {"endpoint": row.get("serving_endpoint"),
                      "served_entity": row.get("served_entity"),
                      "model_name": row.get("model_name"), "model_version": row.get("model_version")},
            "candidates": json.loads(candidates_json) if candidates_json else [],
            "schedule_lookup": lookup,
            "storage": {"original_path": row.get("original_path"), "crop_path": row.get("crop_path"),
                        "crop_width": _optional_int(row.get("crop_width")),
                        "crop_height": _optional_int(row.get("crop_height")),
                        "crop_downscaled": str(row.get("crop_downscaled")).lower() == "true"},
            "recorded": True, "error": row.get("error"),
        }
