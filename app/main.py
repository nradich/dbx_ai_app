"""Tailspot Databricks App: FastAPI backend and static frontend.

The API contract is documented in API.md. The app authenticates with its own
service principal (injected by Databricks Apps); no user identity is stored.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline
import schedule

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("tailspot")

STATIC_DIR = Path(__file__).resolve().parent / "static"
# Multipart framing adds a little overhead beyond the photo itself.
MAX_REQUEST_BYTES = pipeline.MAX_UPLOAD_BYTES + 1024 * 1024
NO_STORE = {"Cache-Control": "no-store"}
TOO_LARGE = {"detail": "The photo is larger than 20 MB. Choose a smaller photo."}


class BodySizeLimit:
    """Reject request bodies over a limit, including chunked uploads without Content-Length."""

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length")
        if length is not None and (not length.isdigit() or int(length) > self.limit):
            return await self._reject(send)
        received, exceeded, started = 0, False, False

        async def limited_receive():
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.limit:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal started
            if exceeded and not started:
                started = True
                await self._reject(send)
            if exceeded:
                return
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await self._reject(send)

    @staticmethod
    async def _reject(send):
        body = json.dumps(TOO_LARGE).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def create_app(service: pipeline.SubmissionService | None = None) -> FastAPI:
    """Build the app; tests inject a service backed by fakes."""
    service = service or pipeline.SubmissionService()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        service.shutdown()

    app = FastAPI(title="Tailspot", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.service = service
    app.add_middleware(BodySizeLimit, limit=MAX_REQUEST_BYTES)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        fields = sorted({str(error["loc"][-1]) for error in exc.errors() if error.get("loc")})
        detail = ("Missing or invalid field: " + ", ".join(fields)) if fields else "Invalid request."
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(pipeline.UserError)
    async def user_error(_request: Request, exc: pipeline.UserError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config():
        return JSONResponse(headers=NO_STORE, content={
            "airport_iata": schedule.AIRPORT_IATA,
            "airport_name": schedule.AIRPORT_NAME,
            "time_zone": schedule.TIME_ZONE,
            "now_local": schedule.now_local_string(datetime.now(timezone.utc)),
            "max_upload_bytes": pipeline.MAX_UPLOAD_BYTES,
            "window_minutes_before": schedule.WINDOW_BEFORE_MIN,
            "window_minutes_after": schedule.WINDOW_AFTER_MIN,
        })

    @app.post("/api/submissions", status_code=202)
    def create_submission(photo: UploadFile = File(...), crop_left: str = Form(...),
                          crop_top: str = Form(...), crop_right: str = Form(...),
                          crop_bottom: str = Form(...), observed_local: str = Form(...),
                          image_width: str | None = Form(None),
                          image_height: str | None = Form(None)):
        data = photo.file.read(pipeline.MAX_UPLOAD_BYTES + 1)
        crop_box = pipeline.parse_crop_box(crop_left, crop_top, crop_right, crop_bottom)
        seen_size = pipeline.parse_seen_size(image_width, image_height)
        try:
            return service.submit(data, crop_box, observed_local, seen_size)
        except pipeline.UserError:
            raise
        except Exception:
            log.exception("Unexpected error accepting a submission")
            return JSONResponse(status_code=500,
                                content={"detail": "Something went wrong. Please try again."})

    @app.get("/api/submissions/{submission_id}")
    def get_submission(submission_id: str):
        try:
            status = service.get(submission_id)
        except Exception:
            log.exception("Could not load submission %s", submission_id)
            return JSONResponse(status_code=503, headers=NO_STORE, content={
                "detail": "The submission status is temporarily unavailable."})
        if status is None:
            return JSONResponse(status_code=404, headers=NO_STORE,
                                content={"detail": "Submission not found"})
        return JSONResponse(content=status, headers=NO_STORE)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True, check_dir=False), name="static")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
