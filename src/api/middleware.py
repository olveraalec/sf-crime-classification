from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from src.logger import get_logger
from src.request_context import (
    reset_request_id,
    set_request_id,
)
from src.structured_logging import log_event


logger = get_logger(__name__)


REQUEST_ID_HEADER = "X-Request-ID"


def register_request_middleware(
    app: FastAPI,
) -> None:
    """Register request identification and HTTP latency logging."""

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next,
    ) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id

        context_token = set_request_id(request_id)

        start_time = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - start_time) * 1000.0

            log_event(
                logger,
                "http_request_failed",
                level=logging.ERROR,
                method=request.method,
                path=request.url.path,
                duration_ms=round(
                    duration_ms,
                    3,
                ),
            )

            raise
        else:
            duration_ms = (perf_counter() - start_time) * 1000.0

            response.headers[REQUEST_ID_HEADER] = request_id

            log_event(
                logger,
                "http_request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(
                    duration_ms,
                    3,
                ),
            )

            return response
        finally:
            reset_request_id(context_token)