from __future__ import annotations

import json
import logging

from src.request_context import (
    reset_request_id,
    set_request_id,
)
from src.structured_logging import (
    build_log_event,
    log_event,
)


def test_build_log_event_includes_event_fields() -> None:
    payload = build_log_event(
        "prediction_completed",
        batch_size=3,
        success=True,
    )

    assert payload["event"] == "prediction_completed"
    assert payload["batch_size"] == 3
    assert payload["success"] is True
    assert "timestamp_utc" in payload


def test_build_log_event_includes_context_request_id() -> None:
    token = set_request_id("request-abc")

    try:
        payload = build_log_event("http_request_completed")
    finally:
        reset_request_id(token)

    assert payload["request_id"] == "request-abc"


def test_log_event_emits_valid_json(
    caplog,
) -> None:
    logger = logging.getLogger("structured-log-test")

    with caplog.at_level(
        logging.INFO,
        logger=logger.name,
    ):
        log_event(
            logger,
            "http_request_completed",
            status_code=200,
            duration_ms=4.5,
        )

    payload = json.loads(caplog.records[-1].message)

    assert payload["event"] == "http_request_completed"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 4.5