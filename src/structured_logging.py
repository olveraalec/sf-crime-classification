from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.request_context import get_request_id


def build_log_event(
    event: str,
    **fields: Any,
) -> dict[str, Any]:
    """Build a structured operational log event."""
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }

    request_id = get_request_id()

    if request_id is not None:
        payload["request_id"] = request_id

    payload.update(fields)

    return payload


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Serialize and emit one structured JSON log event."""
    payload = build_log_event(
        event,
        **fields,
    )

    logger.log(
        level,
        json.dumps(
            payload,
            default=str,
            sort_keys=True,
        ),
    )