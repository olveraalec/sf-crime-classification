from __future__ import annotations

from src.request_context import (
    get_request_id,
    reset_request_id,
    set_request_id,
)


def test_request_context_defaults_to_none() -> None:
    assert get_request_id() is None


def test_request_context_stores_and_resets_value() -> None:
    token = set_request_id("request-123")

    assert get_request_id() == "request-123"

    reset_request_id(token)

    assert get_request_id() is None


def test_nested_request_context_restores_previous_value() -> None:
    outer_token = set_request_id("outer-request")
    inner_token = set_request_id("inner-request")

    assert get_request_id() == "inner-request"

    reset_request_id(inner_token)

    assert get_request_id() == "outer-request"

    reset_request_id(outer_token)

    assert get_request_id() is None