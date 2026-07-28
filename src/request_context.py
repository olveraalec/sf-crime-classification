from __future__ import annotations

from contextvars import ContextVar, Token


_request_id_context: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)


def set_request_id(
    request_id: str,
) -> Token[str | None]:
    """Store a request ID in the current execution context."""
    return _request_id_context.set(request_id)


def get_request_id() -> str | None:
    """Return the request ID associated with the current context."""
    return _request_id_context.get()


def reset_request_id(
    token: Token[str | None],
) -> None:
    """Restore the previous request-context value."""
    _request_id_context.reset(token)