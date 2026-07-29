from __future__ import annotations

from fastapi import Request
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_metrics_registry
from src.metrics import MetricsRegistry


def build_metrics_client() -> TestClient:
    """Create a test application without loading production artifacts."""
    app = create_app(
        engine_factory=lambda: object(),  # type: ignore[arg-type]
    )

    return TestClient(app)


def build_request_for_app(app) -> Request:
    """Construct a minimal request associated with the test application."""
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "app": app,
        }
    )


def test_metrics_registry_initialized_during_startup() -> None:
    """The application lifespan should create one metrics registry."""
    with build_metrics_client() as client:
        registry = getattr(
            client.app.state,
            "metrics_registry",
            None,
        )

        assert registry is not None
        assert isinstance(registry, MetricsRegistry)


def test_metrics_dependency_returns_shared_registry() -> None:
    """The dependency should return the registry owned by the app."""
    with build_metrics_client() as client:
        app_registry = client.app.state.metrics_registry

        request = build_request_for_app(client.app)
        dependency_registry = get_metrics_registry(request)

        assert dependency_registry is app_registry


def test_metrics_registry_remains_shared_across_requests() -> None:
    """Multiple requests should update the same registry instance."""
    with build_metrics_client() as client:
        original_registry = client.app.state.metrics_registry

        first_response = client.get("/health")
        second_response = client.get("/health")

        current_registry = client.app.state.metrics_registry
        snapshot = current_registry.snapshot()

        assert first_response.status_code == 200
        assert second_response.status_code == 200

        assert current_registry is original_registry
        assert snapshot.total_http_requests == 2
        assert snapshot.successful_http_requests == 2
        assert snapshot.failed_http_requests == 0


def test_failed_http_response_updates_failure_metrics() -> None:
    """A client error should count as a failed HTTP request."""
    with build_metrics_client() as client:
        response = client.get("/route-that-does-not-exist")

        snapshot = client.app.state.metrics_registry.snapshot()

        assert response.status_code == 404
        assert snapshot.total_http_requests == 1
        assert snapshot.successful_http_requests == 0
        assert snapshot.failed_http_requests == 1


def test_http_metrics_accumulate_latency() -> None:
    """Completed requests should contribute to HTTP latency metrics."""
    with build_metrics_client() as client:
        client.get("/health")
        client.get("/health")

        snapshot = client.app.state.metrics_registry.snapshot()

        assert snapshot.total_http_requests == 2
        assert snapshot.average_http_latency_ms >= 0.0


def test_metrics_dependency_fails_before_initialization() -> None:
    """Missing application state should produce an explicit error."""
    app = create_app(
        engine_factory=lambda: object(),  # type: ignore[arg-type]
    )

    # The lifespan has not started, so the registry does not exist yet.
    request = build_request_for_app(app)

    try:
        get_metrics_registry(request)
    except RuntimeError as error:
        assert str(error) == ("Metrics registry has not been initialized.")
    else:
        raise AssertionError("Expected get_metrics_registry() to raise RuntimeError.")