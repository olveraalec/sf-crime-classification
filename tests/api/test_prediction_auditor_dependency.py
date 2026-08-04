from __future__ import annotations

from fastapi import Request
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_prediction_auditor
from src.prediction_auditor import PredictionAuditor


def build_client() -> TestClient:
    app = create_app(
        engine_factory=lambda: object(),  # type: ignore[arg-type]
    )
    return TestClient(app)


def build_request(app) -> Request:
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


def test_prediction_auditor_initialized_during_startup() -> None:
    with build_client() as client:
        auditor = client.app.state.prediction_auditor

        assert isinstance(
            auditor,
            PredictionAuditor,
        )


def test_prediction_auditor_dependency_returns_shared_instance() -> None:
    with build_client() as client:
        app_auditor = client.app.state.prediction_auditor
        request = build_request(client.app)

        dependency_auditor = get_prediction_auditor(request)

        assert dependency_auditor is app_auditor


def test_prediction_auditor_uses_shared_metrics_registry() -> None:
    with build_client() as client:
        auditor = client.app.state.prediction_auditor
        registry = client.app.state.metrics_registry

        assert auditor._metrics_registry is registry


def test_prediction_auditor_dependency_fails_before_startup() -> None:
    app = create_app(
        engine_factory=lambda: object(),  # type: ignore[arg-type]
    )

    request = build_request(app)

    try:
        get_prediction_auditor(request)
    except RuntimeError as error:
        assert str(error) == ("Prediction auditor has not been initialized.")
    else:
        raise AssertionError("Expected prediction auditor dependency to fail.")