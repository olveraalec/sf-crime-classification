from __future__ import annotations

from fastapi import Request
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_app_settings
from src.config import AppSettings


TEST_SETTINGS = AppSettings(
    api_title="Test Crime API",
    api_description="Test API description.",
    api_version="9.9.9",
    service_name="test-crime-api",
    model_name="test-model",
    log_level="INFO",
    log_filename="test.log",
    server_host="127.0.0.1",
    server_port=9000,
)


def build_client() -> TestClient:
    app = create_app(
        engine_factory=lambda: object(),  # type: ignore[arg-type]
        settings=TEST_SETTINGS,
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


def test_settings_initialized_during_startup() -> None:
    with build_client() as client:
        settings = client.app.state.settings

        assert settings is TEST_SETTINGS


def test_settings_dependency_returns_shared_instance() -> None:
    with build_client() as client:
        request = build_request(client.app)

        settings = get_app_settings(request)

        assert settings is TEST_SETTINGS


def test_settings_control_openapi_metadata() -> None:
    with build_client() as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    body = response.json()

    assert body["info"]["title"] == ("Test Crime API")
    assert body["info"]["description"] == ("Test API description.")
    assert body["info"]["version"] == "9.9.9"


def test_settings_dependency_fails_before_startup() -> None:
    app = create_app(
        engine_factory=lambda: object(),  # type: ignore[arg-type]
        settings=TEST_SETTINGS,
    )

    request = build_request(app)

    try:
        get_app_settings(request)
    except RuntimeError as error:
        assert str(error) == ("Application settings have not been initialized.")
    else:
        raise AssertionError(
            "Expected settings dependency to fail before application startup."
        )