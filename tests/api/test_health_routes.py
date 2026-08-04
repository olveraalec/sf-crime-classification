from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import AppSettings
from src.inference_engine import ModelInfo


TEST_SETTINGS = AppSettings(
    api_title="Test Crime API",
    api_description="Health-route tests.",
    api_version="3.0.0",
    service_name="test-crime-service",
    model_name="test-model",
    log_level="INFO",
    log_filename="test-health.log",
    server_host="127.0.0.1",
    server_port=8000,
)


@dataclass
class ReadyInferenceEngine:
    """Minimal healthy engine double for readiness testing."""

    def predict_one(
        self,
        incident,
        *,
        top_k: int = 3,
    ):
        raise NotImplementedError

    def predict_batch(
        self,
        incidents,
        *,
        top_k: int = 3,
    ):
        raise NotImplementedError

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="test-model",
            algorithm="xgboost",
            trained_at_utc="2026-07-28T20:00:00+00:00",
            class_count=39,
            raw_feature_columns=30,
            transformed_feature_columns=70,
            frozen_test_log_loss=2.202763,
        )


@dataclass
class BrokenMetadataEngine(ReadyInferenceEngine):
    """Engine double whose model metadata cannot be read."""

    def get_model_info(self) -> ModelInfo:
        raise RuntimeError("Synthetic metadata failure.")


def build_client(
    engine_factory=ReadyInferenceEngine,
) -> TestClient:
    app = create_app(
        engine_factory=engine_factory,
        settings=TEST_SETTINGS,
    )

    return TestClient(app)


def test_legacy_health_endpoint_remains_compatible() -> None:
    with build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "test-crime-service",
        "model_loaded": True,
    }


def test_liveness_reports_alive() -> None:
    with build_client() as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "alive",
        "service": "test-crime-service",
    }


def test_readiness_reports_ready_service() -> None:
    with build_client() as client:
        response = client.get("/health/ready")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["ready"] is True
    assert body["detail"] is None
    assert all(body["checks"].values())


def test_readiness_fails_when_engine_is_missing() -> None:
    with build_client() as client:
        del client.app.state.inference_engine

        response = client.get("/health/ready")

    assert response.status_code == 503

    body = response.json()

    assert body["status"] == "not_ready"
    assert body["ready"] is False
    assert body["checks"]["inference_engine"] is False
    assert body["checks"]["model_metadata"] is False


def test_readiness_fails_when_metrics_registry_is_missing() -> None:
    with build_client() as client:
        del client.app.state.metrics_registry

        response = client.get("/health/ready")

    assert response.status_code == 503

    body = response.json()

    assert body["ready"] is False
    assert body["checks"]["metrics_registry"] is False


def test_readiness_fails_when_model_metadata_is_broken() -> None:
    with build_client(
        engine_factory=BrokenMetadataEngine,
    ) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503

    body = response.json()

    assert body["ready"] is False
    assert body["checks"]["inference_engine"] is True
    assert body["checks"]["model_metadata"] is False
    assert "RuntimeError" in body["detail"]


def test_liveness_remains_healthy_when_readiness_fails() -> None:
    with build_client() as client:
        del client.app.state.inference_engine

        live_response = client.get("/health/live")
        ready_response = client.get("/health/ready")

    assert live_response.status_code == 200
    assert live_response.json()["status"] == "alive"

    assert ready_response.status_code == 503
    assert ready_response.json()["status"] == "not_ready"