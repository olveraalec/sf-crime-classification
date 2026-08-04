from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.inference_engine import (
    InferenceResponse,
    ModelInfo,
)
from src.prediction_service import RankedPrediction


@dataclass
class FakeInferenceEngine:
    """Fast deterministic engine double for API tests."""

    def predict_one(
        self,
        incident,
        *,
        top_k: int = 3,
    ) -> InferenceResponse:
        del incident

        predictions = (
            RankedPrediction(
                rank=1,
                crime_category="LARCENY/THEFT",
                probability=0.70,
            ),
            RankedPrediction(
                rank=2,
                crime_category="ROBBERY",
                probability=0.20,
            ),
            RankedPrediction(
                rank=3,
                crime_category="ASSAULT",
                probability=0.10,
            ),
        )

        return InferenceResponse(
            input_position=0,
            predicted_class="LARCENY/THEFT",
            predicted_probability=0.70,
            top_predictions=predictions[:top_k],
            inference_time_ms=1.25,
        )

    def predict_batch(
        self,
        incidents,
        *,
        top_k: int = 3,
    ) -> list[InferenceResponse]:
        incident_list = list(incidents)

        return [
            InferenceResponse(
                input_position=index,
                predicted_class="LARCENY/THEFT",
                predicted_probability=0.70,
                top_predictions=(
                    RankedPrediction(
                        rank=1,
                        crime_category="LARCENY/THEFT",
                        probability=0.70,
                    ),
                )[:top_k],
                inference_time_ms=0.75,
            )
            for index, _ in enumerate(incident_list)
        ]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="xgboost_finalist",
            algorithm="xgboost",
            trained_at_utc="2026-07-24T20:00:00+00:00",
            class_count=39,
            raw_feature_columns=30,
            transformed_feature_columns=70,
            frozen_test_log_loss=2.202763,
        )


def build_client() -> TestClient:
    app = create_app(
        engine_factory=FakeInferenceEngine,
    )

    return TestClient(app)


VALID_INCIDENT = {
    "incident_timestamp": "2015-05-13T23:53:00",
    "pd_district": "NORTHERN",
    "address": "OAK ST / LAGUNA ST",
    "longitude": -122.425892,
    "latitude": 37.774599,
}


def test_health_endpoint_reports_loaded_model() -> None:
    with build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "sf-crime-classification-api",
        "model_loaded": True,
    }


def test_model_info_endpoint() -> None:
    with build_client() as client:
        response = client.get("/model/info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_name"] == "xgboost_finalist"
    assert body["algorithm"] == "xgboost"
    assert body["class_count"] == 39
    assert body["raw_feature_columns"] == 30
    assert body["transformed_feature_columns"] == 70
    assert body["frozen_test_log_loss"] == 2.202763


def test_single_prediction_endpoint() -> None:
    with build_client() as client:
        response = client.post(
            "/predictions",
            json={
                "incident": VALID_INCIDENT,
                "top_k": 2,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_class"] == "LARCENY/THEFT"
    assert body["predicted_probability"] == 0.70
    assert len(body["top_predictions"]) == 2
    assert body["top_predictions"][0]["rank"] == 1

    snapshot = client.app.state.metrics_registry.snapshot()

    assert snapshot.successful_prediction_operations == 1
    assert snapshot.failed_prediction_operations == 0
    assert snapshot.records_processed == 1


def test_batch_prediction_endpoint() -> None:
    with build_client() as client:
        response = client.post(
            "/predictions/batch",
            json={
                "incidents": [
                    VALID_INCIDENT,
                    {
                        **VALID_INCIDENT,
                        "pd_district": "MISSION",
                        "address": "100 BLOCK OF MISSION ST",
                    },
                ],
                "top_k": 1,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction_count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["input_position"] == 0
    assert body["predictions"][1]["input_position"] == 1

    snapshot = client.app.state.metrics_registry.snapshot()

    assert snapshot.successful_prediction_operations == 1
    assert snapshot.failed_prediction_operations == 0
    assert snapshot.records_processed == 2
    assert snapshot.average_inference_latency_ms >= 0


def test_prediction_rejects_invalid_coordinate() -> None:
    invalid_incident = {
        **VALID_INCIDENT,
        "longitude": -181,
    }

    with build_client() as client:
        response = client.post(
            "/predictions",
            json={
                "incident": invalid_incident,
            },
        )

    assert response.status_code == 422


def test_prediction_records_inference_engine_failure() -> None:
    @dataclass
    class FailingInferenceEngine(FakeInferenceEngine):
        def predict_one(
            self,
            incident,
            *,
            top_k: int = 3,
        ) -> InferenceResponse:
            del incident, top_k
            raise RuntimeError("Synthetic inference failure.")

    app = create_app(
        engine_factory=FailingInferenceEngine,
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/predictions",
            json={
                "incident": VALID_INCIDENT,
                "top_k": 2,
            },
        )

        snapshot = client.app.state.metrics_registry.snapshot()

    assert response.status_code == 500
    assert snapshot.successful_prediction_operations == 0
    assert snapshot.failed_prediction_operations == 1
    assert snapshot.records_processed == 0


def test_prediction_rejects_unknown_request_field() -> None:
    invalid_incident = {
        **VALID_INCIDENT,
        "unexpected_field": "not allowed",
    }

    with build_client() as client:
        response = client.post(
            "/predictions",
            json={
                "incident": invalid_incident,
            },
        )

    assert response.status_code == 422


def test_batch_rejects_empty_incident_list() -> None:
    with build_client() as client:
        response = client.post(
            "/predictions/batch",
            json={
                "incidents": [],
            },
        )

    assert response.status_code == 422


def test_openapi_schema_is_available() -> None:
    with build_client() as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/health/live" in paths
    assert "/health/ready" in paths
    assert "/model/info" in paths
    assert "/predictions" in paths
    assert "/predictions/batch" in paths


def test_response_includes_request_id_header() -> None:
    with build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]


def test_each_request_receives_unique_request_id() -> None:
    with build_client() as client:
        first = client.get("/health")
        second = client.get("/health")

    first_request_id = first.headers["X-Request-ID"]
    second_request_id = second.headers["X-Request-ID"]

    assert first_request_id
    assert second_request_id
    assert first_request_id != second_request_id


def test_error_response_includes_request_id() -> None:
    invalid_incident = {
        **VALID_INCIDENT,
        "longitude": -181,
    }

    with build_client() as client:
        response = client.post(
            "/predictions",
            json={
                "incident": invalid_incident,
            },
        )

    assert response.status_code == 422

    request_id = response.headers["X-Request-ID"]
    body = response.json()

    assert request_id
    assert body["request_id"] == request_id