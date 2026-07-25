from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.artifact_loader import ArtifactBundle
from src.incident_adapter import RawIncident
from src.inference_engine import (
    InferenceEngine,
    InferenceEngineError,
    InferenceResponse,
)
from src.prediction_service import (
    PredictionResult,
    RankedPrediction,
)


class FakePredictionService:
    """Small prediction-service double for inference-engine tests."""

    def __init__(
        self,
        *,
        results: list[PredictionResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.received_frame = None
        self.received_top_k: int | None = None

    def predict(
        self,
        modeling_data,
        *,
        top_k: int = 3,
    ) -> list[PredictionResult]:
        self.received_frame = modeling_data.copy()
        self.received_top_k = top_k

        if self.error is not None:
            raise self.error

        return self.results


def make_incident(
    *,
    district: str = "NORTHERN",
    address: str = "OAK ST / LAGUNA ST",
) -> RawIncident:
    return RawIncident(
        incident_timestamp=datetime(
            2015,
            5,
            13,
            23,
            53,
        ),
        pd_district=district,
        address=address,
        longitude=-122.425892,
        latitude=37.774599,
    )


def make_prediction_result(
    *,
    row_index: int = 0,
    predicted_class: str = "LARCENY/THEFT",
    predicted_probability: float = 0.70,
) -> PredictionResult:
    return PredictionResult(
        row_index=row_index,
        predicted_class=predicted_class,
        predicted_probability=predicted_probability,
        top_predictions=(
            RankedPrediction(
                rank=1,
                crime_category=predicted_class,
                probability=predicted_probability,
            ),
            RankedPrediction(
                rank=2,
                crime_category="ROBBERY",
                probability=0.20,
            ),
        ),
    )


def test_predict_one_returns_single_inference_response() -> None:
    service = FakePredictionService(results=[make_prediction_result()])
    engine = InferenceEngine(service)

    response = engine.predict_one(
        make_incident(),
        top_k=2,
    )

    assert isinstance(response, InferenceResponse)
    assert response.predicted_class == "LARCENY/THEFT"
    assert response.predicted_probability == pytest.approx(0.70)
    assert len(response.top_predictions) == 2
    assert response.input_position == 0
    assert response.inference_time_ms >= 0

    assert service.received_top_k == 2
    assert service.received_frame.shape == (1, 17)


def test_predict_batch_preserves_input_order() -> None:
    service = FakePredictionService(
        results=[
            make_prediction_result(
                row_index=0,
                predicted_class="ASSAULT",
                predicted_probability=0.60,
            ),
            make_prediction_result(
                row_index=1,
                predicted_class="ROBBERY",
                predicted_probability=0.55,
            ),
        ]
    )

    engine = InferenceEngine(service)

    responses = engine.predict_batch(
        [
            make_incident(
                district="NORTHERN",
            ),
            make_incident(
                district="MISSION",
                address="100 BLOCK OF MISSION ST",
            ),
        ],
        top_k=2,
    )

    assert [item.input_position for item in responses] == [0, 1]
    assert [item.predicted_class for item in responses] == [
        "ASSAULT",
        "ROBBERY",
    ]

    assert service.received_frame["pd_district"].tolist() == [
        "NORTHERN",
        "MISSION",
    ]


def test_batch_uses_one_prediction_service_call() -> None:
    service = FakePredictionService(
        results=[
            make_prediction_result(row_index=0),
            make_prediction_result(row_index=1),
        ]
    )

    engine = InferenceEngine(service)

    engine.predict_batch(
        [
            make_incident(),
            make_incident(
                district="MISSION",
            ),
        ]
    )

    assert service.received_frame.shape[0] == 2


def test_predict_batch_rejects_empty_input() -> None:
    engine = InferenceEngine(FakePredictionService())

    with pytest.raises(
        InferenceEngineError,
        match="at least one incident",
    ):
        engine.predict_batch([])


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_engine_rejects_invalid_top_k(
    top_k: int,
) -> None:
    engine = InferenceEngine(FakePredictionService())

    with pytest.raises(
        InferenceEngineError,
        match="top_k must be at least 1",
    ):
        engine.predict_one(
            make_incident(),
            top_k=top_k,
        )


def test_engine_translates_prediction_service_errors() -> None:
    service = FakePredictionService(error=RuntimeError("model failed"))
    engine = InferenceEngine(service)

    with pytest.raises(
        InferenceEngineError,
        match="Inference could not be completed",
    ):
        engine.predict_one(make_incident())


def test_engine_detects_result_count_mismatch() -> None:
    service = FakePredictionService(results=[])
    engine = InferenceEngine(service)

    with pytest.raises(
        InferenceEngineError,
        match="result count",
    ):
        engine.predict_one(make_incident())


def test_model_info_uses_artifact_metadata(
    tmp_path: Path,
) -> None:
    bundle = ArtifactBundle(
        model=object(),
        transformer=object(),
        label_encoder=object(),
        metadata={
            "model_name": "xgboost_finalist",
            "trained_at_utc": "2026-07-24T20:00:00+00:00",
            "class_count": 39,
            "raw_feature_columns": 30,
            "transformed_feature_columns": 70,
            "frozen_test_log_loss": 2.202763,
            "config": {
                "model_name": "xgboost",
            },
        },
        paths=object(),
    )

    class ServiceWithBundle(FakePredictionService):
        artifact_bundle = bundle

    engine = InferenceEngine(ServiceWithBundle())

    info = engine.get_model_info()

    assert info.model_name == "xgboost_finalist"
    assert info.algorithm == "xgboost"
    assert info.class_count == 39
    assert info.raw_feature_columns == 30
    assert info.transformed_feature_columns == 70
    assert info.frozen_test_log_loss == pytest.approx(2.202763)