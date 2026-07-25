from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from src.artifact_loader import (
    ArtifactBundle,
    load_artifact_bundle,
)
from src.incident_adapter import (
    IncidentAdapterError,
    RawIncident,
    build_modeling_frame,
)
from src.logger import get_logger
from src.prediction_service import (
    PredictionResult,
    PredictionService,
    PredictionServiceError,
    RankedPrediction,
)


logger = get_logger(__name__)


class InferenceEngineError(RuntimeError):
    """Raised when the public inference workflow cannot complete."""


@dataclass(frozen=True)
class InferenceResponse:
    """Framework-independent prediction response for one incident."""

    input_position: int
    predicted_class: str
    predicted_probability: float
    top_predictions: tuple[RankedPrediction, ...]
    inference_time_ms: float


@dataclass(frozen=True)
class ModelInfo:
    """Stable model information exposed to delivery layers."""

    model_name: str
    algorithm: str
    trained_at_utc: str | None
    class_count: int
    raw_feature_columns: int
    transformed_feature_columns: int
    frozen_test_log_loss: float | None


class InferenceEngine:
    """
    Public interface for raw incident inference.

    This class coordinates raw-input adaptation and the lower-level
    prediction service while remaining independent of FastAPI or any
    other delivery framework.
    """

    def __init__(
        self,
        prediction_service: PredictionService,
    ) -> None:
        self._prediction_service = prediction_service

    @classmethod
    def from_artifacts(
        cls,
        model_directory: Path | None = None,
    ) -> InferenceEngine:
        """Create a ready-to-use engine from persisted model artifacts."""
        bundle = load_artifact_bundle(model_directory=model_directory)

        service = PredictionService(bundle)

        return cls(service)

    @property
    def prediction_service(self) -> PredictionService:
        """Return the lower-level prediction service."""
        return self._prediction_service

    def predict_one(
        self,
        incident: RawIncident,
        *,
        top_k: int = 3,
    ) -> InferenceResponse:
        """Run inference for one raw incident."""
        responses = self.predict_batch(
            [incident],
            top_k=top_k,
        )

        return responses[0]

    def predict_batch(
        self,
        incidents: Iterable[RawIncident],
        *,
        top_k: int = 3,
    ) -> list[InferenceResponse]:
        """Run one vectorized inference call for multiple raw incidents."""
        if top_k < 1:
            raise InferenceEngineError("top_k must be at least 1.")

        incident_list = list(incidents)

        if not incident_list:
            raise InferenceEngineError("Inference requires at least one incident.")

        start_time = perf_counter()

        try:
            modeling_frame = build_modeling_frame(incident_list)

            prediction_results = self._prediction_service.predict(
                modeling_frame,
                top_k=top_k,
            )
        except (
            IncidentAdapterError,
            PredictionServiceError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as error:
            raise InferenceEngineError("Inference could not be completed.") from error

        if len(prediction_results) != len(incident_list):
            raise InferenceEngineError(
                "Prediction result count does not match the number of input incidents."
            )

        elapsed_ms = (perf_counter() - start_time) * 1000.0

        per_record_time_ms = elapsed_ms / len(incident_list)

        responses = [
            self._build_response(
                input_position=input_position,
                prediction_result=prediction_result,
                inference_time_ms=per_record_time_ms,
            )
            for input_position, prediction_result in enumerate(prediction_results)
        ]

        logger.info(
            "Completed raw inference for %s incident(s) in %.3f ms.",
            len(responses),
            elapsed_ms,
        )

        return responses

    @staticmethod
    def _build_response(
        *,
        input_position: int,
        prediction_result: PredictionResult,
        inference_time_ms: float,
    ) -> InferenceResponse:
        return InferenceResponse(
            input_position=input_position,
            predicted_class=(prediction_result.predicted_class),
            predicted_probability=(prediction_result.predicted_probability),
            top_predictions=(prediction_result.top_predictions),
            inference_time_ms=float(inference_time_ms),
        )

    def get_model_info(self) -> ModelInfo:
        """Return selected metadata without exposing internal artifacts."""
        bundle: ArtifactBundle = self._prediction_service.artifact_bundle

        metadata = bundle.metadata
        config = metadata.get("config")

        algorithm = (
            config.get("model_name", "unknown")
            if isinstance(config, dict)
            else "unknown"
        )

        return ModelInfo(
            model_name=str(
                metadata.get(
                    "model_name",
                    "unknown",
                )
            ),
            algorithm=str(algorithm),
            trained_at_utc=self._optional_string(metadata.get("trained_at_utc")),
            class_count=self._required_int(
                metadata,
                "class_count",
            ),
            raw_feature_columns=self._required_int(
                metadata,
                "raw_feature_columns",
            ),
            transformed_feature_columns=(
                self._required_int(
                    metadata,
                    "transformed_feature_columns",
                )
            ),
            frozen_test_log_loss=self._optional_float(
                metadata.get("frozen_test_log_loss")
            ),
        )

    @staticmethod
    def _required_int(
        metadata: dict[str, Any],
        field_name: str,
    ) -> int:
        value = metadata.get(field_name)

        if not isinstance(value, int):
            raise InferenceEngineError(
                f"Model metadata field '{field_name}' must be an integer."
            )

        return value

    @staticmethod
    def _optional_string(
        value: object,
    ) -> str | None:
        if value is None:
            return None

        return str(value)

    @staticmethod
    def _optional_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise InferenceEngineError(
                "Model metadata contains an invalid floating-point value."
            ) from error