from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from src.logger import get_logger
from src.metrics import MetricsRegistry
from src.structured_logging import log_event


logger = get_logger(__name__)


@dataclass(frozen=True)
class PredictionAuditEvent:
    """Operational facts describing one successful model invocation."""

    predicted_classes: tuple[str, ...]
    predicted_probabilities: tuple[float, ...]
    inference_time_ms: float
    model_name: str

    @property
    def record_count(self) -> int:
        """Return the number of records included in the operation."""
        return len(self.predicted_classes)

    @property
    def mean_confidence(self) -> float:
        """Return the mean winning-class probability."""
        if not self.predicted_probabilities:
            return 0.0

        return sum(self.predicted_probabilities) / len(self.predicted_probabilities)

    @classmethod
    def create(
        cls,
        *,
        predicted_classes: Sequence[str],
        predicted_probabilities: Sequence[float],
        inference_time_ms: float,
        model_name: str,
    ) -> PredictionAuditEvent:
        """Validate input and construct an immutable audit event."""
        classes = tuple(predicted_classes)
        probabilities = tuple(predicted_probabilities)

        if not classes:
            raise ValueError("Prediction audit event must contain at least one record.")

        if len(classes) != len(probabilities):
            raise ValueError(
                "Predicted classes and probabilities must have equal lengths."
            )

        if any(not category.strip() for category in classes):
            raise ValueError("Predicted classes cannot contain empty values.")

        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError("Predicted probabilities must be between zero and one.")

        if inference_time_ms < 0:
            raise ValueError("Inference time cannot be negative.")

        if not model_name.strip():
            raise ValueError("Model name cannot be empty.")

        return cls(
            predicted_classes=classes,
            predicted_probabilities=probabilities,
            inference_time_ms=inference_time_ms,
            model_name=model_name,
        )


class PredictionAuditor:
    """Coordinate prediction metrics and structured audit logging."""

    def __init__(
        self,
        *,
        metrics_registry: MetricsRegistry,
        audit_logger: logging.Logger = logger,
    ) -> None:
        self._metrics_registry = metrics_registry
        self._logger = audit_logger

    def record_prediction(
        self,
        event: PredictionAuditEvent,
    ) -> None:
        """Record one successful prediction operation."""
        self._metrics_registry.record_prediction(
            record_count=event.record_count,
            inference_time_ms=event.inference_time_ms,
            predicted_classes=list(event.predicted_classes),
        )

        log_event(
            self._logger,
            "prediction_completed",
            model_name=event.model_name,
            record_count=event.record_count,
            predicted_classes=list(event.predicted_classes),
            mean_confidence=round(
                event.mean_confidence,
                6,
            ),
            inference_time_ms=round(
                event.inference_time_ms,
                3,
            ),
        )

    def record_prediction_failure(
        self,
        *,
        model_name: str,
        error_type: str,
    ) -> None:
        """Record one failed prediction operation."""
        self._metrics_registry.record_prediction_failure()

        log_event(
            self._logger,
            "prediction_failed",
            level=logging.ERROR,
            model_name=model_name,
            error_type=error_type,
        )