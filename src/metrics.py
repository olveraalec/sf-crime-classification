from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class MetricsSnapshot:
    """Read-only snapshot of the service's current operational metrics."""

    total_http_requests: int
    successful_http_requests: int
    failed_http_requests: int
    average_http_latency_ms: float

    successful_prediction_operations: int
    failed_prediction_operations: int
    records_processed: int
    average_inference_latency_ms: float

    prediction_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert the snapshot into a JSON-compatible dictionary."""
        return asdict(self)


class MetricsRegistry:
    """Own and update in-memory operational metrics for the API."""

    def __init__(self) -> None:
        self._lock = Lock()

        self._total_http_requests = 0
        self._successful_http_requests = 0
        self._failed_http_requests = 0
        self._total_http_latency_ms = 0.0

        self._successful_prediction_operations = 0
        self._failed_prediction_operations = 0
        self._records_processed = 0
        self._total_inference_latency_ms = 0.0

        self._prediction_counts: Counter[str] = Counter()

    def record_http_request(
        self,
        *,
        duration_ms: float,
        status_code: int,
    ) -> None:
        """Record one completed HTTP request."""
        if duration_ms < 0:
            raise ValueError("HTTP request duration cannot be negative.")

        if not 100 <= status_code <= 599:
            raise ValueError("HTTP status code must be between 100 and 599.")

        with self._lock:
            self._total_http_requests += 1
            self._total_http_latency_ms += duration_ms

            if status_code < 400:
                self._successful_http_requests += 1
            else:
                self._failed_http_requests += 1

    def record_prediction(
        self,
        *,
        record_count: int,
        inference_time_ms: float,
        predicted_classes: list[str],
    ) -> None:
        """Record one successful model invocation."""
        if record_count < 1:
            raise ValueError("Record count must be at least one.")

        if inference_time_ms < 0:
            raise ValueError("Inference time cannot be negative.")

        if len(predicted_classes) != record_count:
            raise ValueError("Predicted class count must match record count.")

        if any(not category.strip() for category in predicted_classes):
            raise ValueError("Predicted classes cannot contain empty values.")

        with self._lock:
            self._successful_prediction_operations += 1
            self._records_processed += record_count
            self._total_inference_latency_ms += inference_time_ms
            self._prediction_counts.update(predicted_classes)

    def record_prediction_failure(self) -> None:
        """Record one failed model invocation."""
        with self._lock:
            self._failed_prediction_operations += 1

    def snapshot(self) -> MetricsSnapshot:
        """Return a consistent read-only view of current metrics."""
        with self._lock:
            average_http_latency_ms = self._calculate_average(
                total=self._total_http_latency_ms,
                count=self._total_http_requests,
            )

            average_inference_latency_ms = self._calculate_average(
                total=self._total_inference_latency_ms,
                count=self._successful_prediction_operations,
            )

            return MetricsSnapshot(
                total_http_requests=self._total_http_requests,
                successful_http_requests=(self._successful_http_requests),
                failed_http_requests=self._failed_http_requests,
                average_http_latency_ms=average_http_latency_ms,
                successful_prediction_operations=(
                    self._successful_prediction_operations
                ),
                failed_prediction_operations=(self._failed_prediction_operations),
                records_processed=self._records_processed,
                average_inference_latency_ms=(average_inference_latency_ms),
                prediction_counts=dict(self._prediction_counts),
            )

    @staticmethod
    def _calculate_average(
        *,
        total: float,
        count: int,
    ) -> float:
        """Calculate an average without dividing by zero."""
        if count == 0:
            return 0.0

        return total / count