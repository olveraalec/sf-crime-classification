from __future__ import annotations

import pytest

from src.metrics import MetricsRegistry


def test_empty_registry_returns_zero_metrics() -> None:
    registry = MetricsRegistry()

    snapshot = registry.snapshot()

    assert snapshot.total_http_requests == 0
    assert snapshot.successful_http_requests == 0
    assert snapshot.failed_http_requests == 0
    assert snapshot.average_http_latency_ms == 0.0

    assert snapshot.successful_prediction_operations == 0
    assert snapshot.failed_prediction_operations == 0
    assert snapshot.records_processed == 0
    assert snapshot.average_inference_latency_ms == 0.0
    assert snapshot.prediction_counts == {}


def test_record_http_request_tracks_success() -> None:
    registry = MetricsRegistry()

    registry.record_http_request(
        duration_ms=10.0,
        status_code=200,
    )

    snapshot = registry.snapshot()

    assert snapshot.total_http_requests == 1
    assert snapshot.successful_http_requests == 1
    assert snapshot.failed_http_requests == 0
    assert snapshot.average_http_latency_ms == 10.0


def test_record_http_request_tracks_failure() -> None:
    registry = MetricsRegistry()

    registry.record_http_request(
        duration_ms=12.0,
        status_code=422,
    )

    snapshot = registry.snapshot()

    assert snapshot.total_http_requests == 1
    assert snapshot.successful_http_requests == 0
    assert snapshot.failed_http_requests == 1


def test_http_latency_uses_running_average() -> None:
    registry = MetricsRegistry()

    registry.record_http_request(
        duration_ms=10.0,
        status_code=200,
    )
    registry.record_http_request(
        duration_ms=20.0,
        status_code=200,
    )
    registry.record_http_request(
        duration_ms=30.0,
        status_code=500,
    )

    snapshot = registry.snapshot()

    assert snapshot.total_http_requests == 3
    assert snapshot.average_http_latency_ms == pytest.approx(20.0)


def test_record_single_prediction() -> None:
    registry = MetricsRegistry()

    registry.record_prediction(
        record_count=1,
        inference_time_ms=8.5,
        predicted_classes=["LARCENY/THEFT"],
    )

    snapshot = registry.snapshot()

    assert snapshot.successful_prediction_operations == 1
    assert snapshot.failed_prediction_operations == 0
    assert snapshot.records_processed == 1
    assert snapshot.average_inference_latency_ms == 8.5
    assert snapshot.prediction_counts == {
        "LARCENY/THEFT": 1,
    }


def test_record_batch_prediction() -> None:
    registry = MetricsRegistry()

    registry.record_prediction(
        record_count=3,
        inference_time_ms=15.0,
        predicted_classes=[
            "LARCENY/THEFT",
            "ASSAULT",
            "LARCENY/THEFT",
        ],
    )

    snapshot = registry.snapshot()

    assert snapshot.successful_prediction_operations == 1
    assert snapshot.records_processed == 3
    assert snapshot.prediction_counts == {
        "LARCENY/THEFT": 2,
        "ASSAULT": 1,
    }


def test_inference_latency_averages_model_operations() -> None:
    registry = MetricsRegistry()

    registry.record_prediction(
        record_count=1,
        inference_time_ms=10.0,
        predicted_classes=["ASSAULT"],
    )
    registry.record_prediction(
        record_count=4,
        inference_time_ms=30.0,
        predicted_classes=[
            "ASSAULT",
            "FRAUD",
            "FRAUD",
            "WARRANTS",
        ],
    )

    snapshot = registry.snapshot()

    assert snapshot.successful_prediction_operations == 2
    assert snapshot.records_processed == 5
    assert snapshot.average_inference_latency_ms == pytest.approx(20.0)


def test_record_prediction_failure() -> None:
    registry = MetricsRegistry()

    registry.record_prediction_failure()
    registry.record_prediction_failure()

    snapshot = registry.snapshot()

    assert snapshot.failed_prediction_operations == 2
    assert snapshot.successful_prediction_operations == 0
    assert snapshot.records_processed == 0


def test_snapshot_returns_independent_prediction_counts() -> None:
    registry = MetricsRegistry()

    registry.record_prediction(
        record_count=1,
        inference_time_ms=5.0,
        predicted_classes=["ASSAULT"],
    )

    first_snapshot = registry.snapshot()
    first_snapshot.prediction_counts["ASSAULT"] = 999

    second_snapshot = registry.snapshot()

    assert second_snapshot.prediction_counts["ASSAULT"] == 1


@pytest.mark.parametrize(
    ("duration_ms", "status_code"),
    [
        (-1.0, 200),
        (1.0, 99),
        (1.0, 600),
    ],
)
def test_invalid_http_metrics_are_rejected(
    duration_ms: float,
    status_code: int,
) -> None:
    registry = MetricsRegistry()

    with pytest.raises(ValueError):
        registry.record_http_request(
            duration_ms=duration_ms,
            status_code=status_code,
        )


def test_prediction_rejects_nonpositive_record_count() -> None:
    registry = MetricsRegistry()

    with pytest.raises(
        ValueError,
        match="Record count must be at least one",
    ):
        registry.record_prediction(
            record_count=0,
            inference_time_ms=1.0,
            predicted_classes=[],
        )


def test_prediction_rejects_negative_inference_time() -> None:
    registry = MetricsRegistry()

    with pytest.raises(
        ValueError,
        match="Inference time cannot be negative",
    ):
        registry.record_prediction(
            record_count=1,
            inference_time_ms=-1.0,
            predicted_classes=["ASSAULT"],
        )


def test_prediction_requires_one_class_per_record() -> None:
    registry = MetricsRegistry()

    with pytest.raises(
        ValueError,
        match="Predicted class count must match record count",
    ):
        registry.record_prediction(
            record_count=2,
            inference_time_ms=1.0,
            predicted_classes=["ASSAULT"],
        )


def test_snapshot_converts_to_dictionary() -> None:
    registry = MetricsRegistry()

    registry.record_http_request(
        duration_ms=5.0,
        status_code=200,
    )

    payload = registry.snapshot().to_dict()

    assert payload["total_http_requests"] == 1
    assert payload["average_http_latency_ms"] == 5.0
