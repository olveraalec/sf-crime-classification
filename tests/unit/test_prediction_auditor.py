from __future__ import annotations

import json
import logging

import pytest

from src.metrics import MetricsRegistry
from src.prediction_auditor import (
    PredictionAuditEvent,
    PredictionAuditor,
)


def build_test_logger(
    name: str,
) -> logging.Logger:
    """Return an isolated logger suitable for caplog assertions."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    return logger


def test_audit_event_exposes_record_count() -> None:
    event = PredictionAuditEvent.create(
        predicted_classes=[
            "ASSAULT",
            "FRAUD",
        ],
        predicted_probabilities=[
            0.60,
            0.80,
        ],
        inference_time_ms=12.0,
        model_name="xgboost_finalist",
    )

    assert event.record_count == 2


def test_audit_event_calculates_mean_confidence() -> None:
    event = PredictionAuditEvent.create(
        predicted_classes=[
            "ASSAULT",
            "FRAUD",
        ],
        predicted_probabilities=[
            0.60,
            0.80,
        ],
        inference_time_ms=12.0,
        model_name="xgboost_finalist",
    )

    assert event.mean_confidence == pytest.approx(0.70)


def test_event_converts_sequences_to_immutable_tuples() -> None:
    classes = [
        "ASSAULT",
    ]
    probabilities = [
        0.75,
    ]

    event = PredictionAuditEvent.create(
        predicted_classes=classes,
        predicted_probabilities=probabilities,
        inference_time_ms=5.0,
        model_name="xgboost_finalist",
    )

    classes.append("FRAUD")
    probabilities.append(0.25)

    assert event.predicted_classes == ("ASSAULT",)
    assert event.predicted_probabilities == (0.75,)


def test_record_prediction_updates_metrics() -> None:
    registry = MetricsRegistry()
    auditor = PredictionAuditor(
        metrics_registry=registry,
        audit_logger=build_test_logger("prediction-auditor-metrics-test"),
    )

    event = PredictionAuditEvent.create(
        predicted_classes=[
            "LARCENY/THEFT",
            "ASSAULT",
            "LARCENY/THEFT",
        ],
        predicted_probabilities=[
            0.70,
            0.55,
            0.80,
        ],
        inference_time_ms=18.0,
        model_name="xgboost_finalist",
    )

    auditor.record_prediction(event)

    snapshot = registry.snapshot()

    assert snapshot.successful_prediction_operations == 1
    assert snapshot.failed_prediction_operations == 0
    assert snapshot.records_processed == 3
    assert snapshot.average_inference_latency_ms == 18.0
    assert snapshot.prediction_counts == {
        "LARCENY/THEFT": 2,
        "ASSAULT": 1,
    }


def test_record_prediction_emits_structured_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = MetricsRegistry()
    test_logger = build_test_logger("prediction-auditor-log-test")

    auditor = PredictionAuditor(
        metrics_registry=registry,
        audit_logger=test_logger,
    )

    event = PredictionAuditEvent.create(
        predicted_classes=[
            "ASSAULT",
            "FRAUD",
        ],
        predicted_probabilities=[
            0.60,
            0.80,
        ],
        inference_time_ms=14.25,
        model_name="xgboost_finalist",
    )

    with caplog.at_level(
        logging.INFO,
        logger=test_logger.name,
    ):
        auditor.record_prediction(event)

    payload = json.loads(caplog.records[-1].message)

    assert payload["event"] == "prediction_completed"
    assert payload["model_name"] == "xgboost_finalist"
    assert payload["record_count"] == 2
    assert payload["predicted_classes"] == [
        "ASSAULT",
        "FRAUD",
    ]
    assert payload["mean_confidence"] == pytest.approx(0.70)
    assert payload["inference_time_ms"] == 14.25


def test_record_failure_updates_metrics_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = MetricsRegistry()
    test_logger = build_test_logger("prediction-auditor-failure-test")

    auditor = PredictionAuditor(
        metrics_registry=registry,
        audit_logger=test_logger,
    )

    with caplog.at_level(
        logging.ERROR,
        logger=test_logger.name,
    ):
        auditor.record_prediction_failure(
            model_name="xgboost_finalist",
            error_type="PredictionServiceError",
        )

    snapshot = registry.snapshot()
    payload = json.loads(caplog.records[-1].message)

    assert snapshot.successful_prediction_operations == 0
    assert snapshot.failed_prediction_operations == 1

    assert payload["event"] == "prediction_failed"
    assert payload["model_name"] == "xgboost_finalist"
    assert payload["error_type"] == ("PredictionServiceError")


@pytest.mark.parametrize(
    (
        "predicted_classes",
        "predicted_probabilities",
        "inference_time_ms",
        "model_name",
        "expected_message",
    ),
    [
        (
            [],
            [],
            1.0,
            "model",
            "at least one record",
        ),
        (
            ["ASSAULT"],
            [],
            1.0,
            "model",
            "equal lengths",
        ),
        (
            [""],
            [0.5],
            1.0,
            "model",
            "empty values",
        ),
        (
            ["ASSAULT"],
            [-0.1],
            1.0,
            "model",
            "between zero and one",
        ),
        (
            ["ASSAULT"],
            [1.1],
            1.0,
            "model",
            "between zero and one",
        ),
        (
            ["ASSAULT"],
            [0.5],
            -1.0,
            "model",
            "cannot be negative",
        ),
        (
            ["ASSAULT"],
            [0.5],
            1.0,
            "",
            "Model name cannot be empty",
        ),
    ],
)
def test_invalid_audit_event_is_rejected(
    predicted_classes: list[str],
    predicted_probabilities: list[float],
    inference_time_ms: float,
    model_name: str,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        PredictionAuditEvent.create(
            predicted_classes=predicted_classes,
            predicted_probabilities=(predicted_probabilities),
            inference_time_ms=inference_time_ms,
            model_name=model_name,
        )