from __future__ import annotations

from typing import cast

from fastapi import Request

from src.config import AppSettings
from src.inference_engine import InferenceEngine
from src.metrics import MetricsRegistry
from src.prediction_auditor import PredictionAuditor


def get_inference_engine(
    request: Request,
) -> InferenceEngine:
    """Return the inference engine initialized during app startup."""
    engine = getattr(
        request.app.state,
        "inference_engine",
        None,
    )

    if engine is None:
        raise RuntimeError(
            "Inference engine has not been initialized."
        )

    return cast(
        InferenceEngine,
        engine,
    )


def get_metrics_registry(
    request: Request,
) -> MetricsRegistry:
    """Return the metrics registry initialized during app startup."""
    registry = getattr(
        request.app.state,
        "metrics_registry",
        None,
    )

    if registry is None:
        raise RuntimeError(
            "Metrics registry has not been initialized."
        )

    return cast(
        MetricsRegistry,
        registry,
    )


def get_prediction_auditor(
    request: Request,
) -> PredictionAuditor:
    """Return the prediction auditor initialized during app startup."""
    auditor = getattr(
        request.app.state,
        "prediction_auditor",
        None,
    )

    if auditor is None:
        raise RuntimeError(
            "Prediction auditor has not been initialized."
        )

    return cast(
        PredictionAuditor,
        auditor,
    )


def get_app_settings(
    request: Request,
) -> AppSettings:
    """Return the operational settings initialized at startup."""
    settings = getattr(
        request.app.state,
        "settings",
        None,
    )

    if settings is None:
        raise RuntimeError(
            "Application settings have not been initialized."
        )

    return cast(
        AppSettings,
        settings,
    )