from __future__ import annotations

from typing import cast

from fastapi import Request

from src.inference_engine import InferenceEngine

from src.metrics import MetricsRegistry


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
        raise RuntimeError("Inference engine has not been initialized.")

    return cast(
        InferenceEngine,
        engine,
    )


def get_metrics_registry(
    request: Request,
) -> MetricsRegistry:
    """Return the shared metrics registry."""

    registry = getattr(
        request.app.state,
        "metrics_registry",
        None,
    )

    if registry is None:
        raise RuntimeError("Metrics registry has not been initialized.")

    return cast(
        MetricsRegistry,
        registry,
    )