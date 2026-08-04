from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
)
from src.config import AppSettings
from src.inference_engine import InferenceEngine
from src.metrics import MetricsRegistry
from src.prediction_auditor import PredictionAuditor


router = APIRouter(
    prefix="/health",
    tags=["health"],
)


DEFAULT_SERVICE_NAME = "sf-crime-classification-api"


def get_service_name(
    request: Request,
) -> str:
    """Return the configured public service name."""
    settings = getattr(
        request.app.state,
        "settings",
        None,
    )

    if isinstance(settings, AppSettings):
        return settings.service_name

    return DEFAULT_SERVICE_NAME


def build_readiness_checks(
    request: Request,
) -> tuple[dict[str, bool], str | None]:
    """Evaluate the dependencies required for prediction traffic."""
    engine = getattr(
        request.app.state,
        "inference_engine",
        None,
    )
    registry = getattr(
        request.app.state,
        "metrics_registry",
        None,
    )
    auditor = getattr(
        request.app.state,
        "prediction_auditor",
        None,
    )
    settings = getattr(
        request.app.state,
        "settings",
        None,
    )

    checks = {
        "inference_engine": isinstance(
            engine,
            InferenceEngine,
        ),
        "metrics_registry": isinstance(
            registry,
            MetricsRegistry,
        ),
        "prediction_auditor": isinstance(
            auditor,
            PredictionAuditor,
        ),
        "settings": isinstance(
            settings,
            AppSettings,
        ),
        "model_metadata": False,
    }

    # Tests may inject a structurally compatible engine double rather than
    # a literal InferenceEngine instance.
    if engine is not None:
        checks["inference_engine"] = callable(
            getattr(
                engine,
                "predict_one",
                None,
            )
        ) and callable(
            getattr(
                engine,
                "predict_batch",
                None,
            )
        )

    if checks["inference_engine"]:
        try:
            model_info = engine.get_model_info()
        except Exception as error:
            return (
                checks,
                "Loaded model metadata could not be validated: "
                f"{type(error).__name__}.",
            )

        checks["model_metadata"] = (
            getattr(
                model_info,
                "model_name",
                None,
            )
            not in {
                None,
                "",
                "unknown",
            }
            and isinstance(
                getattr(
                    model_info,
                    "class_count",
                    None,
                ),
                int,
            )
            and getattr(
                model_info,
                "class_count",
                0,
            )
            > 0
        )

    failed_checks = [name for name, passed in checks.items() if not passed]

    if failed_checks:
        return (
            checks,
            "Readiness checks failed: " + ", ".join(failed_checks),
        )

    return checks, None


@router.get(
    "",
    response_model=HealthResponse,
    summary="Check legacy service health",
)
def health_check(
    request: Request,
) -> HealthResponse:
    """
    Return the original Version 3 health contract.

    This endpoint remains available for backward compatibility with
    existing local smoke tests and clients.
    """
    engine = getattr(
        request.app.state,
        "inference_engine",
        None,
    )

    model_loaded = engine is not None

    return HealthResponse(
        status=("healthy" if model_loaded else "unavailable"),
        service=get_service_name(request),
        model_loaded=model_loaded,
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Check process liveness",
)
def liveness_check(
    request: Request,
) -> LivenessResponse:
    """
    Confirm that the FastAPI process can receive and answer requests.

    This endpoint deliberately does not inspect model artifacts or
    prediction dependencies.
    """
    return LivenessResponse(
        status="alive",
        service=get_service_name(request),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        503: {
            "model": ReadinessResponse,
            "description": (
                "The process is alive but not ready for prediction traffic."
            ),
        }
    },
    summary="Check prediction readiness",
)
def readiness_check(
    request: Request,
) -> ReadinessResponse | JSONResponse:
    """Confirm that all dependencies required for inference are ready."""
    checks, detail = build_readiness_checks(request)

    ready = all(checks.values())

    response = ReadinessResponse(
        status=("ready" if ready else "not_ready"),
        service=get_service_name(request),
        ready=ready,
        checks=checks,
        detail=detail,
    )

    if not ready:
        return JSONResponse(
            status_code=503,
            content=response.model_dump(),
        )

    return response