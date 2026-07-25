from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import HealthResponse


router = APIRouter(
    tags=["health"],
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service health",
)
def health_check(
    request: Request,
) -> HealthResponse:
    """Return process and production-model readiness."""
    engine = getattr(
        request.app.state,
        "inference_engine",
        None,
    )

    model_loaded = engine is not None

    return HealthResponse(
        status=("healthy" if model_loaded else "unavailable"),
        service="sf-crime-classification-api",
        model_loaded=model_loaded,
    )