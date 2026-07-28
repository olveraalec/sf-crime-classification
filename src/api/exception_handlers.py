from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.artifact_loader import ArtifactError
from src.incident_adapter import IncidentAdapterError
from src.inference_engine import InferenceEngineError
from src.logger import get_logger
from src.prediction_service import PredictionServiceError
from src.request_context import get_request_id

logger = get_logger(__name__)


def build_error_content(
    *,
    error: str,
    detail: str,
) -> dict[str, str]:
    """Build an API error response with request correlation."""
    content = {
        "error": error,
        "detail": detail,
    }

    request_id = get_request_id()

    if request_id is not None:
        content["request_id"] = request_id

    return content


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """Register consistent JSON handlers for inference failures."""

    @app.exception_handler(IncidentAdapterError)
    async def handle_incident_adapter_error(
        request: Request,
        error: IncidentAdapterError,
    ) -> JSONResponse:
        del request

        return JSONResponse(
            status_code=422,
            content=build_error_content(
                error="invalid_incident",
                detail=str(error),
            ),
        )

    @app.exception_handler(PredictionServiceError)
    async def handle_prediction_service_error(
        request: Request,
        error: PredictionServiceError,
    ) -> JSONResponse:
        logger.exception(
            "Prediction-service failure for %s",
            request.url.path,
            exc_info=error,
        )

        return JSONResponse(
            status_code=500,
            content=build_error_content(
                error="prediction_failed",
                detail="The model could not complete the prediction.",
            ),
        )

    @app.exception_handler(InferenceEngineError)
    async def handle_inference_engine_error(
        request: Request,
        error: InferenceEngineError,
    ) -> JSONResponse:
        logger.exception(
            "Inference-engine failure for %s",
            request.url.path,
            exc_info=error,
        )

        return JSONResponse(
            status_code=500,
            content=build_error_content(
                error="inference_failed",
                detail=str(error),
            ),
        )

    @app.exception_handler(ArtifactError)
    async def handle_artifact_error(
        request: Request,
        error: ArtifactError,
    ) -> JSONResponse:
        logger.exception(
            "Artifact failure for %s",
            request.url.path,
            exc_info=error,
        )

        return JSONResponse(
            status_code=503,
            content=build_error_content(
                error="model_unavailable",
                detail="Production model artifacts are unavailable.",
            ),
        )