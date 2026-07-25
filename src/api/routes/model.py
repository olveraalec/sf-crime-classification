from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_inference_engine
from src.api.schemas import ModelInfoResponse
from src.inference_engine import InferenceEngine


router = APIRouter(
    prefix="/model",
    tags=["model"],
)


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get loaded model information",
)
def get_model_info(
    engine: Annotated[
        InferenceEngine,
        Depends(get_inference_engine),
    ],
) -> ModelInfoResponse:
    """Return selected metadata for the loaded production model."""
    info = engine.get_model_info()

    return ModelInfoResponse(
        model_name=info.model_name,
        algorithm=info.algorithm,
        trained_at_utc=info.trained_at_utc,
        class_count=info.class_count,
        raw_feature_columns=info.raw_feature_columns,
        transformed_feature_columns=(info.transformed_feature_columns),
        frozen_test_log_loss=info.frozen_test_log_loss,
    )