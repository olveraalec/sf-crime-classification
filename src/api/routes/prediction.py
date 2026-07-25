from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_inference_engine
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    IncidentRequest,
    PredictionRequest,
    PredictionResponse,
    RankedPredictionResponse,
)
from src.incident_adapter import RawIncident
from src.inference_engine import (
    InferenceEngine,
    InferenceResponse,
)


router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)


def request_to_raw_incident(
    request: IncidentRequest,
) -> RawIncident:
    """Convert validated API input into the domain incident object."""
    return RawIncident(
        incident_timestamp=request.incident_timestamp,
        pd_district=request.pd_district,
        address=request.address,
        longitude=request.longitude,
        latitude=request.latitude,
    )


def inference_to_response(
    inference: InferenceResponse,
) -> PredictionResponse:
    """Convert a framework-independent result into an API schema."""
    return PredictionResponse(
        input_position=inference.input_position,
        predicted_class=inference.predicted_class,
        predicted_probability=(inference.predicted_probability),
        top_predictions=[
            RankedPredictionResponse(
                rank=item.rank,
                crime_category=item.crime_category,
                probability=item.probability,
            )
            for item in inference.top_predictions
        ],
        inference_time_ms=inference.inference_time_ms,
    )


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict one crime category",
)
def predict_one(
    request: PredictionRequest,
    engine: Annotated[
        InferenceEngine,
        Depends(get_inference_engine),
    ],
) -> PredictionResponse:
    """Run prediction for one raw incident."""
    result = engine.predict_one(
        request_to_raw_incident(request.incident),
        top_k=request.top_k,
    )

    return inference_to_response(result)


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict multiple crime categories",
)
def predict_batch(
    request: BatchPredictionRequest,
    engine: Annotated[
        InferenceEngine,
        Depends(get_inference_engine),
    ],
) -> BatchPredictionResponse:
    """Run one vectorized prediction request for multiple incidents."""
    incidents = [request_to_raw_incident(item) for item in request.incidents]

    results = engine.predict_batch(
        incidents,
        top_k=request.top_k,
    )

    predictions = [inference_to_response(item) for item in results]

    return BatchPredictionResponse(
        predictions=predictions,
        prediction_count=len(predictions),
    )