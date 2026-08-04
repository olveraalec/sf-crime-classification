from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import (
    get_app_settings,
    get_inference_engine,
    get_prediction_auditor,
)
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    IncidentRequest,
    PredictionRequest,
    PredictionResponse,
    RankedPredictionResponse,
)
from src.config import AppSettings
from src.incident_adapter import RawIncident
from src.inference_engine import (
    InferenceEngine,
    InferenceResponse,
)
from src.prediction_auditor import (
    PredictionAuditEvent,
    PredictionAuditor,
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
    """Convert a domain inference result into an API response."""
    return PredictionResponse(
        input_position=inference.input_position,
        predicted_class=inference.predicted_class,
        predicted_probability=inference.predicted_probability,
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
    auditor: Annotated[
        PredictionAuditor,
        Depends(get_prediction_auditor),
    ],
    settings: Annotated[
        AppSettings,
        Depends(get_app_settings),
    ],
) -> PredictionResponse:
    """Run prediction for one raw incident."""
    try:
        result = engine.predict_one(
            request_to_raw_incident(
                request.incident
            ),
            top_k=request.top_k,
        )
    except Exception as error:
        auditor.record_prediction_failure(
            model_name=settings.model_name,
            error_type=type(error).__name__,
        )
        raise

    audit_event = PredictionAuditEvent.create(
        predicted_classes=[
            result.predicted_class,
        ],
        predicted_probabilities=[
            result.predicted_probability,
        ],
        inference_time_ms=result.inference_time_ms,
        model_name=settings.model_name,
    )

    auditor.record_prediction(
        audit_event
    )

    return inference_to_response(
        result
    )


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
    auditor: Annotated[
        PredictionAuditor,
        Depends(get_prediction_auditor),
    ],
    settings: Annotated[
        AppSettings,
        Depends(get_app_settings),
    ],
) -> BatchPredictionResponse:
    """Run one vectorized prediction request."""
    incidents = [
        request_to_raw_incident(item)
        for item in request.incidents
    ]

    try:
        results = engine.predict_batch(
            incidents,
            top_k=request.top_k,
        )
    except Exception as error:
        auditor.record_prediction_failure(
            model_name=settings.model_name,
            error_type=type(error).__name__,
        )
        raise

    audit_event = PredictionAuditEvent.create(
        predicted_classes=[
            item.predicted_class
            for item in results
        ],
        predicted_probabilities=[
            item.predicted_probability
            for item in results
        ],
        inference_time_ms=sum(
            item.inference_time_ms
            for item in results
        ),
        model_name=settings.model_name,
    )

    auditor.record_prediction(
        audit_event
    )

    predictions = [
        inference_to_response(item)
        for item in results
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        prediction_count=len(predictions),
    )