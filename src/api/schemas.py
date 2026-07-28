from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class IncidentRequest(BaseModel):
    """One raw incident submitted for model inference."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    incident_timestamp: datetime = Field(
        description=(
            "Local incident date and time represented by the "
            "historical San Francisco crime record."
        ),
        examples=["2015-05-13T23:53:00"],
    )

    pd_district: str = Field(
        min_length=1,
        description="San Francisco Police Department district.",
        examples=["NORTHERN"],
    )

    address: str = Field(
        min_length=1,
        description=("Incident address, such as an intersection or block address."),
        examples=["OAK ST / LAGUNA ST"],
    )

    longitude: float = Field(
        ge=-180,
        le=180,
        examples=[-122.425892],
    )

    latitude: float = Field(
        ge=-90,
        le=90,
        examples=[37.774599],
    )


class PredictionRequest(BaseModel):
    """Single-record inference request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    incident: IncidentRequest

    top_k: int = Field(
        default=3,
        ge=1,
        le=39,
        description="Number of ranked crime classes to return.",
    )


class BatchPredictionRequest(BaseModel):
    """Vectorized inference request."""

    model_config = ConfigDict(
        extra="forbid",
    )

    incidents: list[IncidentRequest] = Field(
        min_length=1,
        max_length=100,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=39,
    )


class RankedPredictionResponse(BaseModel):
    """One ranked class probability."""

    rank: int
    crime_category: str
    probability: float


class PredictionResponse(BaseModel):
    """Prediction response for one input record."""

    input_position: int
    predicted_class: str
    predicted_probability: float
    top_predictions: list[RankedPredictionResponse]
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response for a vectorized inference request."""

    predictions: list[PredictionResponse]
    prediction_count: int


class HealthResponse(BaseModel):
    """Basic service-health response."""

    status: str
    service: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Public metadata for the loaded production model."""

    model_name: str
    algorithm: str
    trained_at_utc: str | None
    class_count: int
    raw_feature_columns: int
    transformed_feature_columns: int
    frozen_test_log_loss: float | None


class ErrorResponse(BaseModel):
    """Consistent API error response."""

    error: str
    detail: str
    request_id: str | None = None