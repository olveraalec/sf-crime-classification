from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.artifact_loader import ArtifactBundle
from src.experiment_config import ExperimentConfig
from src.logger import get_logger
from src.pipeline_builder import build_features_from_config


logger = get_logger(__name__)


class PredictionServiceError(RuntimeError):
    """Base exception for prediction-service failures."""


class PredictionServiceConfigError(PredictionServiceError):
    """Raised when artifact metadata cannot configure inference."""


class PredictionError(PredictionServiceError):
    """Raised when an inference request cannot be completed."""


@dataclass(frozen=True)
class RankedPrediction:
    """One ranked crime-category probability."""

    rank: int
    crime_category: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    """Prediction output for one input record."""

    row_index: Any
    predicted_class: str
    predicted_probability: float
    top_predictions: tuple[RankedPrediction, ...]


class PredictionService:
    """Run validated inference using one loaded production artifact bundle."""

    def __init__(
        self,
        artifact_bundle: ArtifactBundle,
    ) -> None:
        self._bundle = artifact_bundle
        self._experiment_config = self._build_experiment_config(
            artifact_bundle.metadata
        )

    @property
    def artifact_bundle(self) -> ArtifactBundle:
        """Return the validated artifacts used by this service."""
        return self._bundle

    @property
    def experiment_config(self) -> ExperimentConfig:
        """Return the configuration reconstructed from model metadata."""
        return self._experiment_config

    @staticmethod
    def _build_experiment_config(
        metadata: dict[str, Any],
    ) -> ExperimentConfig:
        config_data = metadata.get("config")

        if not isinstance(config_data, dict):
            raise PredictionServiceConfigError(
                "Artifact metadata does not contain a valid experiment configuration."
            )

        try:
            config = ExperimentConfig(**config_data)
            config.validate()
        except (TypeError, ValueError) as error:
            raise PredictionServiceConfigError(
                "Invalid experiment configuration in artifact metadata."
            ) from error

        return config

    @staticmethod
    def _validate_input(
        modeling_data: pd.DataFrame,
    ) -> None:
        if not isinstance(modeling_data, pd.DataFrame):
            raise PredictionError("Prediction input must be a pandas DataFrame.")

        if modeling_data.empty:
            raise PredictionError("Prediction input cannot be empty.")

        if modeling_data.columns.duplicated().any():
            duplicate_columns = sorted(
                modeling_data.columns[modeling_data.columns.duplicated()].tolist()
            )

            raise PredictionError(
                f"Prediction input contains duplicate columns: {duplicate_columns}"
            )

    def build_features(
        self,
        modeling_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create deterministic features using the trained configuration."""
        self._validate_input(modeling_data)

        try:
            features = build_features_from_config(
                modeling_data,
                self._experiment_config,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PredictionError(
                "Could not build deterministic prediction features."
            ) from error

        expected_columns = self._bundle.metadata.get("raw_feature_columns")

        if isinstance(expected_columns, int) and features.shape[1] != expected_columns:
            raise PredictionError(
                "Deterministic feature count does not match model "
                f"metadata: expected {expected_columns}, "
                f"received {features.shape[1]}."
            )

        return features

    def transform(
        self,
        modeling_data: pd.DataFrame,
    ) -> Any:
        """Create deterministic and learned model features."""
        features = self.build_features(modeling_data)

        try:
            transformed = self._bundle.transformer.transform(features)
        except Exception as error:
            raise PredictionError(
                "Saved transformer could not process prediction features."
            ) from error

        if not hasattr(transformed, "shape"):
            raise PredictionError("Saved transformer returned an invalid result.")

        if transformed.shape[0] != len(modeling_data):
            raise PredictionError(
                "Transformed row count does not match prediction input."
            )

        expected_columns = self._bundle.metadata.get("transformed_feature_columns")

        if (
            isinstance(expected_columns, int)
            and transformed.shape[1] != expected_columns
        ):
            raise PredictionError(
                "Transformed feature count does not match model "
                f"metadata: expected {expected_columns}, "
                f"received {transformed.shape[1]}."
            )

        return transformed

    def predict_proba(
        self,
        modeling_data: pd.DataFrame,
    ) -> np.ndarray:
        """Return validated class probabilities for all input rows."""
        transformed = self.transform(modeling_data)

        try:
            probabilities = np.asarray(
                self._bundle.model.predict_proba(transformed),
                dtype=float,
            )
        except Exception as error:
            raise PredictionError(
                "Saved model could not produce class probabilities."
            ) from error

        self._validate_probabilities(
            probabilities=probabilities,
            expected_rows=len(modeling_data),
        )

        return probabilities

    def _validate_probabilities(
        self,
        *,
        probabilities: np.ndarray,
        expected_rows: int,
    ) -> None:
        if probabilities.ndim != 2:
            raise PredictionError(
                "Model probabilities must be a two-dimensional array."
            )

        if probabilities.shape[0] != expected_rows:
            raise PredictionError(
                "Probability row count does not match prediction input."
            )

        expected_class_count = len(self._bundle.label_encoder.classes_)

        if probabilities.shape[1] != expected_class_count:
            raise PredictionError(
                "Probability class count does not match the fitted label encoder."
            )

        if not np.isfinite(probabilities).all():
            raise PredictionError("Model probabilities contain non-finite values.")

        if (probabilities < 0).any() or (probabilities > 1).any():
            raise PredictionError("Model probabilities must remain between 0 and 1.")

        probability_sums = probabilities.sum(axis=1)

        if not np.allclose(
            probability_sums,
            1.0,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise PredictionError("Model probability rows must sum to 1.")

    def predict(
        self,
        modeling_data: pd.DataFrame,
        *,
        top_k: int = 3,
    ) -> list[PredictionResult]:
        """Return decoded predictions and ranked class probabilities."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        probabilities = self.predict_proba(modeling_data)
        classes = self._bundle.label_encoder.classes_

        resolved_top_k = min(
            top_k,
            len(classes),
        )

        results: list[PredictionResult] = []

        for row_position, row_index in enumerate(modeling_data.index):
            row_probabilities = probabilities[row_position]

            ranked_indices = np.argsort(row_probabilities)[::-1][:resolved_top_k]

            top_predictions = tuple(
                RankedPrediction(
                    rank=rank,
                    crime_category=str(classes[class_index]),
                    probability=float(row_probabilities[class_index]),
                )
                for rank, class_index in enumerate(
                    ranked_indices,
                    start=1,
                )
            )

            highest = top_predictions[0]

            results.append(
                PredictionResult(
                    row_index=row_index,
                    predicted_class=(highest.crime_category),
                    predicted_probability=(highest.probability),
                    top_predictions=top_predictions,
                )
            )

        logger.info(
            "Completed prediction for %s row(s) with top_k=%s.",
            len(results),
            resolved_top_k,
        )

        return results