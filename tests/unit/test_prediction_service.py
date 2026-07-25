from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.artifact_contract import get_final_artifact_paths
from src.artifact_loader import ArtifactBundle
from src.prediction_service import (
    PredictionError,
    PredictionService,
    PredictionServiceConfigError,
)


class FakeTransformer:
    """Small transformer double that validates deterministic input shape."""

    def __init__(self) -> None:
        self.received_features: pd.DataFrame | None = None

    def transform(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        self.received_features = features.copy()

        if features.shape[1] != 30:
            raise ValueError(
                f"Expected 30 deterministic features, got {features.shape[1]}."
            )

        return np.ones((len(features), 70))


class FakeProbabilityModel:
    """Return fixed three-class probabilities for each input row."""

    def predict_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.array(
            [
                [0.10, 0.70, 0.20],
                [0.60, 0.15, 0.25],
            ]
        )

        return probabilities[: len(features)]


class InvalidProbabilityModel:
    """Return malformed probabilities for failure testing."""

    def predict_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        del features
        return np.array([[0.5, 0.5]])


def make_modeling_frame(
    row_count: int = 2,
) -> pd.DataFrame:
    """Create rows matching the DuckDB modeling-view feature contract."""
    frame = pd.DataFrame(
        {
            "incident_year": [2015, 2015],
            "incident_month": [5, 5],
            "incident_day": [13, 14],
            "incident_hour": [23, 0],
            "incident_minute": [53, 30],
            "incident_day_of_week_num": [2, 3],
            "incident_day_of_year": [133, 134],
            "incident_week_of_year": [20, 20],
            "datetime_numeric": [
                1431561180,
                1431563400,
            ],
            "is_weekend": [0, 0],
            "pd_district": [
                "NORTHERN",
                "MISSION",
            ],
            "longitude": [
                -122.425892,
                -122.419416,
            ],
            "latitude": [
                37.774599,
                37.774929,
            ],
            "street_1": [
                "OAK ST",
                "MISSION ST",
            ],
            "street_2": [
                "LAGUNA ST",
                pd.NA,
            ],
            "block_number": [
                pd.NA,
                "100",
            ],
            "is_intersection": [1, 0],
        }
    )

    return frame.iloc[:row_count].copy()


def make_bundle(
    tmp_path: Path,
    *,
    model: object | None = None,
    transformer: object | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> ArtifactBundle:
    label_encoder = LabelEncoder()
    label_encoder.fit(
        [
            "ASSAULT",
            "LARCENY/THEFT",
            "ROBBERY",
        ]
    )

    metadata: dict[str, object] = {
        "model_name": "xgboost_finalist",
        "training_rows": 100,
        "raw_feature_columns": 30,
        "transformed_feature_columns": 70,
        "class_count": 3,
        "classes": [
            "ASSAULT",
            "LARCENY/THEFT",
            "ROBBERY",
        ],
        "config": {
            "experiment_name": "xgboost_finalist",
            "model_name": "xgboost",
            "validation_mode": "temporal_cv",
            "include_time_trend": True,
            "add_cyclical": True,
            "drop_original_cyclical": False,
            "add_interactions": True,
            "add_address_engineering": True,
            "categorical_encoding": "ordinal",
            "numeric_strategy": "passthrough",
            "geo_mode": "raw_distances",
            "n_geo_clusters": 40,
            "numeric_bins": 10,
            "sparse_output": False,
            "logistic_l1_ratio": 0.0,
            "logistic_class_weight": None,
            "logistic_c": 0.1,
            "logistic_max_iter": 500,
            "logistic_tol": 0.0001,
            "logistic_solver": "saga",
            "forest_n_estimators": 200,
            "forest_criterion": "log_loss",
            "forest_max_depth": 24,
            "forest_min_samples_split": 2,
            "forest_min_samples_leaf": 5,
            "forest_max_features": "sqrt",
            "forest_bootstrap": True,
            "forest_class_weight": None,
            "forest_n_jobs": -1,
            "xgb_n_estimators": 600,
            "xgb_max_depth": 8,
            "xgb_learning_rate": 0.03,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.7,
            "xgb_min_child_weight": 5,
            "xgb_reg_alpha": 0.0,
            "xgb_reg_lambda": 12.0,
            "xgb_gamma": 0.0,
            "random_state": 12345,
        },
    }

    if metadata_overrides:
        metadata.update(metadata_overrides)

    return ArtifactBundle(
        model=(model if model is not None else FakeProbabilityModel()),
        transformer=(transformer if transformer is not None else FakeTransformer()),
        label_encoder=label_encoder,
        metadata=metadata,
        paths=get_final_artifact_paths(tmp_path),
    )


def test_service_reconstructs_experiment_configuration(
    tmp_path: Path,
) -> None:
    service = PredictionService(make_bundle(tmp_path))

    assert service.experiment_config.experiment_name == ("xgboost_finalist")
    assert service.experiment_config.model_name == "xgboost"
    assert service.experiment_config.geo_mode == "raw_distances"
    assert service.experiment_config.n_geo_clusters == 40


def test_predict_proba_runs_complete_preprocessing_sequence(
    tmp_path: Path,
) -> None:
    transformer = FakeTransformer()
    service = PredictionService(
        make_bundle(
            tmp_path,
            transformer=transformer,
        )
    )

    probabilities = service.predict_proba(make_modeling_frame())

    assert probabilities.shape == (2, 3)

    np.testing.assert_allclose(
        probabilities,
        np.array(
            [
                [0.10, 0.70, 0.20],
                [0.60, 0.15, 0.25],
            ]
        ),
    )

    assert transformer.received_features is not None
    assert transformer.received_features.shape == (2, 30)


def test_predict_returns_decoded_highest_probability_classes(
    tmp_path: Path,
) -> None:
    service = PredictionService(make_bundle(tmp_path))

    results = service.predict(make_modeling_frame())

    assert len(results) == 2

    assert results[0].predicted_class == "LARCENY/THEFT"
    assert results[0].predicted_probability == pytest.approx(0.70)

    assert results[1].predicted_class == "ASSAULT"
    assert results[1].predicted_probability == pytest.approx(0.60)


def test_predict_returns_ranked_top_k_results(
    tmp_path: Path,
) -> None:
    service = PredictionService(make_bundle(tmp_path))

    results = service.predict(
        make_modeling_frame(row_count=1),
        top_k=2,
    )

    top_predictions = results[0].top_predictions

    assert len(top_predictions) == 2

    assert top_predictions[0].rank == 1
    assert top_predictions[0].crime_category == "LARCENY/THEFT"
    assert top_predictions[0].probability == pytest.approx(0.70)

    assert top_predictions[1].rank == 2
    assert top_predictions[1].crime_category == "ROBBERY"
    assert top_predictions[1].probability == pytest.approx(0.20)


def test_top_k_is_capped_at_number_of_classes(
    tmp_path: Path,
) -> None:
    service = PredictionService(make_bundle(tmp_path))

    results = service.predict(
        make_modeling_frame(row_count=1),
        top_k=50,
    )

    assert len(results[0].top_predictions) == 3


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_predict_rejects_non_positive_top_k(
    tmp_path: Path,
    top_k: int,
) -> None:
    service = PredictionService(make_bundle(tmp_path))

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        service.predict(
            make_modeling_frame(row_count=1),
            top_k=top_k,
        )


def test_predict_rejects_empty_input(
    tmp_path: Path,
) -> None:
    service = PredictionService(make_bundle(tmp_path))

    with pytest.raises(
        PredictionError,
        match="cannot be empty",
    ):
        service.predict(make_modeling_frame().iloc[0:0])


def test_service_rejects_missing_metadata_configuration(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(
        tmp_path,
        metadata_overrides={
            "config": None,
        },
    )

    with pytest.raises(
        PredictionServiceConfigError,
        match="configuration",
    ):
        PredictionService(bundle)


def test_service_rejects_invalid_metadata_configuration(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(tmp_path)
    bundle.metadata["config"]["n_geo_clusters"] = 1

    with pytest.raises(
        PredictionServiceConfigError,
        match="Invalid experiment configuration",
    ):
        PredictionService(bundle)


def test_service_detects_probability_class_mismatch(
    tmp_path: Path,
) -> None:
    service = PredictionService(
        make_bundle(
            tmp_path,
            model=InvalidProbabilityModel(),
        )
    )

    with pytest.raises(
        PredictionError,
        match="class count",
    ):
        service.predict_proba(make_modeling_frame(row_count=1))


def test_service_detects_invalid_probability_rows(
    tmp_path: Path,
) -> None:
    class InvalidSumModel:
        def predict_proba(
            self,
            features: np.ndarray,
        ) -> np.ndarray:
            del features
            return np.array([[0.20, 0.20, 0.20]])

    service = PredictionService(
        make_bundle(
            tmp_path,
            model=InvalidSumModel(),
        )
    )

    with pytest.raises(
        PredictionError,
        match="sum to 1",
    ):
        service.predict_proba(make_modeling_frame(row_count=1))