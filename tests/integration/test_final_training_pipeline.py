from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.experiment_config import xgboost_finalist_config
from src.train_final_model import train_and_save_final_model


def make_training_frame() -> pd.DataFrame:
    """Create a tiny modeling dataset for orchestration testing."""
    return pd.DataFrame(
        {
            "target": [
                "ASSAULT",
                "LARCENY/THEFT",
                "ASSAULT",
            ],
            "incident_year": [2015, 2015, 2015],
            "incident_month": [5, 5, 5],
            "incident_day": [13, 14, 15],
            "incident_hour": [23, 0, 12],
            "incident_minute": [53, 30, 15],
            "incident_day_of_week_num": [2, 3, 4],
            "incident_day_of_year": [133, 134, 135],
            "incident_week_of_year": [20, 20, 20],
            "datetime_numeric": [
                1431561180,
                1431563400,
                1431692100,
            ],
            "is_weekend": [0, 0, 0],
            "pd_district": [
                "NORTHERN",
                "MISSION",
                "SOUTHERN",
            ],
            "longitude": [
                -122.425892,
                -122.419416,
                -122.403405,
            ],
            "latitude": [
                37.774599,
                37.774929,
                37.775421,
            ],
            "street_1": [
                "OAK ST",
                "MISSION ST",
                "MARKET ST",
            ],
            "street_2": [
                "LAGUNA ST",
                pd.NA,
                "5TH ST",
            ],
            "block_number": [
                pd.NA,
                "100",
                pd.NA,
            ],
            "is_intersection": [1, 0, 1],
        }
    )


def test_final_training_orchestrates_centralized_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    training_data = make_training_frame()
    config = xgboost_finalist_config()

    feature_frame = pd.DataFrame(
        np.ones((3, 30)),
        columns=[f"feature_{index}" for index in range(30)],
    )

    transformed_features = np.ones((3, 70))

    transformer = MagicMock()
    transformer.fit_transform.return_value = transformed_features

    model = MagicMock()

    saved_artifacts: dict[str, object] = {}

    monkeypatch.setattr(
        "src.train_final_model.load_modeling_data",
        lambda: training_data,
    )

    build_features = MagicMock(return_value=feature_frame)
    monkeypatch.setattr(
        "src.train_final_model.build_features_from_config",
        build_features,
    )

    build_transformer = MagicMock(return_value=transformer)
    monkeypatch.setattr(
        "src.train_final_model.build_transformer_from_config",
        build_transformer,
    )

    build_model = MagicMock(return_value=model)
    monkeypatch.setattr(
        "src.train_final_model.build_model",
        build_model,
    )

    def fake_save_final_artifacts(**kwargs: object) -> None:
        saved_artifacts.update(kwargs)

    monkeypatch.setattr(
        "src.train_final_model.save_final_artifacts",
        fake_save_final_artifacts,
    )

    result = train_and_save_final_model(
        model_directory=tmp_path,
        config=config,
    )

    build_features.assert_called_once_with(
        training_data,
        config,
    )

    build_transformer.assert_called_once_with(config)
    transformer.fit_transform.assert_called_once_with(feature_frame)

    build_model.assert_called_once_with(config)
    model.fit.assert_called_once()

    fitted_features, fitted_target = model.fit.call_args.args

    np.testing.assert_array_equal(
        fitted_features,
        transformed_features,
    )
    np.testing.assert_array_equal(
        fitted_target,
        np.array([0, 1, 0]),
    )

    assert saved_artifacts["model"] is model
    assert saved_artifacts["transformer"] is transformer

    label_encoder = saved_artifacts["label_encoder"]

    assert label_encoder.classes_.tolist() == [
        "ASSAULT",
        "LARCENY/THEFT",
    ]

    metadata = saved_artifacts["metadata"]

    assert metadata["training_rows"] == 3
    assert metadata["raw_feature_columns"] == 30
    assert metadata["transformed_feature_columns"] == 70
    assert metadata["class_count"] == 2
    assert metadata["config"] == config.to_dict()

    assert result.training_rows == 3
    assert result.raw_feature_columns == 30
    assert result.transformed_feature_columns == 70
    assert result.class_count == 2

    assert result.artifacts.model == (tmp_path / "xgboost_final_model.joblib")