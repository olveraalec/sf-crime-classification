from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.experiment_config import ExperimentConfig
from src.pipeline_builder import (
    build_features_from_config,
    build_transformer_from_config,
)


@pytest.fixture
def modeling_frame() -> pd.DataFrame:
    """Minimal frame matching the deterministic feature contract."""
    return pd.DataFrame(
        {
            "incident_year": [2015],
            "incident_month": [5],
            "incident_day": [13],
            "incident_hour": [23],
            "incident_minute": [53],
            "incident_day_of_week_num": [2],
            "incident_day_of_year": [133],
            "incident_week_of_year": [20],
            "datetime_numeric": [1431561180],
            "is_weekend": [0],
            "pd_district": ["NORTHERN"],
            "longitude": [-122.425892],
            "latitude": [37.774599],
            "street_1": ["OAK ST"],
            "street_2": ["LAGUNA ST"],
            "block_number": [pd.NA],
            "is_intersection": [1],
        }
    )


def make_config(
    model_name: str,
    **overrides: object,
) -> ExperimentConfig:
    defaults: dict[str, object] = {
        "experiment_name": f"test_{model_name}",
        "model_name": model_name,
        "categorical_encoding": "ordinal",
        "numeric_strategy": "passthrough",
        "geo_mode": "raw_distances",
        "n_geo_clusters": 7,
        "random_state": 2468,
        "sparse_output": False,
    }

    if model_name == "logistic":
        defaults.update(
            {
                "categorical_encoding": "onehot",
                "numeric_strategy": "standard",
                "sparse_output": True,
            }
        )

    if model_name == "naive_bayes":
        defaults.update(
            {
                "categorical_encoding": "ordinal",
                "numeric_strategy": "binned",
                "geo_mode": "cluster",
                "numeric_bins": 8,
                "sparse_output": False,
            }
        )

    if model_name == "dummy":
        defaults.update(
            {
                "categorical_encoding": "onehot",
                "numeric_strategy": "passthrough",
                "geo_mode": "none",
                "sparse_output": True,
            }
        )

    defaults.update(overrides)

    return ExperimentConfig(**defaults)


def test_feature_builder_uses_configuration(
    modeling_frame: pd.DataFrame,
) -> None:
    config = make_config(
        "xgboost",
        include_time_trend=False,
        add_cyclical=False,
        add_interactions=False,
        add_address_engineering=False,
    )

    result = build_features_from_config(
        modeling_frame,
        config,
    )

    assert "datetime_numeric" not in result.columns
    assert "incident_hour_sin" not in result.columns
    assert "district_hour" not in result.columns
    assert "street_pair" not in result.columns


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic",
        "naive_bayes",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
        "dummy",
    ],
)
def test_transformer_builder_returns_pipeline(
    model_name: str,
) -> None:
    transformer = build_transformer_from_config(make_config(model_name))

    assert isinstance(transformer, Pipeline)


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost",
    ],
)
def test_transformer_builder_propagates_random_state(
    model_name: str,
) -> None:
    config = make_config(
        model_name,
        random_state=2468,
    )

    transformer = build_transformer_from_config(config)

    geospatial = transformer.named_steps["geospatial"]

    assert geospatial.random_state == 2468


def test_naive_bayes_transformer_propagates_random_state() -> None:
    config = make_config(
        "naive_bayes",
        random_state=2468,
    )

    transformer = build_transformer_from_config(config)

    geospatial = transformer.named_steps["geospatial"]

    assert geospatial.random_state == 2468


def test_dummy_transformer_uses_no_geographic_processing() -> None:
    transformer = build_transformer_from_config(make_config("dummy"))

    geospatial = transformer.named_steps["geospatial"]

    assert geospatial.mode == "none"


def test_tree_transformer_uses_configured_cluster_count() -> None:
    transformer = build_transformer_from_config(
        make_config(
            "xgboost",
            n_geo_clusters=17,
        )
    )

    geospatial = transformer.named_steps["geospatial"]

    assert geospatial.n_clusters == 17


def test_naive_bayes_uses_configured_numeric_bins() -> None:
    transformer = build_transformer_from_config(
        make_config(
            "naive_bayes",
            numeric_bins=13,
        )
    )

    numeric_pipeline = transformer.named_steps["columns"].transformers[0][1]

    discretizer = numeric_pipeline.named_steps["discretizer"]

    assert discretizer.n_bins == 13


def test_invalid_configuration_is_rejected() -> None:
    config = make_config(
        "xgboost",
        n_geo_clusters=1,
    )

    with pytest.raises(ValueError):
        build_transformer_from_config(config)