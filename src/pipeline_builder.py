from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import pandas as pd
from sklearn.pipeline import Pipeline

from src.experiment_config import ExperimentConfig
from src.features import build_feature_frame
from src.transformers import (
    build_logistic_transformer,
    build_naive_bayes_transformer,
    build_tree_transformer,
)


TransformerBuilder: TypeAlias = Callable[
    [ExperimentConfig],
    Pipeline,
]


def build_features_from_config(
    data: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Build deterministic model features from an experiment configuration."""
    config.validate()

    return build_feature_frame(
        data=data,
        include_time_trend=config.include_time_trend,
        add_cyclical=config.add_cyclical,
        drop_original_cyclical=config.drop_original_cyclical,
        add_interactions=config.add_interactions,
        add_address_engineering=config.add_address_engineering,
    )


def build_logistic_transformer_from_config(
    config: ExperimentConfig,
) -> Pipeline:
    """Build Logistic Regression preprocessing from configuration."""
    return build_logistic_transformer(
        categorical_encoding=config.categorical_encoding,
        numeric_strategy=config.numeric_strategy,
        geo_mode=config.geo_mode,
        n_geo_clusters=config.n_geo_clusters,
        sparse_output=config.sparse_output,
        random_state=config.random_state,
    )


def build_naive_bayes_transformer_from_config(
    config: ExperimentConfig,
) -> Pipeline:
    """Build Naive Bayes preprocessing from configuration."""
    return build_naive_bayes_transformer(
        numeric_bins=config.numeric_bins,
        geo_mode=config.geo_mode,
        n_geo_clusters=config.n_geo_clusters,
        random_state=config.random_state,
    )


def build_tree_transformer_from_config(
    config: ExperimentConfig,
) -> Pipeline:
    """Build tree-model preprocessing from configuration."""
    return build_tree_transformer(
        categorical_encoding=config.categorical_encoding,
        numeric_strategy=config.numeric_strategy,
        geo_mode=config.geo_mode,
        n_geo_clusters=config.n_geo_clusters,
        random_state=config.random_state,
    )


def build_dummy_transformer_from_config(
    config: ExperimentConfig,
) -> Pipeline:
    """Build lightweight preprocessing for the dummy baseline."""
    return build_logistic_transformer(
        categorical_encoding="onehot",
        numeric_strategy="passthrough",
        geo_mode="none",
        n_geo_clusters=config.n_geo_clusters,
        sparse_output=True,
        random_state=config.random_state,
    )


def build_transformer_from_config(
    config: ExperimentConfig,
) -> Pipeline:
    """Build the preprocessing pipeline required by an experiment."""
    config.validate()

    builders: dict[str, TransformerBuilder] = {
        "logistic": build_logistic_transformer_from_config,
        "naive_bayes": build_naive_bayes_transformer_from_config,
        "random_forest": build_tree_transformer_from_config,
        "extra_trees": build_tree_transformer_from_config,
        "hist_gradient_boosting": build_tree_transformer_from_config,
        "xgboost": build_tree_transformer_from_config,
        "dummy": build_dummy_transformer_from_config,
    }

    try:
        builder = builders[config.model_name]
    except KeyError as error:
        raise ValueError(
            f"No transformer builder exists for model '{config.model_name}'."
        ) from error

    return builder(config)