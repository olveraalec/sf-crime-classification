from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.experiment_config import ExperimentConfig
from src.features import build_feature_frame
from src.transformers import (
    build_logistic_transformer,
    build_naive_bayes_transformer,
    build_tree_transformer,
)


def build_features_from_config(
    data,
    config: ExperimentConfig,
):
    """Build deterministic features using an experiment configuration."""
    config.validate()

    return build_feature_frame(
        data=data,
        include_time_trend=config.include_time_trend,
        add_cyclical=config.add_cyclical,
        drop_original_cyclical=config.drop_original_cyclical,
        add_interactions=config.add_interactions,
        add_address_engineering=config.add_address_engineering,
    )


def build_transformer_from_config(
    config: ExperimentConfig,
) -> Pipeline:
    """Build the fitted transformer required by an experiment."""
    config.validate()

    if config.model_name == "logistic":
        return build_logistic_transformer(
            categorical_encoding=config.categorical_encoding,
            numeric_strategy=config.numeric_strategy,
            geo_mode=config.geo_mode,
            n_geo_clusters=config.n_geo_clusters,
            sparse_output=config.sparse_output,
        )

    if config.model_name == "naive_bayes":
        return build_naive_bayes_transformer(
            numeric_bins=config.numeric_bins,
            geo_mode=config.geo_mode,
            n_geo_clusters=config.n_geo_clusters,
        )

    if config.model_name in {
        "random_forest",
        "xgboost",
        "extra_trees",
        "hist_gradient_boosting",
    }:
        return build_tree_transformer(
            categorical_encoding=config.categorical_encoding,
            numeric_strategy=config.numeric_strategy,
            geo_mode=config.geo_mode,
            n_geo_clusters=config.n_geo_clusters,
        )

    if config.model_name == "dummy":
        return build_logistic_transformer(
            categorical_encoding="onehot",
            numeric_strategy="passthrough",
            geo_mode="none",
            sparse_output=True,
        )

    raise ValueError(f"No transformer builder exists for model '{config.model_name}'.")