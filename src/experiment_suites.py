from __future__ import annotations

from src.experiment_config import ExperimentConfig


def logistic_feature_ablation_suite() -> list[ExperimentConfig]:
    """
    Return Logistic Regression experiments that isolate major feature families.

    Each successive experiment changes one major component so performance
    differences are easier to interpret.
    """
    return [
        ExperimentConfig(
            experiment_name="logistic_01_raw_baseline",
            model_name="logistic",
            validation_mode="temporal_cv",
            include_time_trend=True,
            add_cyclical=False,
            add_interactions=False,
            add_address_engineering=False,
            categorical_encoding="onehot",
            numeric_strategy="standard",
            geo_mode="raw",
            sparse_output=True,
        ),
        ExperimentConfig(
            experiment_name="logistic_02_add_address",
            model_name="logistic",
            validation_mode="temporal_cv",
            include_time_trend=True,
            add_cyclical=False,
            add_interactions=False,
            add_address_engineering=True,
            categorical_encoding="onehot",
            numeric_strategy="standard",
            geo_mode="raw",
            sparse_output=True,
        ),
        ExperimentConfig(
            experiment_name="logistic_03_add_cyclical",
            model_name="logistic",
            validation_mode="temporal_cv",
            include_time_trend=True,
            add_cyclical=True,
            drop_original_cyclical=False,
            add_interactions=False,
            add_address_engineering=True,
            categorical_encoding="onehot",
            numeric_strategy="standard",
            geo_mode="raw",
            sparse_output=True,
        ),
        ExperimentConfig(
            experiment_name="logistic_04_add_interactions",
            model_name="logistic",
            validation_mode="temporal_cv",
            include_time_trend=True,
            add_cyclical=True,
            drop_original_cyclical=False,
            add_interactions=True,
            add_address_engineering=True,
            categorical_encoding="onehot",
            numeric_strategy="standard",
            geo_mode="raw",
            sparse_output=True,
        ),
        ExperimentConfig(
            experiment_name="logistic_05_add_geo40",
            model_name="logistic",
            validation_mode="temporal_cv",
            include_time_trend=True,
            add_cyclical=True,
            drop_original_cyclical=False,
            add_interactions=True,
            add_address_engineering=True,
            categorical_encoding="onehot",
            numeric_strategy="standard",
            geo_mode="raw_cluster",
            n_geo_clusters=40,
            sparse_output=True,
        ),
    ]


def logistic_geo_ablation_suite() -> list[ExperimentConfig]:
    """Compare alternative geographic representations."""
    common = {
        "model_name": "logistic",
        "validation_mode": "temporal_cv",
        "include_time_trend": True,
        "add_cyclical": True,
        "drop_original_cyclical": False,
        "add_interactions": True,
        "add_address_engineering": True,
        "categorical_encoding": "onehot",
        "numeric_strategy": "standard",
        "sparse_output": True,
    }

    return [
        ExperimentConfig(
            experiment_name="logistic_geo_raw",
            geo_mode="raw",
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_cluster20",
            geo_mode="raw_cluster",
            n_geo_clusters=20,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_cluster40",
            geo_mode="raw_cluster",
            n_geo_clusters=40,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_cluster80",
            geo_mode="raw_cluster",
            n_geo_clusters=80,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_distances20",
            geo_mode="raw_distances",
            n_geo_clusters=20,
            **common,
        ),
    ]


def logistic_encoding_ablation_suite() -> list[ExperimentConfig]:
    """Compare categorical encoding strategies for Logistic Regression."""
    common = {
        "model_name": "logistic",
        "validation_mode": "temporal_cv",
        "include_time_trend": True,
        "add_cyclical": True,
        "drop_original_cyclical": False,
        "add_interactions": True,
        "add_address_engineering": True,
        "numeric_strategy": "standard",
        "geo_mode": "raw_cluster",
        "n_geo_clusters": 40,
    }

    return [
        ExperimentConfig(
            experiment_name="logistic_encoding_onehot",
            categorical_encoding="onehot",
            sparse_output=True,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_encoding_frequency",
            categorical_encoding="frequency",
            sparse_output=False,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_encoding_ordinal",
            categorical_encoding="ordinal",
            sparse_output=False,
            **common,
        ),
    ]


def logistic_geo_followup_suite() -> list[ExperimentConfig]:
    """
    Compare new geographic representations against the current
    raw-coordinate plus 40-cluster winner.

    The existing raw and raw-cluster-40 results are already stored, so this
    suite avoids rerunning those identical experiments.
    """
    common = {
        "model_name": "logistic",
        "validation_mode": "temporal_cv",
        "include_time_trend": True,
        "add_cyclical": True,
        "drop_original_cyclical": False,
        "add_interactions": True,
        "add_address_engineering": True,
        "categorical_encoding": "onehot",
        "numeric_strategy": "standard",
        "sparse_output": True,
        "logistic_c": 0.1,
        "logistic_max_iter": 500,
    }

    return [
        ExperimentConfig(
            experiment_name="logistic_geo_raw_cluster20",
            geo_mode="raw_cluster",
            n_geo_clusters=20,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_raw_cluster80",
            geo_mode="raw_cluster",
            n_geo_clusters=80,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_cluster40_only",
            geo_mode="cluster",
            n_geo_clusters=40,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_raw_distances20",
            geo_mode="raw_distances",
            n_geo_clusters=20,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_geo_raw_distances40",
            geo_mode="raw_distances",
            n_geo_clusters=40,
            **common,
        ),
    ]


def logistic_encoding_followup_suite() -> list[ExperimentConfig]:
    """
    Compare compact categorical encodings against the stored one-hot winner.

    The one-hot raw-distances-40 result already exists, so this suite runs
    only frequency and ordinal alternatives.
    """
    common = {
        "model_name": "logistic",
        "validation_mode": "temporal_cv",
        "include_time_trend": True,
        "add_cyclical": True,
        "drop_original_cyclical": False,
        "add_interactions": True,
        "add_address_engineering": True,
        "numeric_strategy": "standard",
        "geo_mode": "raw_distances",
        "n_geo_clusters": 40,
        "logistic_c": 0.1,
        "logistic_max_iter": 500,
    }

    return [
        ExperimentConfig(
            experiment_name="logistic_encoding_frequency_geo_dist40",
            categorical_encoding="frequency",
            sparse_output=False,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_encoding_ordinal_geo_dist40",
            categorical_encoding="ordinal",
            sparse_output=False,
            **common,
        ),
    ]