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


def logistic_convergence_suite() -> list[ExperimentConfig]:
    """
    Test whether the current winning Logistic configuration improves
    with a larger optimization budget.
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
        "geo_mode": "raw_distances",
        "n_geo_clusters": 40,
        "sparse_output": True,
        "logistic_c": 0.1,
        "logistic_tol": 1e-4,
        "logistic_solver": "saga",
        "logistic_l1_ratio": 0.0,
    }

    return [
        ExperimentConfig(
            experiment_name="logistic_convergence_750",
            logistic_max_iter=750,
            **common,
        ),
        ExperimentConfig(
            experiment_name="logistic_convergence_1000",
            logistic_max_iter=1000,
            **common,
        ),
    ]


def logistic_regularization_suite() -> list[ExperimentConfig]:
    """
    Tune inverse regularization strength for the current winning
    Logistic Regression feature representation.
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
        "geo_mode": "raw_distances",
        "n_geo_clusters": 40,
        "sparse_output": True,
        "logistic_max_iter": 750,
        "logistic_tol": 1e-4,
        "logistic_solver": "saga",
        "logistic_l1_ratio": 0.0,
        "logistic_class_weight": None,
    }

    c_values = [0.01, 0.03, 0.1, 0.3, 1.0]

    return [
        ExperimentConfig(
            experiment_name=f"logistic_c_{str(c_value).replace('.', '_')}",
            logistic_c=c_value,
            **common,
        )
        for c_value in c_values
    ]


def logistic_elastic_net_suite() -> list[ExperimentConfig]:
    """
    Test whether a small L1 component improves the current Logistic winner.

    l1_ratio=0.0 is the existing pure-L2 benchmark, so this suite runs only
    new Elastic Net alternatives.
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
        "geo_mode": "raw_distances",
        "n_geo_clusters": 40,
        "sparse_output": True,
        "logistic_c": 0.1,
        "logistic_max_iter": 750,
        "logistic_tol": 1e-4,
        "logistic_solver": "saga",
        "logistic_class_weight": None,
    }

    l1_ratios = [0.05, 0.10, 0.25]

    return [
        ExperimentConfig(
            experiment_name=(f"logistic_elastic_net_{str(l1_ratio).replace('.', '_')}"),
            logistic_l1_ratio=l1_ratio,
            **common,
        )
        for l1_ratio in l1_ratios
    ]


def random_forest_leaf_suite() -> list[ExperimentConfig]:
    """
    Test tree complexity and leaf smoothing for Random Forest probability quality.
    """
    common = {
        "model_name": "random_forest",
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
        "sparse_output": False,
        "forest_n_estimators": 300,
        "forest_criterion": "log_loss",
        "forest_min_samples_split": 2,
        "forest_max_features": "sqrt",
        "forest_bootstrap": True,
        "forest_class_weight": None,
        "forest_n_jobs": -1,
        "random_state": 12345,
    }

    settings = [
        (16, 5),
        (16, 20),
        (24, 10),
        (24, 25),
        (32, 10),
        (32, 25),
        (None, 25),
    ]

    return [
        ExperimentConfig(
            experiment_name=(f"random_forest_depth_{depth}_leaf_{leaf}"),
            forest_max_depth=depth,
            forest_min_samples_leaf=leaf,
            **common,
        )
        for depth, leaf in settings
    ]


def xgboost_baseline_suite():

    return [
        ExperimentConfig(
            experiment_name="xgboost_baseline",
            model_name="xgboost",
            validation_mode="temporal_cv",
            include_time_trend=True,
            add_cyclical=True,
            add_interactions=True,
            add_address_engineering=True,
            categorical_encoding="ordinal",
            geo_mode="raw_distances",
            n_geo_clusters=40,
            xgb_n_estimators=300,
            xgb_max_depth=6,
            xgb_learning_rate=0.1,
            xgb_subsample=0.8,
            xgb_colsample_bytree=0.8,
        )
    ]


def xgboost_complexity_suite() -> list[ExperimentConfig]:
    """
    Tune XGBoost tree depth and minimum child weight.

    Depth controls interaction complexity. Minimum child weight prevents
    splits based on insufficient training evidence.
    """
    common = {
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
        "sparse_output": False,
        "xgb_n_estimators": 200,
        "xgb_learning_rate": 0.10,
        "xgb_subsample": 1.0,
        "xgb_colsample_bytree": 1.0,
        "xgb_reg_alpha": 0.0,
        "xgb_reg_lambda": 1.0,
        "random_state": 12345,
    }

    settings = [
        (4, 1),
        (4, 5),
        (6, 1),
        (6, 5),
        (8, 1),
        (8, 5),
    ]

    return [
        ExperimentConfig(
            experiment_name=(f"xgboost_depth_{depth}_child_{child_weight}"),
            xgb_max_depth=depth,
            xgb_min_child_weight=child_weight,
            **common,
        )
        for depth, child_weight in settings
    ]