from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ModelName = Literal[
    "dummy",
    "logistic",
    "naive_bayes",
    "random_forest",
    "xgboost",
    "extra_trees",
    "hist_gradient_boosting",
]

ValidationMode = Literal[
    "random_v1",
    "temporal_cv",
    "frozen_test",
]

CategoricalEncoding = Literal[
    "onehot",
    "ordinal",
    "frequency",
]

NumericStrategy = Literal[
    "standard",
    "robust",
    "passthrough",
    "binned",
]

GeoMode = Literal[
    "raw",
    "cluster",
    "distances",
    "raw_cluster",
    "raw_distances",
    "all",
    "none",
]


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for one reproducible modeling experiment."""

    experiment_name: str
    model_name: ModelName

    validation_mode: ValidationMode = "temporal_cv"

    # Deterministic feature choices
    include_time_trend: bool = True
    add_cyclical: bool = True
    drop_original_cyclical: bool = False
    add_interactions: bool = True
    add_address_engineering: bool = True

    # Learned preprocessing choices
    categorical_encoding: CategoricalEncoding = "onehot"
    numeric_strategy: NumericStrategy = "standard"
    geo_mode: GeoMode = "raw"
    n_geo_clusters: int = 40
    numeric_bins: int = 10
    sparse_output: bool = True

    # Logistic Regression parameters
    logistic_l1_ratio: float = 0.0
    logistic_class_weight: str | None = None
    logistic_c: float = 0.1
    logistic_max_iter: int = 500
    logistic_tol: float = 1e-4
    logistic_solver: str = "saga"

    # Random Forest parameters
    forest_n_estimators: int = 200
    forest_criterion: str = "log_loss"
    forest_max_depth: int | None = 24
    forest_min_samples_split: int = 2
    forest_min_samples_leaf: int = 5
    forest_max_features: str | float | None = "sqrt"
    forest_bootstrap: bool = True
    forest_class_weight: str | None = None
    forest_n_jobs: int = -1

    # XGBoost parameters
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    xgb_min_child_weight: int = 1

    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0

    xgb_gamma: float = 0.0

    # General reproducibility
    random_state: int = 12345

    def to_dict(self) -> dict[str, object]:
        """Convert the experiment configuration into a serializable dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Validate model and preprocessing compatibility."""
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty.")

        if self.n_geo_clusters < 2:
            raise ValueError("n_geo_clusters must be at least 2.")

        if self.numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2.")

        if self.model_name == "naive_bayes":
            if self.categorical_encoding != "ordinal":
                raise ValueError(
                    "Categorical Naive Bayes requires ordinal categorical encoding."
                )

            if self.numeric_strategy != "binned":
                raise ValueError(
                    "Categorical Naive Bayes requires binned numeric features."
                )

            if self.sparse_output:
                raise ValueError(
                    "Categorical Naive Bayes should use dense transformed output."
                )

        if (
            self.model_name == "hist_gradient_boosting"
            and self.categorical_encoding == "onehot"
        ):
            raise ValueError(
                "HistGradientBoosting should use ordinal or frequency encoding, "
                "not sparse one-hot encoding."
            )
        if self.logistic_c <= 0:
            raise ValueError("logistic_c must be positive.")

        if self.logistic_max_iter < 1:
            raise ValueError("logistic_max_iter must be at least 1.")

        if self.logistic_tol <= 0:
            raise ValueError("logistic_tol must be positive.")
        if not 0.0 <= self.logistic_l1_ratio <= 1.0:
            raise ValueError("logistic_l1_ratio must be between 0 and 1.")
        if self.logistic_class_weight not in {None, "balanced"}:
            raise ValueError("logistic_class_weight must be None or 'balanced'.")

        if self.logistic_l1_ratio > 0 and self.logistic_solver != "saga":
            raise ValueError(
                "L1 and Elastic Net Logistic Regression require the saga solver."
            )
        if self.forest_n_estimators < 1:
            raise ValueError("forest_n_estimators must be at least 1.")

        if self.forest_max_depth is not None and self.forest_max_depth < 1:
            raise ValueError("forest_max_depth must be None or at least 1.")

        if self.forest_min_samples_split < 2:
            raise ValueError("forest_min_samples_split must be at least 2.")

        if self.forest_min_samples_leaf < 1:
            raise ValueError("forest_min_samples_leaf must be at least 1.")

        if self.forest_criterion not in {
            "gini",
            "entropy",
            "log_loss",
        }:
            raise ValueError(
                "forest_criterion must be 'gini', 'entropy', or 'log_loss'."
            )

        if self.forest_class_weight not in {
            None,
            "balanced",
            "balanced_subsample",
        }:
            raise ValueError(
                "forest_class_weight must be None, 'balanced', or 'balanced_subsample'."
            )
        if self.xgb_gamma < 0:
            raise ValueError("xgb_gamma must be non-negative.")

        if self.xgb_reg_alpha < 0:
            raise ValueError("xgb_reg_alpha must be non-negative.")

        if self.xgb_reg_lambda < 0:
            raise ValueError("xgb_reg_lambda must be non-negative.")


def logistic_baseline_config() -> ExperimentConfig:
    """Return the primary Logistic Regression baseline configuration."""
    return ExperimentConfig(
        experiment_name="logistic_raw_onehot_standard",
        model_name="logistic",
        validation_mode="temporal_cv",
        add_cyclical=False,
        add_interactions=False,
        add_address_engineering=True,
        categorical_encoding="onehot",
        numeric_strategy="standard",
        geo_mode="raw",
        sparse_output=True,
    )


def logistic_original_style_config() -> ExperimentConfig:
    """Approximate the strongest original-notebook Logistic configuration."""
    return ExperimentConfig(
        experiment_name="logistic_cyclical_interactions_geo40",
        model_name="logistic",
        validation_mode="temporal_cv",
        add_cyclical=True,
        drop_original_cyclical=False,
        add_interactions=True,
        add_address_engineering=True,
        categorical_encoding="onehot",
        numeric_strategy="standard",
        geo_mode="raw_cluster",
        n_geo_clusters=40,
        sparse_output=True,
    )


def naive_bayes_legacy_config() -> ExperimentConfig:
    """Return the retained legacy Categorical Naive Bayes configuration."""
    return ExperimentConfig(
        experiment_name="naive_bayes_legacy",
        model_name="naive_bayes",
        validation_mode="temporal_cv",
        add_cyclical=False,
        add_interactions=False,
        add_address_engineering=True,
        categorical_encoding="ordinal",
        numeric_strategy="binned",
        geo_mode="cluster",
        n_geo_clusters=40,
        numeric_bins=10,
        sparse_output=False,
    )


def tree_baseline_config(
    model_name: Literal[
        "xgboost",
        "extra_trees",
        "hist_gradient_boosting",
    ] = "extra_trees",
) -> ExperimentConfig:
    """Return a configurable nonlinear tree-model baseline."""
    return ExperimentConfig(
        experiment_name=f"{model_name}_ordinal_raw_cluster",
        model_name=model_name,
        validation_mode="temporal_cv",
        add_cyclical=True,
        add_interactions=True,
        add_address_engineering=True,
        categorical_encoding="ordinal",
        numeric_strategy="passthrough",
        geo_mode="raw_cluster",
        n_geo_clusters=40,
        sparse_output=False,
    )


def dummy_baseline_config() -> ExperimentConfig:
    """Return a class-prior DummyClassifier baseline."""
    return ExperimentConfig(
        experiment_name="dummy_class_prior",
        model_name="dummy",
        validation_mode="temporal_cv",
        include_time_trend=False,
        add_cyclical=False,
        add_interactions=False,
        add_address_engineering=False,
        categorical_encoding="onehot",
        numeric_strategy="passthrough",
        geo_mode="none",
        sparse_output=True,
    )


def logistic_finalist_config() -> ExperimentConfig:
    """Return the optimized Logistic Regression finalist configuration."""
    return ExperimentConfig(
        experiment_name="logistic_finalist_temporal_cv",
        model_name="logistic",
        validation_mode="temporal_cv",
        include_time_trend=True,
        add_cyclical=True,
        drop_original_cyclical=False,
        add_interactions=True,
        add_address_engineering=True,
        categorical_encoding="onehot",
        numeric_strategy="standard",
        geo_mode="raw_distances",
        n_geo_clusters=40,
        sparse_output=True,
        logistic_c=0.1,
        logistic_max_iter=750,
        logistic_tol=1e-4,
        logistic_solver="saga",
        logistic_l1_ratio=0.0,
        logistic_class_weight=None,
        random_state=12345,
    )


def random_forest_baseline_config() -> ExperimentConfig:
    """Return the initial Random Forest nonlinear benchmark."""
    return ExperimentConfig(
        experiment_name="random_forest_baseline",
        model_name="random_forest",
        validation_mode="temporal_cv",
        include_time_trend=True,
        add_cyclical=True,
        drop_original_cyclical=False,
        add_interactions=True,
        add_address_engineering=True,
        categorical_encoding="ordinal",
        numeric_strategy="passthrough",
        geo_mode="raw_distances",
        n_geo_clusters=40,
        sparse_output=False,
        forest_n_estimators=200,
        forest_criterion="log_loss",
        forest_max_depth=24,
        forest_min_samples_split=2,
        forest_min_samples_leaf=5,
        forest_max_features="sqrt",
        forest_bootstrap=True,
        forest_class_weight=None,
        forest_n_jobs=-1,
        random_state=12345,
    )


def xgboost_baseline_config() -> ExperimentConfig:
    """Return the initial XGBoost multiclass benchmark."""
    return ExperimentConfig(
        experiment_name="xgboost_baseline",
        model_name="xgboost",
        validation_mode="temporal_cv",
        include_time_trend=True,
        add_cyclical=True,
        drop_original_cyclical=False,
        add_interactions=True,
        add_address_engineering=True,
        categorical_encoding="ordinal",
        numeric_strategy="passthrough",
        geo_mode="raw_distances",
        n_geo_clusters=40,
        sparse_output=False,
        xgb_n_estimators=200,
        xgb_max_depth=6,
        xgb_learning_rate=0.10,
        xgb_subsample=1.0,
        xgb_colsample_bytree=1.0,
        xgb_min_child_weight=1,
        xgb_reg_alpha=0.0,
        xgb_reg_lambda=1.0,
        random_state=12345,
    )


def xgboost_finalist_config() -> ExperimentConfig:
    """Return the optimized XGBoost finalist configuration."""
    return ExperimentConfig(
        experiment_name="xgboost_finalist",
        model_name="xgboost",
        validation_mode="temporal_cv",
        include_time_trend=True,
        add_cyclical=True,
        drop_original_cyclical=False,
        add_interactions=True,
        add_address_engineering=True,
        categorical_encoding="ordinal",
        numeric_strategy="passthrough",
        geo_mode="raw_distances",
        n_geo_clusters=40,
        sparse_output=False,
        xgb_n_estimators=600,
        xgb_learning_rate=0.03,
        xgb_max_depth=8,
        xgb_min_child_weight=5,
        xgb_subsample=0.80,
        xgb_colsample_bytree=0.70,
        xgb_gamma=0.0,
        xgb_reg_alpha=0.0,
        xgb_reg_lambda=12.0,
        random_state=12345,
    )


def xgboost_feature_importance_suite():
    return [xgboost_finalist_config(experiment_name="xgboost_feature_importance")]