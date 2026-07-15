from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ModelName = Literal[
    "dummy",
    "logistic",
    "naive_bayes",
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
    logistic_c: float = 0.1
    logistic_max_iter: int = 500
    logistic_tol: float = 1e-4
    logistic_solver: str = "saga"

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