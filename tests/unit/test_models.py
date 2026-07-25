from __future__ import annotations

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier

from src.experiment_config import ExperimentConfig
from src.models import (
    build_dummy_classifier,
    build_extra_trees,
    build_hist_gradient_boosting,
    build_logistic_regression,
    build_model,
    build_naive_bayes,
    build_random_forest,
    build_xgboost,
)


def make_config(
    model_name: str,
    **overrides: object,
) -> ExperimentConfig:
    """Create a valid test configuration for a requested model."""
    defaults: dict[str, object] = {
        "experiment_name": f"test_{model_name}",
        "model_name": model_name,
        "categorical_encoding": "ordinal",
        "numeric_strategy": "passthrough",
        "geo_mode": "raw",
        "sparse_output": False,
    }

    if model_name == "naive_bayes":
        defaults.update(
            {
                "numeric_strategy": "binned",
                "categorical_encoding": "ordinal",
                "sparse_output": False,
            }
        )

    defaults.update(overrides)

    return ExperimentConfig(**defaults)


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("dummy", DummyClassifier),
        ("logistic", LogisticRegression),
        ("naive_bayes", CategoricalNB),
        ("random_forest", RandomForestClassifier),
        ("extra_trees", ExtraTreesClassifier),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier,
        ),
        ("xgboost", XGBClassifier),
    ],
)
def test_build_model_returns_expected_estimator(
    model_name: str,
    expected_type: type,
) -> None:
    config = make_config(model_name)

    model = build_model(config)

    assert isinstance(model, expected_type)


def test_dummy_classifier_preserves_configuration() -> None:
    config = make_config(
        "dummy",
        random_state=2468,
    )

    model = build_dummy_classifier(config)

    assert model.strategy == "prior"
    assert model.random_state == 2468


def test_logistic_regression_preserves_configuration() -> None:
    config = make_config(
        "logistic",
        logistic_c=0.3,
        logistic_max_iter=900,
        logistic_tol=1e-5,
        logistic_solver="saga",
        logistic_l1_ratio=0.2,
        logistic_class_weight="balanced",
        random_state=2468,
    )

    model = build_logistic_regression(config)

    assert model.C == 0.3
    assert model.max_iter == 900
    assert model.tol == 1e-5
    assert model.solver == "saga"
    assert model.l1_ratio == 0.2
    assert model.class_weight == "balanced"
    assert model.random_state == 2468


def test_naive_bayes_preserves_version_2_alpha() -> None:
    config = make_config("naive_bayes")

    model = build_naive_bayes(config)

    assert model.alpha == 0.5


def test_random_forest_preserves_configuration() -> None:
    config = make_config(
        "random_forest",
        forest_n_estimators=125,
        forest_criterion="entropy",
        forest_max_depth=18,
        forest_min_samples_split=4,
        forest_min_samples_leaf=3,
        forest_max_features=0.5,
        forest_bootstrap=False,
        forest_class_weight="balanced",
        forest_n_jobs=2,
        random_state=2468,
    )

    model = build_random_forest(config)

    assert model.n_estimators == 125
    assert model.criterion == "entropy"
    assert model.max_depth == 18
    assert model.min_samples_split == 4
    assert model.min_samples_leaf == 3
    assert model.max_features == 0.5
    assert model.bootstrap is False
    assert model.class_weight == "balanced"
    assert model.n_jobs == 2
    assert model.random_state == 2468


def test_extra_trees_preserves_version_2_parameters() -> None:
    config = make_config(
        "extra_trees",
        random_state=2468,
    )

    model = build_extra_trees(config)

    assert model.n_estimators == 300
    assert model.max_depth is None
    assert model.min_samples_leaf == 2
    assert model.n_jobs == -1
    assert model.random_state == 2468


def test_hist_gradient_boosting_preserves_version_2_parameters() -> None:
    config = make_config(
        "hist_gradient_boosting",
        random_state=2468,
    )

    model = build_hist_gradient_boosting(config)

    assert model.learning_rate == 0.1
    assert model.max_iter == 200
    assert model.max_leaf_nodes == 31
    assert model.l2_regularization == 0.0
    assert model.random_state == 2468


def test_xgboost_preserves_configuration() -> None:
    config = make_config(
        "xgboost",
        xgb_n_estimators=600,
        xgb_max_depth=8,
        xgb_learning_rate=0.03,
        xgb_subsample=0.8,
        xgb_colsample_bytree=0.7,
        xgb_min_child_weight=5,
        xgb_reg_alpha=0.1,
        xgb_reg_lambda=12.0,
        xgb_gamma=0.2,
        random_state=2468,
    )

    model = build_xgboost(config)

    parameters = model.get_params()

    assert parameters["objective"] == "multi:softprob"
    assert parameters["eval_metric"] == "mlogloss"
    assert parameters["n_estimators"] == 600
    assert parameters["max_depth"] == 8
    assert parameters["learning_rate"] == 0.03
    assert parameters["subsample"] == 0.8
    assert parameters["colsample_bytree"] == 0.7
    assert parameters["min_child_weight"] == 5
    assert parameters["reg_alpha"] == 0.1
    assert parameters["reg_lambda"] == 12.0
    assert parameters["gamma"] == 0.2
    assert parameters["tree_method"] == "hist"
    assert parameters["random_state"] == 2468
    assert parameters["n_jobs"] == -1
    assert parameters["verbosity"] == 1


def test_build_model_validates_configuration() -> None:
    invalid_config = ExperimentConfig(
        experiment_name="invalid_logistic",
        model_name="logistic",
        logistic_c=0,
    )

    with pytest.raises(
        ValueError,
        match="logistic_c must be positive",
    ):
        build_model(invalid_config)