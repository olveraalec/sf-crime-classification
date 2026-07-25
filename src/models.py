from __future__ import annotations

from typing import TypeAlias

from sklearn.base import ClassifierMixin
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


Classifier: TypeAlias = ClassifierMixin


def build_dummy_classifier(
    config: ExperimentConfig,
) -> DummyClassifier:
    """Build the class-prior dummy baseline."""
    return DummyClassifier(
        strategy="prior",
        random_state=config.random_state,
    )


def build_logistic_regression(
    config: ExperimentConfig,
) -> LogisticRegression:
    """Build a configured Logistic Regression classifier."""
    return LogisticRegression(
        C=config.logistic_c,
        l1_ratio=config.logistic_l1_ratio,
        max_iter=config.logistic_max_iter,
        tol=config.logistic_tol,
        solver=config.logistic_solver,
        class_weight=config.logistic_class_weight,
        random_state=config.random_state,
    )


def build_naive_bayes(
    config: ExperimentConfig,
) -> CategoricalNB:
    """Build the retained Categorical Naive Bayes benchmark."""
    del config

    return CategoricalNB(
        alpha=0.5,
    )


def build_random_forest(
    config: ExperimentConfig,
) -> RandomForestClassifier:
    """Build a configured Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=config.forest_n_estimators,
        criterion=config.forest_criterion,
        max_depth=config.forest_max_depth,
        min_samples_split=config.forest_min_samples_split,
        min_samples_leaf=config.forest_min_samples_leaf,
        max_features=config.forest_max_features,
        bootstrap=config.forest_bootstrap,
        class_weight=config.forest_class_weight,
        random_state=config.random_state,
        n_jobs=config.forest_n_jobs,
        verbose=1,
    )


def build_extra_trees(
    config: ExperimentConfig,
) -> ExtraTreesClassifier:
    """Build the retained Extra Trees benchmark."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=config.random_state,
        n_jobs=-1,
        class_weight=None,
    )


def build_hist_gradient_boosting(
    config: ExperimentConfig,
) -> HistGradientBoostingClassifier:
    """Build the retained histogram gradient boosting benchmark."""
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=config.random_state,
    )


def build_xgboost(
    config: ExperimentConfig,
) -> XGBClassifier:
    """Build a configured multiclass XGBoost classifier."""
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        min_child_weight=config.xgb_min_child_weight,
        reg_alpha=config.xgb_reg_alpha,
        reg_lambda=config.xgb_reg_lambda,
        gamma=config.xgb_gamma,
        tree_method="hist",
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=1,
    )


def build_model(
    config: ExperimentConfig,
) -> Classifier:
    """Build the classifier specified by an experiment configuration."""
    config.validate()

    builders = {
        "dummy": build_dummy_classifier,
        "logistic": build_logistic_regression,
        "naive_bayes": build_naive_bayes,
        "random_forest": build_random_forest,
        "extra_trees": build_extra_trees,
        "hist_gradient_boosting": build_hist_gradient_boosting,
        "xgboost": build_xgboost,
    }

    try:
        builder = builders[config.model_name]
    except KeyError as error:
        raise ValueError(
            f"No model builder exists for model '{config.model_name}'."
        ) from error

    return builder(config)