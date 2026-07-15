from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB

from src.experiment_config import ExperimentConfig


def build_model(config: ExperimentConfig):
    """Build the estimator specified by an experiment configuration."""
    config.validate()

    if config.model_name == "dummy":
        return DummyClassifier(
            strategy="prior",
            random_state=config.random_state,
        )

    if config.model_name == "logistic":
        return LogisticRegression(
            C=config.logistic_c,
            max_iter=config.logistic_max_iter,
            tol=config.logistic_tol,
            solver=config.logistic_solver,
            random_state=config.random_state,
        )

    if config.model_name == "naive_bayes":
        return CategoricalNB(alpha=0.5)

    if config.model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=config.random_state,
            n_jobs=-1,
            class_weight=None,
        )

    if config.model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            random_state=config.random_state,
        )

    if config.model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "XGBoost is not installed. Run `uv add xgboost` first."
            ) from error

        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=300,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            random_state=config.random_state,
            n_jobs=-1,
        )

    raise ValueError(f"No model builder exists for model '{config.model_name}'.")