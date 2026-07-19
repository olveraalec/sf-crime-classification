from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier

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
            l1_ratio=config.logistic_l1_ratio,
            max_iter=config.logistic_max_iter,
            tol=config.logistic_tol,
            solver=config.logistic_solver,
            class_weight=config.logistic_class_weight,
            random_state=config.random_state,
        )

    if config.model_name == "naive_bayes":
        return CategoricalNB(alpha=0.5)

    if config.model_name == "random_forest":
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
        from xgboost import XGBClassifier

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
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=1,
            gamma=config.xgb_gamma,
        )

    elif config.model_name == "xgboost":
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
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=1,
            gamma=config.xgb_gamma,
        )

    raise ValueError(f"No model builder exists for model '{config.model_name}'.")