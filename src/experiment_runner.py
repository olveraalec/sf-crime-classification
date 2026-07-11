from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import load_modeling_data
from src.evaluation import (
    align_probability_columns,
    evaluate_multiclass_predictions,
)
from src.experiment_config import ExperimentConfig
from src.logger import get_logger
from src.models import build_model
from src.pipeline_builder import (
    build_features_from_config,
    build_transformer_from_config,
)
from src.validation import (
    create_calendar_temporal_folds,
    create_frozen_temporal_test,
)


logger = get_logger(__name__)


def build_full_pipeline(
    config: ExperimentConfig,
) -> Pipeline:
    """Build a fitted preprocessing + model pipeline."""
    transformer = build_transformer_from_config(config)
    model = build_model(config)

    return Pipeline(
        steps=[
            ("transformer", transformer),
            ("model", model),
        ]
    )


def run_temporal_cv(
    config: ExperimentConfig,
    data: pd.DataFrame | None = None,
    validation_years: tuple[int, ...] = (2011, 2012, 2013),
) -> pd.DataFrame:
    """Run one configured experiment across calendar-based temporal folds."""
    config.validate()

    if data is None:
        data = load_modeling_data()

    development_data, _ = create_frozen_temporal_test(data)
    folds = create_calendar_temporal_folds(
        development_data,
        validation_years=validation_years,
    )

    development_data = development_data.sort_values("incident_timestamp").reset_index(
        drop=True
    )

    all_classes = np.sort(development_data["target"].unique())

    rows: list[dict[str, object]] = []

    for fold in folds:
        logger.info(
            "Starting experiment '%s', fold %s",
            config.experiment_name,
            fold.fold,
        )

        train_data = development_data.iloc[fold.train_indices].copy()

        validation_data = development_data.iloc[fold.validation_indices].copy()

        X_train = build_features_from_config(
            train_data,
            config,
        )

        X_validation = build_features_from_config(
            validation_data,
            config,
        )

        y_train = train_data["target"].copy()
        y_validation = validation_data["target"].copy()

        pipeline = build_full_pipeline(config)

        start_time = time.perf_counter()

        pipeline.fit(X_train, y_train)

        training_seconds = time.perf_counter() - start_time

        probabilities = pipeline.predict_proba(X_validation)

        model_classes = pipeline.named_steps["model"].classes_

        probabilities = align_probability_columns(
            probabilities=probabilities,
            model_classes=model_classes,
            all_classes=all_classes,
        )

        metrics = evaluate_multiclass_predictions(
            y_true=y_validation,
            probabilities=probabilities,
            classes=all_classes,
        )

        row = {
            **config.to_dict(),
            "fold": fold.fold,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "validation_start": fold.validation_start,
            "validation_end": fold.validation_end,
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
            "training_seconds": training_seconds,
            **metrics,
        }

        rows.append(row)

        logger.info(
            "Finished experiment '%s', fold %s: "
            "log_loss=%.6f, accuracy=%.6f, top_3=%.6f",
            config.experiment_name,
            fold.fold,
            metrics["log_loss"],
            metrics["accuracy"],
            metrics["top_3_accuracy"],
        )

    return pd.DataFrame(rows)


def summarize_experiment_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize fold-level metrics for one or more experiments."""
    metric_columns = [
        "log_loss",
        "accuracy",
        "top_3_accuracy",
        "macro_f1",
        "training_seconds",
    ]

    summary = fold_results.groupby(
        [
            "experiment_name",
            "model_name",
        ],
        as_index=False,
    )[metric_columns].agg(["mean", "std", "min", "max"])

    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in summary.columns.to_flat_index()
    ]

    return summary