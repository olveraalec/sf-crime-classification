from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import get_project_root
from src.data_loader import load_modeling_data
from src.experiment_config import (
    ExperimentConfig,
    xgboost_finalist_config,
)
from src.logger import get_logger
from src.models import build_model
from src.pipeline_builder import (
    build_features_from_config,
    build_transformer_from_config,
)

from src.artifact_contract import (
    FinalArtifactPaths,
    get_final_artifact_paths,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class FinalTrainingResult:
    """Summary returned after final model training succeeds."""

    artifacts: FinalArtifactPaths
    training_rows: int
    raw_feature_columns: int
    transformed_feature_columns: int
    class_count: int

def build_final_metadata(
    *,
    training_rows: int,
    raw_feature_shape: tuple[int, int],
    transformed_feature_shape: tuple[int, int],
    classes: np.ndarray,
    config: dict[str, object],
    trained_at_utc: str,
) -> dict[str, Any]:
    """
    Build final model metadata.

    The evaluation metrics are preserved from the completed Version 2
    temporal-validation and frozen-test evaluation process.
    """
    return {
        "model_name": "xgboost_finalist",
        "trained_at_utc": trained_at_utc,
        "training_rows": int(training_rows),
        "raw_feature_rows": int(raw_feature_shape[0]),
        "raw_feature_columns": int(raw_feature_shape[1]),
        "transformed_feature_columns": int(transformed_feature_shape[1]),
        "class_count": int(len(classes)),
        "classes": classes.tolist(),
        "config": config,
        "validated_temporal_cv_log_loss_mean": 2.277305,
        "frozen_test_log_loss": 2.202763,
        "frozen_test_accuracy": 0.340791,
        "frozen_test_top_3_accuracy": 0.595975,
        "frozen_test_macro_f1": 0.068753,
        "expected_calibration_error": 0.011697,
    }


def save_metadata(
    metadata: dict[str, Any],
    metadata_path: Path,
) -> None:
    """Write model metadata as formatted UTF-8 JSON."""
    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def save_final_artifacts(
    *,
    model: object,
    transformer: object,
    label_encoder: LabelEncoder,
    metadata: dict[str, Any],
    artifact_paths: FinalArtifactPaths,
) -> None:
    """Persist all artifacts required for Version 3 inference."""
    artifact_paths.model.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        artifact_paths.model,
    )
    joblib.dump(
        transformer,
        artifact_paths.transformer,
    )
    joblib.dump(
        label_encoder,
        artifact_paths.label_encoder,
    )

    save_metadata(
        metadata,
        artifact_paths.metadata,
    )


def train_and_save_final_model(
    *,
    model_directory: Path | None = None,
    config: ExperimentConfig | None = None,
) -> FinalTrainingResult:
    """
    Train the finalized XGBoost model using all available modeling data.

    The deterministic feature frame, learned transformer, and estimator
    are all created from the same centralized experiment configuration.
    """
    project_root = get_project_root()

    resolved_model_directory = (
        model_directory if model_directory is not None else project_root / "models"
    )

    resolved_config = config if config is not None else xgboost_finalist_config()

    resolved_config.validate()

    artifact_paths = get_final_artifact_paths(resolved_model_directory)

    logger.info("Loading final modeling dataset.")
    data = load_modeling_data()

    logger.info(
        "Building deterministic features using configuration: %s",
        resolved_config.experiment_name,
    )
    features = build_features_from_config(
        data,
        resolved_config,
    )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(data["target"])

    logger.info("Fitting final preprocessing transformer.")
    transformer = build_transformer_from_config(resolved_config)
    transformed_features = transformer.fit_transform(features)

    logger.info("Training final estimator.")
    model = build_model(resolved_config)
    model.fit(
        transformed_features,
        encoded_target,
    )

    trained_at_utc = datetime.now(timezone.utc).isoformat()

    metadata = build_final_metadata(
        training_rows=len(data),
        raw_feature_shape=features.shape,
        transformed_feature_shape=transformed_features.shape,
        classes=label_encoder.classes_,
        config=resolved_config.to_dict(),
        trained_at_utc=trained_at_utc,
    )

    save_final_artifacts(
        model=model,
        transformer=transformer,
        label_encoder=label_encoder,
        metadata=metadata,
        artifact_paths=artifact_paths,
    )

    result = FinalTrainingResult(
        artifacts=artifact_paths,
        training_rows=int(len(data)),
        raw_feature_columns=int(features.shape[1]),
        transformed_feature_columns=int(transformed_features.shape[1]),
        class_count=int(len(label_encoder.classes_)),
    )

    logger.info(
        "Final artifacts saved to: %s",
        resolved_model_directory,
    )
    logger.info(
        "Training rows=%s, raw features=%s, transformed features=%s, classes=%s",
        result.training_rows,
        result.raw_feature_columns,
        result.transformed_feature_columns,
        result.class_count,
    )

    return result


if __name__ == "__main__":
    train_and_save_final_model()
