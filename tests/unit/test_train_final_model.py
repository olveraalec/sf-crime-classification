from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.artifact_contract import (
    FinalArtifactPaths,
    get_final_artifact_paths,
)
from src.train_final_model import (
    FinalTrainingResult,
    build_final_metadata,
    save_metadata,
)


def test_get_final_artifact_paths_preserves_version_2_names(
    tmp_path: Path,
) -> None:
    paths = get_final_artifact_paths(tmp_path)

    assert paths == FinalArtifactPaths(
        model=tmp_path / "xgboost_final_model.joblib",
        transformer=tmp_path / "xgboost_final_transformer.joblib",
        label_encoder=tmp_path / "xgboost_label_encoder.joblib",
        metadata=tmp_path / "xgboost_final_metadata.json",
    )


def test_build_final_metadata_uses_runtime_training_values() -> None:
    classes = np.array(
        [
            "ASSAULT",
            "LARCENY/THEFT",
            "OTHER OFFENSES",
        ]
    )

    metadata = build_final_metadata(
        training_rows=100,
        raw_feature_shape=(100, 30),
        transformed_feature_shape=(100, 70),
        classes=classes,
        config={
            "experiment_name": "xgboost_finalist",
            "model_name": "xgboost",
        },
        trained_at_utc="2026-07-24T20:00:00+00:00",
    )

    assert metadata["model_name"] == "xgboost_finalist"
    assert metadata["trained_at_utc"] == ("2026-07-24T20:00:00+00:00")
    assert metadata["training_rows"] == 100
    assert metadata["raw_feature_rows"] == 100
    assert metadata["raw_feature_columns"] == 30
    assert metadata["transformed_feature_columns"] == 70
    assert metadata["class_count"] == 3
    assert metadata["classes"] == [
        "ASSAULT",
        "LARCENY/THEFT",
        "OTHER OFFENSES",
    ]


def test_build_final_metadata_preserves_validated_metrics() -> None:
    metadata = build_final_metadata(
        training_rows=10,
        raw_feature_shape=(10, 30),
        transformed_feature_shape=(10, 70),
        classes=np.array(["A", "B"]),
        config={
            "experiment_name": "xgboost_finalist",
            "model_name": "xgboost",
        },
        trained_at_utc="2026-07-24T20:00:00+00:00",
    )

    assert metadata["validated_temporal_cv_log_loss_mean"] == 2.277305
    assert metadata["frozen_test_log_loss"] == 2.202763
    assert metadata["frozen_test_accuracy"] == 0.340791
    assert metadata["frozen_test_top_3_accuracy"] == 0.595975
    assert metadata["frozen_test_macro_f1"] == 0.068753
    assert metadata["expected_calibration_error"] == 0.011697


def test_save_metadata_writes_readable_json(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "model_name": "xgboost_finalist",
        "training_rows": 100,
    }

    save_metadata(metadata, metadata_path)

    assert metadata_path.exists()

    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert loaded == metadata


def test_final_training_result_records_artifacts_and_shapes(
    tmp_path: Path,
) -> None:
    paths = get_final_artifact_paths(tmp_path)

    result = FinalTrainingResult(
        artifacts=paths,
        training_rows=100,
        raw_feature_columns=30,
        transformed_feature_columns=70,
        class_count=39,
    )

    assert result.artifacts == paths
    assert result.training_rows == 100
    assert result.raw_feature_columns == 30
    assert result.transformed_feature_columns == 70
    assert result.class_count == 39