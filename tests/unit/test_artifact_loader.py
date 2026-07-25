from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from src.artifact_contract import get_final_artifact_paths
from src.artifact_loader import (
    ArtifactLoadError,
    ArtifactValidationError,
    load_artifact_bundle,
    load_metadata,
    validate_artifact_paths,
)


class FakeProbabilityModel:
    """Small pickleable model double for artifact-loading tests."""

    def predict_proba(self, features: object) -> np.ndarray:
        del features
        return np.array([[0.25, 0.75]])


class FakeTransformer:
    """Small pickleable transformer double for artifact-loading tests."""

    def transform(self, features: object) -> np.ndarray:
        del features
        return np.ones((1, 70))


def create_valid_artifacts(
    model_directory: Path,
) -> None:
    """Create a small, valid artifact set without training a real model."""
    paths = get_final_artifact_paths(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)

    model = FakeProbabilityModel()
    transformer = FakeTransformer()

    label_encoder = LabelEncoder()
    label_encoder.fit(
        [
            "ASSAULT",
            "LARCENY/THEFT",
        ]
    )

    metadata = {
        "model_name": "xgboost_finalist",
        "training_rows": 100,
        "raw_feature_columns": 30,
        "transformed_feature_columns": 70,
        "class_count": 2,
        "classes": [
            "ASSAULT",
            "LARCENY/THEFT",
        ],
        "config": {
            "experiment_name": "xgboost_finalist",
            "model_name": "xgboost",
        },
    }

    joblib.dump(model, paths.model)
    joblib.dump(transformer, paths.transformer)
    joblib.dump(label_encoder, paths.label_encoder)

    paths.metadata.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

def test_validate_artifact_paths_accepts_complete_set(
    tmp_path: Path,
) -> None:
    create_valid_artifacts(tmp_path)
    paths = get_final_artifact_paths(tmp_path)

    validate_artifact_paths(paths)


def test_validate_artifact_paths_reports_all_missing_files(
    tmp_path: Path,
) -> None:
    paths = get_final_artifact_paths(tmp_path)

    with pytest.raises(
        ArtifactLoadError,
        match="Missing required model artifacts",
    ) as error:
        validate_artifact_paths(paths)

    message = str(error.value)

    assert "xgboost_final_model.joblib" in message
    assert "xgboost_final_transformer.joblib" in message
    assert "xgboost_label_encoder.joblib" in message
    assert "xgboost_final_metadata.json" in message


def test_load_metadata_reads_valid_json(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "xgboost_finalist",
            }
        ),
        encoding="utf-8",
    )

    metadata = load_metadata(metadata_path)

    assert metadata["model_name"] == "xgboost_finalist"


def test_load_metadata_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactLoadError,
        match="Could not decode metadata JSON",
    ):
        load_metadata(metadata_path)


def test_load_artifact_bundle_returns_validated_objects(
    tmp_path: Path,
) -> None:
    create_valid_artifacts(tmp_path)

    bundle = load_artifact_bundle(tmp_path)

    assert hasattr(bundle.model, "predict_proba")
    assert hasattr(bundle.transformer, "transform")

    assert bundle.label_encoder.classes_.tolist() == [
        "ASSAULT",
        "LARCENY/THEFT",
    ]

    assert bundle.metadata["class_count"] == 2
    assert bundle.paths == get_final_artifact_paths(tmp_path)


def test_loader_rejects_metadata_missing_required_fields(
    tmp_path: Path,
) -> None:
    create_valid_artifacts(tmp_path)
    paths = get_final_artifact_paths(tmp_path)

    paths.metadata.write_text(
        json.dumps(
            {
                "model_name": "xgboost_finalist",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactValidationError,
        match="missing required fields",
    ):
        load_artifact_bundle(tmp_path)


def test_loader_rejects_class_count_mismatch(
    tmp_path: Path,
) -> None:
    create_valid_artifacts(tmp_path)
    paths = get_final_artifact_paths(tmp_path)

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    metadata["class_count"] = 39

    paths.metadata.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactValidationError,
        match="class_count",
    ):
        load_artifact_bundle(tmp_path)


def test_loader_rejects_class_label_mismatch(
    tmp_path: Path,
) -> None:
    create_valid_artifacts(tmp_path)
    paths = get_final_artifact_paths(tmp_path)

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    metadata["classes"] = [
        "ASSAULT",
        "ROBBERY",
    ]

    paths.metadata.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ArtifactValidationError,
        match="class labels",
    ):
        load_artifact_bundle(tmp_path)


def test_loader_rejects_model_without_predict_proba(
    tmp_path: Path,
) -> None:
    create_valid_artifacts(tmp_path)
    paths = get_final_artifact_paths(tmp_path)

    invalid_model = object()
    joblib.dump(invalid_model, paths.model)

    with pytest.raises(
        ArtifactValidationError,
        match="predict_proba",
    ):
        load_artifact_bundle(tmp_path)


def test_loader_rejects_transformer_without_transform(
    tmp_path: Path,
) -> None:
    create_valid_artifacts(tmp_path)
    paths = get_final_artifact_paths(tmp_path)

    invalid_transformer = object()
    joblib.dump(invalid_transformer, paths.transformer)

    with pytest.raises(
        ArtifactValidationError,
        match="transform",
    ):
        load_artifact_bundle(tmp_path)
