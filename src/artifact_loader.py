from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import LabelEncoder

from src.artifact_contract import (
    FinalArtifactPaths,
    get_final_artifact_paths,
)
from src.config import get_project_root
from src.logger import get_logger


logger = get_logger(__name__)


REQUIRED_METADATA_FIELDS = frozenset(
    {
        "model_name",
        "training_rows",
        "raw_feature_columns",
        "transformed_feature_columns",
        "class_count",
        "classes",
        "config",
    }
)


class ArtifactError(RuntimeError):
    """Base exception for production artifact failures."""


class ArtifactLoadError(ArtifactError):
    """Raised when artifact files cannot be located or deserialized."""


class ArtifactValidationError(ArtifactError):
    """Raised when loaded artifacts are mutually incompatible."""


@dataclass(frozen=True)
class ArtifactBundle:
    """Validated production artifacts required by inference."""

    model: Any
    transformer: Any
    label_encoder: LabelEncoder
    metadata: dict[str, Any]
    paths: FinalArtifactPaths


def validate_artifact_paths(
    paths: FinalArtifactPaths,
) -> None:
    """Verify that every required production artifact exists."""
    missing_paths = [path for path in paths.all_paths() if not path.is_file()]

    if not missing_paths:
        return

    formatted_paths = "\n".join(f"- {path}" for path in missing_paths)

    raise ArtifactLoadError(f"Missing required model artifacts:\n{formatted_paths}")


def load_joblib_artifact(
    path: Path,
    artifact_name: str,
) -> Any:
    """Load one joblib artifact with a contextual error message."""
    try:
        return joblib.load(path)
    except Exception as error:
        raise ArtifactLoadError(
            f"Could not load {artifact_name} artifact: {path}"
        ) from error


def load_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load and validate the basic JSON structure of model metadata."""
    try:
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)
    except FileNotFoundError as error:
        raise ArtifactLoadError(f"Metadata file not found: {metadata_path}") from error
    except JSONDecodeError as error:
        raise ArtifactLoadError(
            f"Could not decode metadata JSON: {metadata_path}"
        ) from error
    except OSError as error:
        raise ArtifactLoadError(
            f"Could not read metadata file: {metadata_path}"
        ) from error

    if not isinstance(metadata, dict):
        raise ArtifactValidationError("Model metadata must contain a JSON object.")

    return metadata


def validate_metadata_fields(
    metadata: dict[str, Any],
) -> None:
    """Verify that metadata includes the production contract fields."""
    missing_fields = sorted(REQUIRED_METADATA_FIELDS.difference(metadata))

    if missing_fields:
        raise ArtifactValidationError(
            f"Model metadata is missing required fields: {', '.join(missing_fields)}"
        )


def validate_artifact_interfaces(
    *,
    model: Any,
    transformer: Any,
    label_encoder: Any,
) -> None:
    """Verify that loaded artifacts expose required inference methods."""
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactValidationError("Loaded model does not provide predict_proba().")

    if not callable(getattr(transformer, "transform", None)):
        raise ArtifactValidationError(
            "Loaded transformer does not provide transform()."
        )

    if not isinstance(label_encoder, LabelEncoder):
        raise ArtifactValidationError(
            "Loaded label encoder is not a sklearn LabelEncoder."
        )

    if not hasattr(label_encoder, "classes_"):
        raise ArtifactValidationError("Loaded label encoder has not been fitted.")


def validate_artifact_compatibility(
    *,
    label_encoder: LabelEncoder,
    metadata: dict[str, Any],
) -> None:
    """Verify that class metadata matches the fitted label encoder."""
    encoded_classes = label_encoder.classes_.tolist()
    metadata_classes = metadata["classes"]
    metadata_class_count = metadata["class_count"]

    if not isinstance(metadata_classes, list):
        raise ArtifactValidationError("Metadata field 'classes' must be a list.")

    if not isinstance(metadata_class_count, int):
        raise ArtifactValidationError(
            "Metadata field 'class_count' must be an integer."
        )

    if metadata_class_count != len(encoded_classes):
        raise ArtifactValidationError(
            "Metadata class_count does not match the fitted label encoder."
        )

    if metadata_class_count != len(metadata_classes):
        raise ArtifactValidationError(
            "Metadata class_count does not match the metadata class-label list."
        )

    if metadata_classes != encoded_classes:
        raise ArtifactValidationError(
            "Metadata class labels do not match the fitted label encoder."
        )


def validate_artifact_bundle(
    bundle: ArtifactBundle,
) -> None:
    """Validate the complete production artifact bundle."""
    validate_metadata_fields(bundle.metadata)

    validate_artifact_interfaces(
        model=bundle.model,
        transformer=bundle.transformer,
        label_encoder=bundle.label_encoder,
    )

    validate_artifact_compatibility(
        label_encoder=bundle.label_encoder,
        metadata=bundle.metadata,
    )


def load_artifact_bundle(
    model_directory: Path | None = None,
) -> ArtifactBundle:
    """Load and validate all artifacts needed for model inference."""
    resolved_model_directory = (
        model_directory
        if model_directory is not None
        else get_project_root() / "models"
    )

    paths = get_final_artifact_paths(resolved_model_directory)

    logger.info(
        "Loading production artifacts from: %s",
        resolved_model_directory,
    )

    validate_artifact_paths(paths)

    bundle = ArtifactBundle(
        model=load_joblib_artifact(
            paths.model,
            "model",
        ),
        transformer=load_joblib_artifact(
            paths.transformer,
            "transformer",
        ),
        label_encoder=load_joblib_artifact(
            paths.label_encoder,
            "label encoder",
        ),
        metadata=load_metadata(paths.metadata),
        paths=paths,
    )

    validate_artifact_bundle(bundle)

    logger.info(
        "Loaded artifact bundle: model=%s, classes=%s, transformed_features=%s",
        bundle.metadata["model_name"],
        bundle.metadata["class_count"],
        bundle.metadata["transformed_feature_columns"],
    )

    return bundle