from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FINAL_MODEL_FILENAME = "xgboost_final_model.joblib"
FINAL_TRANSFORMER_FILENAME = "xgboost_final_transformer.joblib"
FINAL_LABEL_ENCODER_FILENAME = "xgboost_label_encoder.joblib"
FINAL_METADATA_FILENAME = "xgboost_final_metadata.json"


@dataclass(frozen=True)
class FinalArtifactPaths:
    """Filesystem paths for all artifacts required during inference."""

    model: Path
    transformer: Path
    label_encoder: Path
    metadata: Path

    def all_paths(self) -> tuple[Path, ...]:
        """Return every required artifact path."""
        return (
            self.model,
            self.transformer,
            self.label_encoder,
            self.metadata,
        )


def get_final_artifact_paths(
    model_directory: Path,
) -> FinalArtifactPaths:
    """Construct the standard production artifact paths."""
    return FinalArtifactPaths(
        model=model_directory / FINAL_MODEL_FILENAME,
        transformer=model_directory / FINAL_TRANSFORMER_FILENAME,
        label_encoder=(model_directory / FINAL_LABEL_ENCODER_FILENAME),
        metadata=model_directory / FINAL_METADATA_FILENAME,
    )