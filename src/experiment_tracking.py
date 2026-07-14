from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_project_root
from src.experiment_config import ExperimentConfig
from src.logger import get_logger


logger = get_logger(__name__)


def create_run_id(config: ExperimentConfig) -> str:
    """Create a readable, unique identifier for an experiment run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.experiment_name}__{timestamp}"


def get_results_paths(
    run_id: str,
) -> tuple[Path, Path, Path]:
    """Return paths for fold results, summary results, and metadata."""
    project_root = get_project_root()

    experiments_dir = project_root / "results" / "experiments"
    summaries_dir = project_root / "results" / "summaries"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    fold_path = experiments_dir / f"{run_id}_folds.csv"
    summary_path = summaries_dir / f"{run_id}_summary.csv"
    metadata_path = experiments_dir / f"{run_id}_metadata.json"

    return fold_path, summary_path, metadata_path


def save_experiment_results(
    config: ExperimentConfig,
    fold_results: pd.DataFrame,
    summary_results: pd.DataFrame,
    run_id: str | None = None,
    additional_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save fold metrics, summary metrics, and experiment metadata."""
    if run_id is None:
        run_id = create_run_id(config)

    fold_path, summary_path, metadata_path = get_results_paths(run_id)

    fold_output = fold_results.copy()
    fold_output.insert(0, "run_id", run_id)

    summary_output = summary_results.copy()
    summary_output.insert(0, "run_id", run_id)

    fold_output.to_csv(fold_path, index=False)
    summary_output.to_csv(summary_path, index=False)

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config.to_dict(),
        "fold_result_file": str(fold_path.relative_to(get_project_root())),
        "summary_result_file": str(summary_path.relative_to(get_project_root())),
    }

    if additional_metadata:
        metadata.update(additional_metadata)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("Saved fold results to: %s", fold_path)
    logger.info("Saved summary results to: %s", summary_path)
    logger.info("Saved experiment metadata to: %s", metadata_path)

    return {
        "fold_results": fold_path,
        "summary_results": summary_path,
        "metadata": metadata_path,
    }


def rebuild_master_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine all saved experiment files into master Tableau-ready CSV files.
    """
    project_root = get_project_root()

    experiments_dir = project_root / "results" / "experiments"
    summaries_dir = project_root / "results" / "summaries"

    fold_files = sorted(experiments_dir.glob("*_folds.csv"))
    summary_files = sorted(summaries_dir.glob("*_summary.csv"))

    fold_frames = [pd.read_csv(path) for path in fold_files]
    summary_frames = [pd.read_csv(path) for path in summary_files]

    master_folds = (
        pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    )

    master_summaries = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )

    tableau_dir = project_root / "data" / "tableau"
    tableau_dir.mkdir(parents=True, exist_ok=True)

    master_fold_path = tableau_dir / "experiment_fold_results.csv"
    master_summary_path = tableau_dir / "experiment_summary_results.csv"

    master_folds.to_csv(master_fold_path, index=False)
    master_summaries.to_csv(master_summary_path, index=False)

    logger.info("Rebuilt master fold results: %s", master_fold_path)
    logger.info("Rebuilt master summary results: %s", master_summary_path)

    return master_folds, master_summaries