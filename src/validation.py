from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class TemporalFold:
    """Container for one expanding-window temporal validation fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_indices: np.ndarray
    validation_indices: np.ndarray


def create_original_random_split(
    data: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.40,
    random_state: int = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the original Version 1 stratified random split."""
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[target_column],
    )

    logger.info(
        "Created original random split: train=%s, test=%s",
        len(train_data),
        len(test_data),
    )

    return train_data.copy(), test_data.copy()


def create_frozen_temporal_test(
    data: pd.DataFrame,
    timestamp_column: str = "incident_timestamp",
    test_start: str = "2014-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a frozen future test set using an explicit date boundary."""
    boundary = pd.Timestamp(test_start)

    sorted_data = data.sort_values(timestamp_column).reset_index(drop=True).copy()

    development_data = sorted_data[sorted_data[timestamp_column] < boundary].copy()

    frozen_test_data = sorted_data[sorted_data[timestamp_column] >= boundary].copy()

    if development_data.empty:
        raise ValueError("Development dataset is empty.")

    if frozen_test_data.empty:
        raise ValueError("Frozen test dataset is empty.")

    logger.info(
        "Created frozen temporal test split: development=%s, frozen_test=%s",
        len(development_data),
        len(frozen_test_data),
    )

    logger.info(
        "Development period: %s to %s",
        development_data[timestamp_column].min(),
        development_data[timestamp_column].max(),
    )

    logger.info(
        "Frozen test period: %s to %s",
        frozen_test_data[timestamp_column].min(),
        frozen_test_data[timestamp_column].max(),
    )

    return development_data, frozen_test_data


def create_calendar_temporal_folds(
    development_data: pd.DataFrame,
    timestamp_column: str = "incident_timestamp",
    validation_years: tuple[int, ...] = (2011, 2012, 2013),
) -> list[TemporalFold]:
    """Create expanding-window folds using full calendar years."""
    sorted_data = (
        development_data.sort_values(timestamp_column).reset_index(drop=True).copy()
    )

    folds: list[TemporalFold] = []

    for fold_number, validation_year in enumerate(
        validation_years,
        start=1,
    ):
        validation_start = pd.Timestamp(f"{validation_year}-01-01")
        validation_end = pd.Timestamp(f"{validation_year + 1}-01-01")

        train_mask = sorted_data[timestamp_column] < validation_start

        validation_mask = (sorted_data[timestamp_column] >= validation_start) & (
            sorted_data[timestamp_column] < validation_end
        )

        train_indices = np.flatnonzero(train_mask.to_numpy())
        validation_indices = np.flatnonzero(validation_mask.to_numpy())

        if len(train_indices) == 0:
            raise ValueError(f"Fold {fold_number} has no training observations.")

        if len(validation_indices) == 0:
            raise ValueError(f"Fold {fold_number} has no validation observations.")

        train_dates = sorted_data.iloc[train_indices][timestamp_column]

        validation_dates = sorted_data.iloc[validation_indices][timestamp_column]

        fold = TemporalFold(
            fold=fold_number,
            train_start=train_dates.min(),
            train_end=train_dates.max(),
            validation_start=validation_dates.min(),
            validation_end=validation_dates.max(),
            train_indices=train_indices,
            validation_indices=validation_indices,
        )

        folds.append(fold)

        logger.info(
            "Temporal fold %s: train=%s rows, validation=%s rows",
            fold_number,
            len(train_indices),
            len(validation_indices),
        )

    return folds


def validate_class_coverage(
    reference_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    target_column: str = "target",
) -> dict[str, object]:
    """Compare target-class coverage between two datasets."""
    reference_classes = set(reference_data[target_column].unique())

    comparison_classes = set(comparison_data[target_column].unique())

    return {
        "reference_class_count": len(reference_classes),
        "comparison_class_count": len(comparison_classes),
        "missing_from_comparison": sorted(reference_classes - comparison_classes),
        "unseen_in_comparison": sorted(comparison_classes - reference_classes),
    }


def summarize_temporal_folds(
    development_data: pd.DataFrame,
    folds: list[TemporalFold],
    timestamp_column: str = "incident_timestamp",
    target_column: str = "target",
) -> pd.DataFrame:
    """Create a reporting table for temporal validation folds."""
    sorted_data = development_data.sort_values(timestamp_column).reset_index(drop=True)

    rows = []

    for fold in folds:
        train_data = sorted_data.iloc[fold.train_indices]
        validation_data = sorted_data.iloc[fold.validation_indices]

        coverage = validate_class_coverage(
            train_data,
            validation_data,
            target_column=target_column,
        )

        rows.append(
            {
                "fold": fold.fold,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_rows": len(train_data),
                "validation_rows": len(validation_data),
                "train_classes": coverage["reference_class_count"],
                "validation_classes": coverage["comparison_class_count"],
                "unseen_validation_classes": len(coverage["unseen_in_comparison"]),
            }
        )

    return pd.DataFrame(rows)