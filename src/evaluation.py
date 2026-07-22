from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)


def align_probability_columns(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    all_classes: np.ndarray,
) -> np.ndarray:
    """
    Align model probability columns to a fixed global class order.

    This is important when a validation or test period lacks one or more
    rare classes.
    """
    aligned = np.zeros(
        (probabilities.shape[0], len(all_classes)),
        dtype=float,
    )

    class_to_index = {
        class_label: index for index, class_label in enumerate(all_classes)
    }

    for model_index, class_label in enumerate(model_classes):
        aligned[:, class_to_index[class_label]] = probabilities[:, model_index]

    row_sums = aligned.sum(axis=1, keepdims=True)

    if np.any(row_sums == 0):
        raise ValueError("At least one prediction row has zero total probability.")

    aligned = aligned / row_sums

    return aligned


def evaluate_multiclass_predictions(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    """Calculate the project's main multiclass metrics."""
    y_true_array = np.asarray(y_true)

    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = classes[predicted_indices]

    metrics = {
        "log_loss": log_loss(
            y_true_array,
            probabilities,
            labels=classes,
        ),
        "accuracy": accuracy_score(
            y_true_array,
            predicted_labels,
        ),
        "top_3_accuracy": top_k_accuracy_score(
            y_true_array,
            probabilities,
            k=min(3, len(classes)),
            labels=classes,
        ),
        "macro_f1": f1_score(
            y_true_array,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics