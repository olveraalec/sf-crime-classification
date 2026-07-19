from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_modeling_data
from src.experiment_config import xgboost_finalist_config
from src.features import build_feature_frame
from src.models import build_model
from src.transformers import build_tree_transformer
from src.validation import create_frozen_temporal_test


FIGURE_DIR = Path("figures/evaluation")
TABLEAU_DIR = Path("data/tableau")

TOP_CLASS_COUNT = 15


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    all_classes: np.ndarray,
) -> np.ndarray:
    """
    Align model probability columns to the complete encoded class order.

    This protects evaluation when a class is absent from the development
    period but appears in the frozen test period.
    """
    aligned = np.zeros(
        (probabilities.shape[0], len(all_classes)),
        dtype=float,
    )

    class_to_column = {
        class_value: column_index
        for column_index, class_value in enumerate(all_classes)
    }

    for model_column, class_value in enumerate(model_classes):
        aligned_column = class_to_column[class_value]
        aligned[:, aligned_column] = probabilities[:, model_column]

    row_sums = aligned.sum(axis=1, keepdims=True)

    aligned = np.divide(
        aligned,
        row_sums,
        out=np.full_like(aligned, 1.0 / len(all_classes)),
        where=row_sums > 0,
    )

    return aligned


def prepare_final_evaluation():
    """Train the finalist and return frozen-test predictions."""
    config = xgboost_finalist_config()

    df = load_modeling_data()
    development, frozen_test = create_frozen_temporal_test(df)

    label_encoder = LabelEncoder()
    label_encoder.fit(df["target"])

    y_train = label_encoder.transform(development["target"])
    y_test = label_encoder.transform(frozen_test["target"])

    X_train = build_feature_frame(development)
    X_test = build_feature_frame(frozen_test)

    transformer = build_tree_transformer(
        geo_mode=config.geo_mode,
        n_geo_clusters=config.n_geo_clusters,
    )

    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    model = build_model(config)
    model.fit(X_train_transformed, y_train)

    raw_probabilities = model.predict_proba(X_test_transformed)

    all_encoded_classes = np.arange(len(label_encoder.classes_))

    probabilities = align_probabilities(
        probabilities=raw_probabilities,
        model_classes=model.classes_,
        all_classes=all_encoded_classes,
    )

    predictions = probabilities.argmax(axis=1)

    return {
        "development": development,
        "frozen_test": frozen_test,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "label_encoder": label_encoder,
    }


def calculate_overall_metrics(
    y_test: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> pd.DataFrame:
    """Calculate final frozen-test model metrics."""
    classes = np.arange(class_count)

    metrics = {
        "experiment_name": "xgboost_finalist_frozen_test",
        "log_loss": log_loss(
            y_test,
            probabilities,
            labels=classes,
        ),
        "accuracy": accuracy_score(y_test, predictions),
        "top_3_accuracy": top_k_accuracy_score(
            y_test,
            probabilities,
            k=3,
            labels=classes,
        ),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "test_rows": len(y_test),
    }

    return pd.DataFrame([metrics])


def calculate_per_class_metrics(
    y_test: np.ndarray,
    predictions: np.ndarray,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Create a Tableau-ready per-class performance table."""
    class_names = label_encoder.classes_
    encoded_classes = np.arange(len(class_names))

    report = classification_report(
        y_test,
        predictions,
        labels=encoded_classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for class_name in class_names:
        class_metrics = report[class_name]

        rows.append(
            {
                "crime_category": class_name,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1_score": class_metrics["f1-score"],
                "support": int(class_metrics["support"]),
            }
        )

    metrics_df = pd.DataFrame(rows)

    metrics_df["f1_rank"] = (
        metrics_df["f1_score"].rank(method="min", ascending=False).astype(int)
    )

    return metrics_df.sort_values(
        ["f1_score", "support"],
        ascending=[False, False],
    ).reset_index(drop=True)


def save_full_confusion_matrix(
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: np.ndarray,
) -> None:
    """Save a row-normalized confusion matrix for all crime classes."""
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(22, 18))

    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Fraction of True Class"},
        ax=ax,
    )

    ax.set_title("XGBoost Frozen-Test Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Crime Category")
    ax.set_ylabel("True Crime Category")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "xgboost_confusion_matrix_all_classes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_top_class_confusion_matrix(
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: np.ndarray,
) -> None:
    """
    Save a more readable confusion matrix for the most common test classes.
    """
    class_counts = pd.Series(y_test).value_counts()

    top_encoded_classes = class_counts.head(TOP_CLASS_COUNT).index.to_numpy()

    mask = np.isin(y_test, top_encoded_classes)

    matrix = confusion_matrix(
        y_test[mask],
        predictions[mask],
        labels=top_encoded_classes,
        normalize="true",
    )

    top_class_names = class_names[top_encoded_classes]

    fig, ax = plt.subplots(figsize=(15, 12))

    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=top_class_names,
        yticklabels=top_class_names,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Fraction of True Class"},
        ax=ax,
    )

    ax.set_title(
        f"XGBoost Confusion Matrix: {TOP_CLASS_COUNT} Most Common Crime Classes"
    )
    ax.set_xlabel("Predicted Crime Category")
    ax.set_ylabel("True Crime Category")

    plt.xticks(rotation=75, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "xgboost_confusion_matrix_top_classes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_class_performance_plot(
    per_class_metrics: pd.DataFrame,
) -> None:
    """Plot F1 scores for classes with meaningful frozen-test support."""
    plot_df = (
        per_class_metrics[per_class_metrics["support"] >= 100]
        .sort_values("f1_score", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.barh(
        plot_df["crime_category"],
        plot_df["f1_score"],
    )

    ax.set_title("Best-Supported Crime Classes by Frozen-Test F1 Score")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Crime Category")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "xgboost_per_class_f1.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_outputs(artifacts: dict) -> None:
    """Calculate and save final diagnostic artifacts."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)

    y_test = artifacts["y_test"]
    predictions = artifacts["predictions"]
    probabilities = artifacts["probabilities"]
    label_encoder = artifacts["label_encoder"]

    overall_metrics = calculate_overall_metrics(
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        class_count=len(label_encoder.classes_),
    )

    per_class_metrics = calculate_per_class_metrics(
        y_test=y_test,
        predictions=predictions,
        label_encoder=label_encoder,
    )

    overall_metrics.to_csv(
        TABLEAU_DIR / "xgboost_frozen_test_metrics.csv",
        index=False,
    )

    per_class_metrics.to_csv(
        TABLEAU_DIR / "xgboost_per_class_metrics.csv",
        index=False,
    )

    save_full_confusion_matrix(
        y_test=y_test,
        predictions=predictions,
        class_names=label_encoder.classes_,
    )

    save_top_class_confusion_matrix(
        y_test=y_test,
        predictions=predictions,
        class_names=label_encoder.classes_,
    )

    save_class_performance_plot(per_class_metrics)

    print("\nFrozen-test metrics:")
    print(overall_metrics.to_string(index=False))

    print("\nTop 10 classes by F1:")
    print(per_class_metrics.head(10).to_string(index=False))

    print("\nBottom 10 supported classes by F1:")
    print(
        per_class_metrics[per_class_metrics["support"] >= 100]
        .tail(10)
        .to_string(index=False)
    )


def main() -> None:
    artifacts = prepare_final_evaluation()
    save_outputs(artifacts)

    print("\nFinal evaluation artifacts created successfully.")


if __name__ == "__main__":
    main()