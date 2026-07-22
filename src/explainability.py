from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_modeling_data
from src.experiment_config import xgboost_finalist_config
from src.features import build_feature_frame
from src.models import build_model
from src.transformers import build_tree_transformer


FIGURE_DIR = Path("figures/explainability")
TABLEAU_DIR = Path("data/tableau")

SHAP_SAMPLE_SIZE = 2_000
SHAP_CLASS_NAME = "LARCENY/THEFT"
RANDOM_STATE = 12345


def get_feature_names(transformer, n_features: int) -> list[str]:
    """
    Return transformed feature names when the fitted transformer supports
    get_feature_names_out. Fall back to XGBoost-style feature indices.
    """
    try:
        names = transformer.get_feature_names_out()
        return [str(name) for name in names]
    except (AttributeError, ValueError, TypeError):
        return [f"f{i}" for i in range(n_features)]


def train_final_xgboost():
    """
    Train the finalized XGBoost model on the complete modeling dataset.

    This full-data fit is used only for final descriptive explainability
    after temporal model selection has been completed.
    """
    config = xgboost_finalist_config()

    df = load_modeling_data()

    X = build_feature_frame(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["target"])

    transformer = build_tree_transformer(
        geo_mode=config.geo_mode,
        n_geo_clusters=config.n_geo_clusters,
    )

    X_transformed = transformer.fit_transform(X)

    feature_names = get_feature_names(
        transformer=transformer,
        n_features=X_transformed.shape[1],
    )

    model = build_model(config)
    model.fit(X_transformed, y)

    # Attach meaningful names to the underlying XGBoost booster.
    booster = model.get_booster()
    booster.feature_names = feature_names

    return {
        "model": model,
        "transformer": transformer,
        "X_transformed": X_transformed,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
    }


def save_xgboost_feature_importance(
    model,
    feature_names: list[str],
) -> None:
    """Save XGBoost weight, gain, and cover importance plots."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    booster.feature_names = feature_names

    for importance_type in ["weight", "gain", "cover"]:
        fig, ax = plt.subplots(figsize=(12, 9))

        xgb.plot_importance(
            booster,
            importance_type=importance_type,
            max_num_features=20,
            show_values=False,
            ax=ax,
        )

        ax.set_title(
            f"XGBoost Feature Importance: {importance_type.replace('_', ' ').title()}"
        )
        ax.set_xlabel(importance_type.replace("_", " ").title())
        ax.set_ylabel("Feature")

        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"xgb_importance_{importance_type}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def normalize_multiclass_shap_values(
    shap_values,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """
    Normalize SHAP multiclass output to:

        (n_samples, n_features, n_classes)

    Different SHAP versions may return a list or different array order.
    """
    if isinstance(shap_values, list):
        # List with one (n_samples, n_features) array per class.
        values = np.stack(shap_values, axis=-1)
    else:
        values = np.asarray(shap_values)

    if values.shape == (n_samples, n_features, n_classes):
        return values

    if values.shape == (n_classes, n_samples, n_features):
        return np.transpose(values, (1, 2, 0))

    if values.shape == (n_samples, n_classes, n_features):
        return np.transpose(values, (0, 2, 1))

    raise ValueError(
        "Unexpected multiclass SHAP shape: "
        f"{values.shape}. Expected dimensions compatible with "
        f"samples={n_samples}, features={n_features}, "
        f"classes={n_classes}."
    )


def calculate_shap_values(
    model,
    X_transformed,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate multiclass SHAP values on a reproducible sample."""
    rng = np.random.default_rng(RANDOM_STATE)

    sample_size = min(SHAP_SAMPLE_SIZE, X_transformed.shape[0])
    sample_indices = rng.choice(
        X_transformed.shape[0],
        size=sample_size,
        replace=False,
    )

    X_sample = X_transformed[sample_indices]

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_sample)

    shap_values = normalize_multiclass_shap_values(
        shap_values=raw_shap_values,
        n_samples=X_sample.shape[0],
        n_features=X_sample.shape[1],
        n_classes=n_classes,
    )

    return X_sample, shap_values


def save_global_shap_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Aggregate absolute SHAP magnitude over both observations and classes.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLEAU_DIR.mkdir(parents=True, exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_absolute_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    importance_df.to_csv(
        TABLEAU_DIR / "xgboost_shap_global_importance.csv",
        index=False,
    )

    plot_df = importance_df.head(20).sort_values(
        "mean_absolute_shap",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(
        plot_df["feature"],
        plot_df["mean_absolute_shap"],
    )
    ax.set_title("Global SHAP Feature Importance Across 39 Crime Classes")
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return importance_df


def save_class_shap_summary(
    X_sample,
    shap_values: np.ndarray,
    feature_names: list[str],
    label_encoder: LabelEncoder,
    class_name: str = SHAP_CLASS_NAME,
) -> None:
    """Save a SHAP beeswarm plot for one selected crime class."""
    matches = np.where(label_encoder.classes_ == class_name)[0]

    if len(matches) == 0:
        raise ValueError(f"Requested SHAP class was not found: {class_name}")

    class_index = int(matches[0])
    class_shap_values = shap_values[:, :, class_index]

    shap.summary_plot(
        class_shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )

    plt.title(f"SHAP Summary: {class_name}")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "shap_class_larceny_theft.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    artifacts = train_final_xgboost()

    model = artifacts["model"]
    X_transformed = artifacts["X_transformed"]
    feature_names = artifacts["feature_names"]
    label_encoder = artifacts["label_encoder"]

    save_xgboost_feature_importance(
        model=model,
        feature_names=feature_names,
    )

    X_sample, shap_values = calculate_shap_values(
        model=model,
        X_transformed=X_transformed,
        n_classes=len(label_encoder.classes_),
    )

    importance_df = save_global_shap_importance(
        shap_values=shap_values,
        feature_names=feature_names,
    )

    save_class_shap_summary(
        X_sample=X_sample,
        shap_values=shap_values,
        feature_names=feature_names,
        label_encoder=label_encoder,
    )

    print("\nExplainability artifacts created successfully.")
    print("\nTop 20 global SHAP features:")
    print(importance_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()