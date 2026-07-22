import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_modeling_data
from src.experiment_config import xgboost_finalist_config
from src.features import build_feature_frame
from src.models import build_model
from src.transformers import build_tree_transformer


MODEL_DIR = Path("models")


def train_and_save_final_model() -> None:
    """
    Train the finalized XGBoost pipeline on all available modeling data
    after model selection and frozen-test evaluation are complete.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = xgboost_finalist_config()
    df = load_modeling_data()

    X = build_feature_frame(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["target"])

    # Keep this transformer call identical to the working call in
    # explainability.py and evaluate_final_model.py.
    transformer = build_tree_transformer(
        geo_mode=config.geo_mode,
        n_geo_clusters=config.n_geo_clusters,
    )

    X_transformed = transformer.fit_transform(X)

    model = build_model(config)
    model.fit(X_transformed, y)

    joblib.dump(
        model,
        MODEL_DIR / "xgboost_final_model.joblib",
    )
    joblib.dump(
        transformer,
        MODEL_DIR / "xgboost_final_transformer.joblib",
    )
    joblib.dump(
        label_encoder,
        MODEL_DIR / "xgboost_label_encoder.joblib",
    )

    metadata = {
        "model_name": "xgboost_finalist",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(df)),
        "raw_feature_rows": int(X.shape[0]),
        "raw_feature_columns": int(X.shape[1]),
        "transformed_feature_columns": int(X_transformed.shape[1]),
        "class_count": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
        "config": config.to_dict(),
        "validated_temporal_cv_log_loss_mean": 2.277305,
        "frozen_test_log_loss": 2.202763,
        "frozen_test_accuracy": 0.340791,
        "frozen_test_top_3_accuracy": 0.595975,
        "frozen_test_macro_f1": 0.068753,
        "expected_calibration_error": 0.011697,
    }

    with (MODEL_DIR / "xgboost_final_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nFinal model artifacts saved:")
    for path in sorted(MODEL_DIR.glob("xgboost_final_*")):
        print(f"  {path}")

    print(f"  {MODEL_DIR / 'xgboost_label_encoder.joblib'}")


if __name__ == "__main__":
    train_and_save_final_model()