from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.artifact_contract import get_final_artifact_paths
from src.artifact_loader import ArtifactBundle
from src.incident_adapter import RawIncident
from src.inference_engine import InferenceEngine
from src.prediction_service import PredictionService


class FakeFittedTransformer:
    """Represent the saved Version 2 preprocessing transformer."""

    def transform(self, features):
        assert features.shape == (1, 30)
        return np.ones((1, 70))


class FakeFittedModel:
    """Represent the saved final multiclass model."""

    def predict_proba(self, features):
        assert features.shape == (1, 70)

        return np.array(
            [
                [0.10, 0.70, 0.20],
            ]
        )


def test_raw_incident_reaches_decoded_prediction(
    tmp_path: Path,
) -> None:
    label_encoder = LabelEncoder()
    label_encoder.fit(
        [
            "ASSAULT",
            "LARCENY/THEFT",
            "ROBBERY",
        ]
    )

    metadata = {
        "model_name": "xgboost_finalist",
        "training_rows": 100,
        "raw_feature_columns": 30,
        "transformed_feature_columns": 70,
        "class_count": 3,
        "classes": label_encoder.classes_.tolist(),
        "config": {
            "experiment_name": "xgboost_finalist",
            "model_name": "xgboost",
            "validation_mode": "temporal_cv",
            "include_time_trend": True,
            "add_cyclical": True,
            "drop_original_cyclical": False,
            "add_interactions": True,
            "add_address_engineering": True,
            "categorical_encoding": "ordinal",
            "numeric_strategy": "passthrough",
            "geo_mode": "raw_distances",
            "n_geo_clusters": 40,
            "numeric_bins": 10,
            "sparse_output": False,
            "logistic_l1_ratio": 0.0,
            "logistic_class_weight": None,
            "logistic_c": 0.1,
            "logistic_max_iter": 500,
            "logistic_tol": 0.0001,
            "logistic_solver": "saga",
            "forest_n_estimators": 200,
            "forest_criterion": "log_loss",
            "forest_max_depth": 24,
            "forest_min_samples_split": 2,
            "forest_min_samples_leaf": 5,
            "forest_max_features": "sqrt",
            "forest_bootstrap": True,
            "forest_class_weight": None,
            "forest_n_jobs": -1,
            "xgb_n_estimators": 600,
            "xgb_max_depth": 8,
            "xgb_learning_rate": 0.03,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.7,
            "xgb_min_child_weight": 5,
            "xgb_reg_alpha": 0.0,
            "xgb_reg_lambda": 12.0,
            "xgb_gamma": 0.0,
            "random_state": 12345,
        },
    }

    bundle = ArtifactBundle(
        model=FakeFittedModel(),
        transformer=FakeFittedTransformer(),
        label_encoder=label_encoder,
        metadata=metadata,
        paths=get_final_artifact_paths(tmp_path),
    )

    engine = InferenceEngine(PredictionService(bundle))

    response = engine.predict_one(
        RawIncident(
            incident_timestamp=datetime(
                2015,
                5,
                13,
                23,
                53,
            ),
            pd_district="NORTHERN",
            address="OAK ST / LAGUNA ST",
            longitude=-122.425892,
            latitude=37.774599,
        ),
        top_k=2,
    )

    assert response.predicted_class == "LARCENY/THEFT"
    assert response.predicted_probability == 0.70

    assert [item.crime_category for item in response.top_predictions] == [
        "LARCENY/THEFT",
        "ROBBERY",
    ]