from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_feature_frame
from src.transformers import (
    FrequencyEncoder,
    GeoSpatialTransformer,
    build_categorical_pipeline,
    build_tree_transformer,
)


@pytest.fixture
def geographic_frame() -> pd.DataFrame:
    """Small spatial frame with enough distinct points for clustering."""
    return pd.DataFrame(
        {
            "longitude": [
                -122.425892,
                -122.419416,
                -122.403405,
                -122.410000,
                -122.430000,
                -122.400000,
            ],
            "latitude": [
                37.774599,
                37.774929,
                37.775421,
                37.780000,
                37.760000,
                37.790000,
            ],
            "pd_district": [
                "NORTHERN",
                "MISSION",
                "SOUTHERN",
                "CENTRAL",
                "RICHMOND",
                "BAYVIEW",
            ],
        }
    )


@pytest.fixture
def modeling_frame() -> pd.DataFrame:
    """Synthetic frame matching the DuckDB modeling-view contract."""
    row_count = 50

    timestamps = pd.date_range(
        "2015-01-01 00:00:00",
        periods=row_count,
        freq="6h",
    )

    longitude = np.linspace(
        -122.50,
        -122.36,
        row_count,
    )

    latitude = np.linspace(
        37.71,
        37.81,
        row_count,
    )

    districts = np.resize(
        np.array(
            [
                "NORTHERN",
                "MISSION",
                "SOUTHERN",
                "CENTRAL",
                "BAYVIEW",
            ]
        ),
        row_count,
    )

    street_1 = np.resize(
        np.array(
            [
                "OAK ST",
                "MISSION ST",
                "MARKET ST",
                "GEARY BL",
                "3RD ST",
            ]
        ),
        row_count,
    )

    street_2 = np.resize(
        np.array(
            [
                "LAGUNA ST",
                pd.NA,
                "5TH ST",
                pd.NA,
                "EVANS AV",
            ],
            dtype=object,
        ),
        row_count,
    )

    block_number = np.resize(
        np.array(
            [
                pd.NA,
                "100",
                pd.NA,
                "800",
                pd.NA,
            ],
            dtype=object,
        ),
        row_count,
    )

    is_intersection = pd.Series(street_2).notna().astype("int8")

    return pd.DataFrame(
        {
            "incident_timestamp": timestamps,
            "incident_year": timestamps.year,
            "incident_month": timestamps.month,
            "incident_day": timestamps.day,
            "incident_hour": timestamps.hour,
            "incident_minute": timestamps.minute,
            "incident_day_of_week_num": timestamps.dayofweek,
            "incident_day_of_year": timestamps.dayofyear,
            "incident_week_of_year": (timestamps.isocalendar().week.astype(int)).to_numpy(),
            "datetime_numeric": (timestamps.astype("int64") // 10**9),
            "is_weekend": (timestamps.dayofweek >= 5).astype("int8"),
            "pd_district": districts,
            "longitude": longitude,
            "latitude": latitude,
            "address": [
                (
                    f"{first} / {second}"
                    if pd.notna(second)
                    else f"{block} BLOCK OF {first}"
                )
                for first, second, block in zip(
                    street_1,
                    street_2,
                    block_number,
                    strict=True,
                )
            ],
            "street_1": street_1,
            "street_2": street_2,
            "block_number": block_number,
            "is_intersection": is_intersection,
        }
    )


def test_frequency_encoder_uses_training_relative_frequencies() -> None:
    training = pd.DataFrame(
        {
            "district": [
                "NORTHERN",
                "NORTHERN",
                "MISSION",
                "SOUTHERN",
            ]
        }
    )

    encoder = FrequencyEncoder()
    result = encoder.fit_transform(training)

    np.testing.assert_allclose(
        result.ravel(),
        [
            0.5,
            0.5,
            0.25,
            0.25,
        ],
    )


def test_frequency_encoder_assigns_zero_to_unseen_categories() -> None:
    training = pd.DataFrame(
        {
            "district": [
                "NORTHERN",
                "NORTHERN",
                "MISSION",
            ]
        }
    )

    validation = pd.DataFrame(
        {
            "district": [
                "NORTHERN",
                "UNKNOWN_DISTRICT",
            ]
        }
    )

    encoder = FrequencyEncoder().fit(training)
    result = encoder.transform(validation)

    np.testing.assert_allclose(
        result.ravel(),
        [
            2 / 3,
            0.0,
        ],
    )


def test_frequency_encoder_rejects_changed_column_count() -> None:
    training = pd.DataFrame(
        {
            "district": ["NORTHERN", "MISSION"],
        }
    )

    validation = pd.DataFrame(
        {
            "district": ["NORTHERN"],
            "street": ["OAK ST"],
        }
    )

    encoder = FrequencyEncoder().fit(training)

    with pytest.raises(
        ValueError,
        match="different number of columns",
    ):
        encoder.transform(validation)


def test_geospatial_raw_mode_preserves_original_columns(
    geographic_frame: pd.DataFrame,
) -> None:
    transformer = GeoSpatialTransformer(
        mode="raw",
        n_clusters=2,
    )

    result = transformer.fit_transform(geographic_frame)

    pd.testing.assert_frame_equal(
        result,
        geographic_frame,
    )


def test_geospatial_cluster_mode_replaces_raw_coordinates(
    geographic_frame: pd.DataFrame,
) -> None:
    transformer = GeoSpatialTransformer(
        mode="cluster",
        n_clusters=2,
        random_state=12345,
    )

    result = transformer.fit_transform(geographic_frame)

    assert "longitude" not in result.columns
    assert "latitude" not in result.columns
    assert "geo_cluster" in result.columns
    assert result["geo_cluster"].dtype.name == "string"
    assert set(result["geo_cluster"].unique()).issubset({"0", "1"})


def test_geospatial_raw_distances_adds_expected_columns(
    geographic_frame: pd.DataFrame,
) -> None:
    transformer = GeoSpatialTransformer(
        mode="raw_distances",
        n_clusters=3,
        random_state=12345,
    )

    result = transformer.fit_transform(geographic_frame)

    assert "longitude" in result.columns
    assert "latitude" in result.columns

    expected_distance_columns = {
        "geo_distance_0",
        "geo_distance_1",
        "geo_distance_2",
    }

    assert expected_distance_columns.issubset(result.columns)
    assert result.shape[1] == geographic_frame.shape[1] + 3


def test_geospatial_distances_mode_drops_raw_coordinates(
    geographic_frame: pd.DataFrame,
) -> None:
    transformer = GeoSpatialTransformer(
        mode="distances",
        n_clusters=3,
        random_state=12345,
    )

    result = transformer.fit_transform(geographic_frame)

    assert "longitude" not in result.columns
    assert "latitude" not in result.columns

    assert {
        "geo_distance_0",
        "geo_distance_1",
        "geo_distance_2",
    }.issubset(result.columns)


def test_geospatial_transformer_imputes_missing_coordinates_from_training(
    geographic_frame: pd.DataFrame,
) -> None:
    transformer = GeoSpatialTransformer(
        mode="raw_distances",
        n_clusters=2,
        random_state=12345,
    ).fit(geographic_frame)

    validation = geographic_frame.iloc[[0]].copy()
    validation.loc[:, "longitude"] = np.nan
    validation.loc[:, "latitude"] = np.nan

    result = transformer.transform(validation)

    assert (
        not result[
            [
                "geo_distance_0",
                "geo_distance_1",
            ]
        ]
        .isna()
        .any()
        .any()
    )


def test_geospatial_transformer_is_reproducible_for_fixed_seed(
    geographic_frame: pd.DataFrame,
) -> None:
    first = GeoSpatialTransformer(
        mode="raw_distances",
        n_clusters=3,
        random_state=12345,
    )

    second = GeoSpatialTransformer(
        mode="raw_distances",
        n_clusters=3,
        random_state=12345,
    )

    first_result = first.fit_transform(geographic_frame)
    second_result = second.fit_transform(geographic_frame)

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )


def test_geospatial_transformer_requires_coordinate_columns() -> None:
    frame = pd.DataFrame(
        {
            "pd_district": ["NORTHERN"],
        }
    )

    transformer = GeoSpatialTransformer(
        mode="raw",
        n_clusters=2,
    )

    with pytest.raises(
        KeyError,
        match="Missing coordinate columns",
    ):
        transformer.fit(frame)


def test_ordinal_pipeline_encodes_unknown_category_as_negative_one() -> None:
    training = pd.DataFrame(
        {
            "district": [
                "NORTHERN",
                "MISSION",
            ]
        }
    )

    validation = pd.DataFrame(
        {
            "district": [
                "NORTHERN",
                "UNKNOWN_DISTRICT",
            ]
        }
    )

    pipeline = build_categorical_pipeline(
        encoding="ordinal",
        sparse_output=False,
    )

    pipeline.fit(training)
    result = pipeline.transform(validation)

    assert result.shape == (2, 1)
    assert result[1, 0] == -1


def test_final_tree_transformer_preserves_70_column_contract(
    modeling_frame: pd.DataFrame,
) -> None:
    feature_frame = build_feature_frame(modeling_frame)

    assert feature_frame.shape[1] == 30

    transformer = build_tree_transformer(
        categorical_encoding="ordinal",
        numeric_strategy="passthrough",
        geo_mode="raw_distances",
        n_geo_clusters=40,
    )

    transformed = transformer.fit_transform(feature_frame)

    assert transformed.shape == (
        len(feature_frame),
        70,
    )


def test_final_tree_transformer_returns_dense_numeric_output(
    modeling_frame: pd.DataFrame,
) -> None:
    feature_frame = build_feature_frame(modeling_frame)

    transformer = build_tree_transformer(
        categorical_encoding="ordinal",
        numeric_strategy="passthrough",
        geo_mode="raw_distances",
        n_geo_clusters=40,
    )

    transformed = transformer.fit_transform(feature_frame)

    assert isinstance(transformed, np.ndarray)
    assert np.issubdtype(
        transformed.dtype,
        np.number,
    )
    assert np.isfinite(transformed).all()