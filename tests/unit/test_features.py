from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureGroups,
    add_address_features,
    add_cyclical_features,
    add_interaction_features,
    build_base_feature_frame,
    build_feature_frame,
    validate_required_columns,
)


@pytest.fixture
def modeling_frame() -> pd.DataFrame:
    """Small synthetic frame matching the DuckDB modeling-view contract."""
    return pd.DataFrame(
        {
            "incident_timestamp": pd.to_datetime(
                [
                    "2015-05-13 23:53:00",
                    "2015-05-14 00:30:00",
                    "2015-05-15 12:15:00",
                ]
            ),
            "incident_year": [2015, 2015, 2015],
            "incident_month": [5, 5, 5],
            "incident_day": [13, 14, 15],
            "incident_hour": [23, 0, 12],
            "incident_minute": [53, 30, 15],
            "incident_day_of_week_num": [2, 3, 4],
            "incident_day_of_year": [133, 134, 135],
            "incident_week_of_year": [20, 20, 20],
            "datetime_numeric": [1431561180, 1431563400, 1431692100],
            "is_weekend": [0, 0, 0],
            "pd_district": ["NORTHERN", "MISSION", "SOUTHERN"],
            "longitude": [-122.425892, -122.419416, -122.403405],
            "latitude": [37.774599, 37.774929, 37.775421],
            "address": [
                "OAK ST / LAGUNA ST",
                "100 BLOCK OF MISSION ST",
                "MARKET ST / 5TH ST",
            ],
            "street_1": ["OAK ST", "MISSION ST", "MARKET ST"],
            "street_2": ["LAGUNA ST", pd.NA, "5TH ST"],
            "block_number": [pd.NA, "100", pd.NA],
            "is_intersection": [1, 0, 1],
        }
    )


def test_feature_groups_return_expected_base_columns() -> None:
    groups = FeatureGroups()

    columns = groups.all_model_features(
        include_time_trend=True,
        include_identifiers=False,
        include_engineered=False,
    )

    assert len(columns) == 17
    assert "datetime_numeric" in columns
    assert "incident_timestamp" not in columns
    assert "address" not in columns


def test_validate_required_columns_raises_informative_error() -> None:
    frame = pd.DataFrame({"incident_hour": [12]})

    with pytest.raises(KeyError, match="pd_district"):
        validate_required_columns(
            frame,
            ["incident_hour", "pd_district"],
        )


def test_build_base_feature_frame_preserves_version_2_contract(
    modeling_frame: pd.DataFrame,
) -> None:
    result = build_base_feature_frame(modeling_frame)

    assert result.shape == (3, 17)
    assert "incident_timestamp" not in result.columns
    assert "address" not in result.columns
    assert list(result.index) == list(modeling_frame.index)


def test_build_base_feature_frame_does_not_mutate_input(
    modeling_frame: pd.DataFrame,
) -> None:
    original = modeling_frame.copy(deep=True)

    build_base_feature_frame(modeling_frame)

    pd.testing.assert_frame_equal(modeling_frame, original)


def test_cyclical_features_match_version_2_formulas(
    modeling_frame: pd.DataFrame,
) -> None:
    base = build_base_feature_frame(modeling_frame)

    result = add_cyclical_features(base)

    expected_hour_sin = np.sin(2 * np.pi * modeling_frame["incident_hour"] / 24)
    expected_hour_cos = np.cos(2 * np.pi * modeling_frame["incident_hour"] / 24)

    np.testing.assert_allclose(
        result["incident_hour_sin"],
        expected_hour_sin,
    )
    np.testing.assert_allclose(
        result["incident_hour_cos"],
        expected_hour_cos,
    )

    assert result.shape[1] == 25


def test_cyclical_features_can_drop_original_columns(
    modeling_frame: pd.DataFrame,
) -> None:
    base = build_base_feature_frame(modeling_frame)

    result = add_cyclical_features(
        base,
        drop_original=True,
    )

    assert "incident_hour" not in result.columns
    assert "incident_day_of_week_num" not in result.columns
    assert "incident_month" not in result.columns
    assert "incident_day_of_year" not in result.columns

    assert "incident_hour_sin" in result.columns
    assert "incident_hour_cos" in result.columns


def test_interaction_feature_format_is_preserved(
    modeling_frame: pd.DataFrame,
) -> None:
    base = build_base_feature_frame(modeling_frame)

    result = add_interaction_features(base)

    assert result.loc[0, "district_hour"] == "NORTHERN__23"
    assert result.loc[0, "district_day"] == "NORTHERN__2"
    assert result.loc[0, "district_intersection"] == "NORTHERN__1"

    assert result.loc[1, "district_hour"] == "MISSION__0"
    assert result.loc[1, "district_intersection"] == "MISSION__0"


def test_address_features_preserve_missing_value_rules(
    modeling_frame: pd.DataFrame,
) -> None:
    base = build_base_feature_frame(modeling_frame)

    result = add_address_features(base)

    assert result.loc[0, "street_pair"] == "OAK ST__LAGUNA ST"
    assert result.loc[1, "street_pair"] == ("MISSION ST__NO_SECOND_STREET")

    assert result.loc[0, "has_block_number"] == 0
    assert result.loc[1, "has_block_number"] == 1
    assert result["has_block_number"].dtype == "int8"


def test_final_deterministic_feature_frame_has_30_columns(
    modeling_frame: pd.DataFrame,
) -> None:
    result = build_feature_frame(modeling_frame)

    assert result.shape == (3, 30)

    expected_engineered_columns = {
        "incident_hour_sin",
        "incident_hour_cos",
        "incident_day_of_week_num_sin",
        "incident_day_of_week_num_cos",
        "incident_month_sin",
        "incident_month_cos",
        "incident_day_of_year_sin",
        "incident_day_of_year_cos",
        "district_hour",
        "district_day",
        "district_intersection",
        "street_pair",
        "has_block_number",
    }

    assert expected_engineered_columns.issubset(result.columns)


def test_build_feature_frame_does_not_mutate_input(
    modeling_frame: pd.DataFrame,
) -> None:
    original = modeling_frame.copy(deep=True)

    build_feature_frame(modeling_frame)

    pd.testing.assert_frame_equal(modeling_frame, original)