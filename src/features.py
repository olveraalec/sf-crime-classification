from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from src.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureGroups:
    """Named groups of SF Crime predictor columns."""

    calendar: tuple[str, ...] = (
        "incident_year",
        "incident_month",
        "incident_day",
        "incident_hour",
        "incident_minute",
        "incident_day_of_week_num",
        "incident_day_of_year",
        "incident_week_of_year",
        "is_weekend",
    )

    spatial: tuple[str, ...] = (
        "pd_district",
        "longitude",
        "latitude",
    )

    address: tuple[str, ...] = (
        "street_1",
        "street_2",
        "block_number",
        "is_intersection",
    )

    time_trend: tuple[str, ...] = ("datetime_numeric",)

    identifiers: tuple[str, ...] = (
        "incident_timestamp",
        "address",
    )

    engineered: tuple[str, ...] = field(default_factory=tuple)

    def all_model_features(
        self,
        include_time_trend: bool = True,
        include_identifiers: bool = False,
        include_engineered: bool = True,
    ) -> list[str]:
        """Return a combined feature list based on enabled groups."""
        columns = [
            *self.calendar,
            *self.spatial,
            *self.address,
        ]

        if include_time_trend:
            columns.extend(self.time_trend)

        if include_identifiers:
            columns.extend(self.identifiers)

        if include_engineered:
            columns.extend(self.engineered)

        return list(dict.fromkeys(columns))


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """Raise an informative error when required columns are absent."""
    required = list(required_columns)
    missing = sorted(set(required) - set(data.columns))

    if missing:
        raise KeyError(
            "Required feature columns are missing: "
            f"{missing}. Available columns: {list(data.columns)}"
        )


def add_cyclical_features(
    data: pd.DataFrame,
    cyclical_periods: dict[str, int] | None = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Add sine/cosine representations for periodic variables.

    Parameters
    ----------
    data:
        Input feature DataFrame.
    cyclical_periods:
        Mapping from column name to cycle length.
    drop_original:
        Whether to remove original periodic columns after encoding.
    """
    if cyclical_periods is None:
        cyclical_periods = {
            "incident_hour": 24,
            "incident_day_of_week_num": 7,
            "incident_month": 12,
            "incident_day_of_year": 365,
        }

    validate_required_columns(data, cyclical_periods.keys())

    output = data.copy()

    for column, period in cyclical_periods.items():
        if period <= 0:
            raise ValueError(f"Cycle period for '{column}' must be positive.")

        values = pd.to_numeric(output[column], errors="coerce")

        output[f"{column}_sin"] = np.sin(2 * np.pi * values / period)

        output[f"{column}_cos"] = np.cos(2 * np.pi * values / period)

    if drop_original:
        output = output.drop(columns=list(cyclical_periods))

    logger.info(
        "Added cyclical features for columns: %s",
        list(cyclical_periods),
    )

    return output


def add_interaction_features(
    data: pd.DataFrame,
    include_district_hour: bool = True,
    include_district_day: bool = True,
    include_district_intersection: bool = True,
    include_coordinate_interaction: bool = False,
) -> pd.DataFrame:
    """
    Add deterministic interaction features.

    These transformations do not learn parameters and are safe to apply
    before model-specific fitted encoders.
    """
    output = data.copy()

    if include_district_hour:
        validate_required_columns(
            output,
            ["pd_district", "incident_hour"],
        )

        output["district_hour"] = (
            output["pd_district"].astype("string")
            + "__"
            + output["incident_hour"].astype("Int64").astype("string")
        )

    if include_district_day:
        validate_required_columns(
            output,
            ["pd_district", "incident_day_of_week_num"],
        )

        output["district_day"] = (
            output["pd_district"].astype("string")
            + "__"
            + output["incident_day_of_week_num"].astype("Int64").astype("string")
        )

    if include_district_intersection:
        validate_required_columns(
            output,
            ["pd_district", "is_intersection"],
        )

        output["district_intersection"] = (
            output["pd_district"].astype("string")
            + "__"
            + output["is_intersection"].astype("Int64").astype("string")
        )

    if include_coordinate_interaction:
        validate_required_columns(
            output,
            ["longitude", "latitude"],
        )

        output["longitude_latitude_interaction"] = (
            output["longitude"] * output["latitude"]
        )

    logger.info("Added configured deterministic interaction features.")

    return output


def add_address_features(
    data: pd.DataFrame,
    add_street_pair: bool = True,
    add_block_presence: bool = True,
) -> pd.DataFrame:
    """Add deterministic address-derived features."""
    output = data.copy()

    if add_street_pair:
        validate_required_columns(
            output,
            ["street_1", "street_2"],
        )

        street_1 = output["street_1"].fillna("UNKNOWN")
        street_2 = output["street_2"].fillna("NO_SECOND_STREET")

        output["street_pair"] = (
            street_1.astype("string") + "__" + street_2.astype("string")
        )

    if add_block_presence:
        validate_required_columns(output, ["block_number"])

        output["has_block_number"] = output["block_number"].notna().astype("int8")

    logger.info("Added configured address features.")

    return output


def build_base_feature_frame(
    data: pd.DataFrame,
    groups: FeatureGroups | None = None,
    include_time_trend: bool = True,
    include_identifiers: bool = False,
) -> pd.DataFrame:
    """
    Select a configurable base set of features from modeling data.
    """
    if groups is None:
        groups = FeatureGroups()

    selected_columns = groups.all_model_features(
        include_time_trend=include_time_trend,
        include_identifiers=include_identifiers,
        include_engineered=False,
    )

    validate_required_columns(data, selected_columns)

    output = data[selected_columns].copy()

    logger.info(
        "Built base feature frame with %s rows and %s columns",
        len(output),
        len(output.columns),
    )

    return output


def build_feature_frame(
    data: pd.DataFrame,
    groups: FeatureGroups | None = None,
    include_time_trend: bool = True,
    include_identifiers: bool = False,
    add_cyclical: bool = True,
    drop_original_cyclical: bool = False,
    add_interactions: bool = True,
    add_address_engineering: bool = True,
) -> pd.DataFrame:
    """
    Build a configurable deterministic feature frame.

    Learned operations such as encoding, scaling, KMeans, PCA, feature
    selection, and entropy filtering belong in fitted transformers.
    """
    output = build_base_feature_frame(
        data=data,
        groups=groups,
        include_time_trend=include_time_trend,
        include_identifiers=include_identifiers,
    )

    if add_cyclical:
        output = add_cyclical_features(
            output,
            drop_original=drop_original_cyclical,
        )

    if add_interactions:
        output = add_interaction_features(output)

    if add_address_engineering:
        output = add_address_features(output)

    logger.info(
        "Built deterministic feature frame with %s rows and %s columns",
        len(output),
        len(output.columns),
    )

    return output