from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.incident_adapter import (
    IncidentAdapterError,
    RawIncident,
    build_modeling_frame,
    parse_address,
)


def test_parse_intersection_address() -> None:
    result = parse_address("OAK ST / LAGUNA ST")

    assert result.normalized_address == ("OAK ST / LAGUNA ST")
    assert result.street_1 == "OAK ST"
    assert result.street_2 == "LAGUNA ST"
    assert result.block_number is None
    assert result.is_intersection == 1


def test_parse_intersection_normalizes_separator_spacing() -> None:
    result = parse_address("OAK ST/LAGUNA ST")

    assert result.normalized_address == ("OAK ST / LAGUNA ST")
    assert result.street_1 == "OAK ST"
    assert result.street_2 == "LAGUNA ST"


def test_parse_block_address() -> None:
    result = parse_address("100 BLOCK OF MISSION ST")

    assert result.normalized_address == ("100 BLOCK OF MISSION ST")
    assert result.street_1 == "MISSION ST"
    assert result.street_2 is None
    assert result.block_number == "100"
    assert result.is_intersection == 0


def test_parse_non_block_single_street() -> None:
    result = parse_address("MARKET ST")

    assert result.street_1 == "MARKET ST"
    assert result.street_2 is None
    assert result.block_number is None
    assert result.is_intersection == 0


def test_parse_address_trims_and_uppercases_values() -> None:
    result = parse_address("  oak st / laguna st  ")

    assert result.normalized_address == ("OAK ST / LAGUNA ST")
    assert result.street_1 == "OAK ST"
    assert result.street_2 == "LAGUNA ST"


def test_parse_address_rejects_empty_value() -> None:
    with pytest.raises(
        IncidentAdapterError,
        match="address cannot be empty",
    ):
        parse_address("   ")


def test_raw_incident_normalizes_district() -> None:
    incident = RawIncident(
        incident_timestamp=datetime(
            2015,
            5,
            13,
            23,
            53,
        ),
        pd_district=" northern ",
        address="OAK ST / LAGUNA ST",
        longitude=-122.425892,
        latitude=37.774599,
    )

    assert incident.pd_district == "NORTHERN"


def test_raw_incident_rejects_empty_district() -> None:
    with pytest.raises(
        IncidentAdapterError,
        match="pd_district cannot be empty",
    ):
        RawIncident(
            incident_timestamp=datetime(
                2015,
                5,
                13,
                23,
                53,
            ),
            pd_district=" ",
            address="OAK ST / LAGUNA ST",
            longitude=-122.425892,
            latitude=37.774599,
        )


@pytest.mark.parametrize(
    ("longitude", "latitude"),
    [
        (-181.0, 37.77),
        (181.0, 37.77),
        (-122.42, -91.0),
        (-122.42, 91.0),
    ],
)
def test_raw_incident_rejects_invalid_coordinate_ranges(
    longitude: float,
    latitude: float,
) -> None:
    with pytest.raises(
        IncidentAdapterError,
        match="coordinate",
    ):
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
            longitude=longitude,
            latitude=latitude,
        )


def test_build_modeling_frame_recreates_calendar_contract() -> None:
    incident = RawIncident(
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
    )

    frame = build_modeling_frame([incident])

    assert frame.shape == (1, 17)

    row = frame.iloc[0]

    assert row["incident_year"] == 2015
    assert row["incident_month"] == 5
    assert row["incident_day"] == 13
    assert row["incident_hour"] == 23
    assert row["incident_minute"] == 53

    # Python Monday=0. May 13, 2015 was Wednesday.
    assert row["incident_day_of_week_num"] == 2

    assert row["incident_day_of_year"] == 133
    assert row["incident_week_of_year"] == 20
    assert row["is_weekend"] == 0


def test_build_modeling_frame_recreates_address_contract() -> None:
    incidents = [
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
        RawIncident(
            incident_timestamp=datetime(
                2015,
                5,
                14,
                0,
                30,
            ),
            pd_district="MISSION",
            address="100 BLOCK OF MISSION ST",
            longitude=-122.419416,
            latitude=37.774929,
        ),
    ]

    frame = build_modeling_frame(incidents)

    assert frame.loc[0, "street_1"] == "OAK ST"
    assert frame.loc[0, "street_2"] == "LAGUNA ST"
    assert pd.isna(frame.loc[0, "block_number"])
    assert frame.loc[0, "is_intersection"] == 1

    assert frame.loc[1, "street_1"] == "MISSION ST"
    assert pd.isna(frame.loc[1, "street_2"])
    assert frame.loc[1, "block_number"] == "100"
    assert frame.loc[1, "is_intersection"] == 0


def test_build_modeling_frame_uses_unix_seconds() -> None:
    incident = RawIncident(
        incident_timestamp=datetime(
            1970,
            1,
            1,
            0,
            1,
            tzinfo=timezone.utc,
        ),
        pd_district="NORTHERN",
        address="OAK ST",
        longitude=-122.425892,
        latitude=37.774599,
    )

    frame = build_modeling_frame([incident])

    assert frame.loc[0, "datetime_numeric"] == 60


def test_build_modeling_frame_rejects_empty_collection() -> None:
    with pytest.raises(
        IncidentAdapterError,
        match="At least one incident",
    ):
        build_modeling_frame([])


def test_build_modeling_frame_preserves_required_column_order() -> None:
    incident = RawIncident(
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
    )

    frame = build_modeling_frame([incident])

    assert frame.columns.tolist() == [
        "incident_year",
        "incident_month",
        "incident_day",
        "incident_hour",
        "incident_minute",
        "incident_day_of_week_num",
        "incident_day_of_year",
        "incident_week_of_year",
        "datetime_numeric",
        "is_weekend",
        "pd_district",
        "longitude",
        "latitude",
        "street_1",
        "street_2",
        "block_number",
        "is_intersection",
    ]