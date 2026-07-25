from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


MODELING_INPUT_COLUMNS = (
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
)


class IncidentAdapterError(ValueError):
    """Raised when raw incident input cannot be adapted for inference."""


@dataclass(frozen=True)
class ParsedAddress:
    """Address components matching the Version 2 SQL feature contract."""

    normalized_address: str
    street_1: str
    street_2: str | None
    block_number: str | None
    is_intersection: int


@dataclass(frozen=True)
class RawIncident:
    """External incident fields required for one model prediction."""

    incident_timestamp: datetime
    pd_district: str
    address: str
    longitude: float
    latitude: float

    def __post_init__(self) -> None:
        district = self.pd_district.strip().upper()

        if not district:
            raise IncidentAdapterError("pd_district cannot be empty.")

        normalized_address = self.address.strip()

        if not normalized_address:
            raise IncidentAdapterError("address cannot be empty.")

        try:
            longitude = float(self.longitude)
            latitude = float(self.latitude)
        except (TypeError, ValueError) as error:
            raise IncidentAdapterError(
                "Incident coordinates must be numeric."
            ) from error

        if not -180 <= longitude <= 180:
            raise IncidentAdapterError(
                "Longitude coordinate must be between -180 and 180."
            )

        if not -90 <= latitude <= 90:
            raise IncidentAdapterError(
                "Latitude coordinate must be between -90 and 90."
            )

        if not isinstance(
            self.incident_timestamp,
            datetime,
        ):
            raise IncidentAdapterError("incident_timestamp must be a datetime.")

        object.__setattr__(
            self,
            "pd_district",
            district,
        )
        object.__setattr__(
            self,
            "address",
            normalized_address,
        )
        object.__setattr__(
            self,
            "longitude",
            longitude,
        )
        object.__setattr__(
            self,
            "latitude",
            latitude,
        )


def normalize_address(address: str) -> str:
    """Normalize whitespace, case, and intersection separators."""
    if not isinstance(address, str):
        raise IncidentAdapterError("address must be a string.")

    normalized = address.strip().upper()

    if not normalized:
        raise IncidentAdapterError("address cannot be empty.")

    normalized = re.sub(
        r"\s*/\s*",
        " / ",
        normalized,
    )
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized


def parse_address(address: str) -> ParsedAddress:
    """Parse an address using Version 2 SQL-equivalent rules."""
    normalized = normalize_address(address)

    if " / " in normalized:
        street_parts = normalized.split(
            " / ",
            maxsplit=1,
        )

        street_1 = street_parts[0].strip()
        street_2 = street_parts[1].strip()

        if not street_1 or not street_2:
            raise IncidentAdapterError(
                "Intersection addresses must contain two streets."
            )

        return ParsedAddress(
            normalized_address=normalized,
            street_1=street_1,
            street_2=street_2,
            block_number=None,
            is_intersection=1,
        )

    block_match = re.match(
        r"^([0-9]+)\s+BLOCK\s+OF\s+(.+)$",
        normalized,
    )

    if block_match:
        block_number = block_match.group(1)
        street_1 = block_match.group(2).strip()

        if not street_1:
            raise IncidentAdapterError("Block address must contain a street.")

        return ParsedAddress(
            normalized_address=normalized,
            street_1=street_1,
            street_2=None,
            block_number=block_number,
            is_intersection=0,
        )

    return ParsedAddress(
        normalized_address=normalized,
        street_1=normalized,
        street_2=None,
        block_number=None,
        is_intersection=0,
    )


def datetime_to_unix_seconds(
    value: datetime,
) -> int:
    """
    Convert a datetime to Unix seconds.

    Naive datetimes are treated as UTC to ensure deterministic behavior
    across local machines and deployment environments.
    """
    resolved = value

    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    else:
        resolved = resolved.astimezone(timezone.utc)

    return int(resolved.timestamp())


def incident_to_modeling_record(
    incident: RawIncident,
) -> dict[str, object]:
    """Convert one raw incident into the Version 2 modeling-row contract."""
    timestamp = incident.incident_timestamp
    parsed_address = parse_address(incident.address)

    iso_calendar = timestamp.isocalendar()

    return {
        "incident_year": timestamp.year,
        "incident_month": timestamp.month,
        "incident_day": timestamp.day,
        "incident_hour": timestamp.hour,
        "incident_minute": timestamp.minute,
        "incident_day_of_week_num": timestamp.weekday(),
        "incident_day_of_year": timestamp.timetuple().tm_yday,
        "incident_week_of_year": iso_calendar.week,
        "datetime_numeric": datetime_to_unix_seconds(timestamp),
        "is_weekend": int(timestamp.weekday() >= 5),
        "pd_district": incident.pd_district,
        "longitude": incident.longitude,
        "latitude": incident.latitude,
        "street_1": parsed_address.street_1,
        "street_2": parsed_address.street_2,
        "block_number": parsed_address.block_number,
        "is_intersection": (parsed_address.is_intersection),
    }


def build_modeling_frame(
    incidents: Iterable[RawIncident],
) -> pd.DataFrame:
    """Convert raw incident records into prediction-service input."""
    incident_list = list(incidents)

    if not incident_list:
        raise IncidentAdapterError("At least one incident is required.")

    records = [incident_to_modeling_record(incident) for incident in incident_list]

    frame = pd.DataFrame.from_records(
        records,
        columns=MODELING_INPUT_COLUMNS,
    )

    frame["is_weekend"] = frame["is_weekend"].astype("int8")

    frame["is_intersection"] = frame["is_intersection"].astype("int8")

    return frame