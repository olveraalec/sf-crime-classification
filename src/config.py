from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(
    config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Load the project YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Project configuration must contain a YAML mapping.")

    return config


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return PROJECT_ROOT


@dataclass(frozen=True)
class AppSettings:
    """Validated operational settings for the API service."""

    api_title: str
    api_description: str
    api_version: str
    service_name: str

    model_name: str

    log_level: str
    log_filename: str

    server_host: str
    server_port: int


def _read_environment(
    name: str,
    default: str,
) -> str:
    """Return an environment override or its configured default."""
    return os.getenv(
        name,
        default,
    )


def _read_port(
    name: str,
    default: int,
) -> int:
    """Return and validate an integer port environment override."""
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        port = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error

    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535.")

    return port


def get_app_settings(
    config_path: Path = CONFIG_PATH,
) -> AppSettings:
    """Load operational settings with environment overrides."""
    config = load_config(config_path)

    api = config.get("api", {})
    model = config.get("model", {})
    logging_config = config.get("logging", {})
    server = config.get("server", {})

    if not all(
        isinstance(section, dict)
        for section in (
            api,
            model,
            logging_config,
            server,
        )
    ):
        raise ValueError(
            "API, model, logging, and server configuration "
            "sections must be YAML mappings."
        )

    configured_port = server.get(
        "port",
        8000,
    )

    if not isinstance(configured_port, int):
        raise ValueError("Configured server port must be an integer.")

    settings = AppSettings(
        api_title=_read_environment(
            "SF_API_TITLE",
            str(
                api.get(
                    "title",
                    "San Francisco Crime Classification API",
                )
            ),
        ),
        api_description=_read_environment(
            "SF_API_DESCRIPTION",
            str(
                api.get(
                    "description",
                    "Production crime-classification inference service.",
                )
            ),
        ),
        api_version=_read_environment(
            "SF_API_VERSION",
            str(
                api.get(
                    "version",
                    "3.0.0",
                )
            ),
        ),
        service_name=_read_environment(
            "SF_API_SERVICE_NAME",
            str(
                api.get(
                    "service_name",
                    "sf-crime-classification-api",
                )
            ),
        ),
        model_name=_read_environment(
            "SF_API_MODEL_NAME",
            str(
                model.get(
                    "name",
                    "xgboost_finalist",
                )
            ),
        ),
        log_level=_read_environment(
            "SF_API_LOG_LEVEL",
            str(
                logging_config.get(
                    "level",
                    "INFO",
                )
            ),
        ).upper(),
        log_filename=_read_environment(
            "SF_API_LOG_FILENAME",
            str(
                logging_config.get(
                    "filename",
                    "sf_crime_pipeline.log",
                )
            ),
        ),
        server_host=_read_environment(
            "SF_API_HOST",
            str(
                server.get(
                    "host",
                    "0.0.0.0",
                )
            ),
        ),
        server_port=_read_port(
            "SF_API_PORT",
            configured_port,
        ),
    )

    if not settings.api_title.strip():
        raise ValueError("API title cannot be empty.")

    if not settings.service_name.strip():
        raise ValueError("Service name cannot be empty.")

    if not settings.model_name.strip():
        raise ValueError("Model name cannot be empty.")

    if settings.log_level not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")

    return settings