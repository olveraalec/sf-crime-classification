from __future__ import annotations

from pathlib import Path

import pytest

from src.config import get_app_settings


CONFIG_TEXT = """
api:
  title: Test API
  description: Test description
  version: "1.2.3"
  service_name: test-service

model:
  name: configured-model

logging:
  level: INFO
  filename: configured.log

server:
  host: 127.0.0.1
  port: 8000
"""


def write_config(
    tmp_path: Path,
    content: str = CONFIG_TEXT,
) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        content,
        encoding="utf-8",
    )

    return config_path


def test_app_settings_load_from_yaml(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path)

    settings = get_app_settings(config_path)

    assert settings.api_title == "Test API"
    assert settings.api_version == "1.2.3"
    assert settings.service_name == "test-service"
    assert settings.model_name == "configured-model"
    assert settings.log_level == "INFO"
    assert settings.log_filename == "configured.log"
    assert settings.server_host == "127.0.0.1"
    assert settings.server_port == 8000


def test_environment_overrides_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path)

    monkeypatch.setenv(
        "SF_API_MODEL_NAME",
        "environment-model",
    )
    monkeypatch.setenv(
        "SF_API_LOG_LEVEL",
        "debug",
    )
    monkeypatch.setenv(
        "SF_API_PORT",
        "9000",
    )

    settings = get_app_settings(config_path)

    assert settings.model_name == "environment-model"
    assert settings.log_level == "DEBUG"
    assert settings.server_port == 9000


@pytest.mark.parametrize(
    "port",
    [
        "not-an-integer",
        "0",
        "65536",
    ],
)
def test_invalid_environment_port_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    port: str,
) -> None:
    config_path = write_config(tmp_path)

    monkeypatch.setenv(
        "SF_API_PORT",
        port,
    )

    with pytest.raises(ValueError):
        get_app_settings(config_path)


def test_invalid_log_level_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_config(tmp_path)

    monkeypatch.setenv(
        "SF_API_LOG_LEVEL",
        "VERBOSE",
    )

    with pytest.raises(
        ValueError,
        match="Log level",
    ):
        get_app_settings(config_path)