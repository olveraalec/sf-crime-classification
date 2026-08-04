from __future__ import annotations

import logging

from src.config import (
    get_app_settings,
    get_project_root,
    load_config,
)


def get_logger(
    name: str,
) -> logging.Logger:
    """Create a project logger for console and file output."""
    config = load_config()
    settings = get_app_settings()
    project_root = get_project_root()

    logs_dir = project_root / config["paths"]["logs"]
    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = logs_dir / settings.log_filename

    logger = logging.getLogger(name)

    try:
        log_level = getattr(
            logging,
            settings.log_level,
        )
    except AttributeError as error:
        raise ValueError(f"Unsupported log level: {settings.log_level}") from error

    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger