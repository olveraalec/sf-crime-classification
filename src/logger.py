import logging
from pathlib import Path

from src.config import get_project_root, load_config


def get_logger(name: str) -> logging.Logger:
    """Create a project logger that writes to both console and a log file."""
    config = load_config()
    project_root = get_project_root()

    logs_dir = project_root / config["paths"]["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "sf_crime_pipeline.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger