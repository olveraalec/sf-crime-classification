from pathlib import Path

import duckdb
import pandas as pd

from src.config import get_project_root, load_config
from src.logger import get_logger


logger = get_logger(__name__)


def get_database_path() -> Path:
    """Return the configured DuckDB database path."""
    config = load_config()
    project_root = get_project_root()

    return project_root / config["paths"]["database"]


def load_modeling_data() -> pd.DataFrame:
    """Load the SQL modeling view into a pandas DataFrame."""
    database_path = get_database_path()

    if not database_path.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {database_path}\n"
            "Run `python -m src.build_database` first."
        )

    logger.info("Loading modeling data from: %s", database_path)

    query = """
        SELECT *
        FROM sf_crime_modeling
        ORDER BY incident_timestamp
    """

    with duckdb.connect(str(database_path), read_only=True) as connection:
        data = connection.execute(query).fetchdf()

    logger.info(
        "Loaded modeling data with %s rows and %s columns",
        len(data),
        len(data.columns),
    )

    return data


def separate_target(
    data: pd.DataFrame,
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and target without modifying the input DataFrame."""
    if target_column not in data.columns:
        raise KeyError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    X = data.drop(columns=[target_column]).copy()
    y = data[target_column].copy()

    return X, y