from pathlib import Path

import duckdb

from src.config import get_project_root, load_config
from src.logger import get_logger


logger = get_logger(__name__)


def run_sql_file(connection: duckdb.DuckDBPyConnection, sql_path: Path) -> None:
    """Run a SQL file against the DuckDB connection."""
    logger.info("Running SQL file: %s", sql_path)

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql = sql_path.read_text()
    connection.execute(sql)


def build_database() -> None:
    """Build the DuckDB database from raw SF Crime data."""
    config = load_config()
    project_root = get_project_root()

    raw_file = project_root / config["data"]["raw_file"]
    database_path = project_root / config["paths"]["database"]

    sql_raw = project_root / "sql" / "01_create_raw_table.sql"
    sql_clean = project_root / "sql" / "02_create_clean_view.sql"
    sql_features = project_root / "sql" / "03_create_feature_view.sql"
    sql_modeling = project_root / "sql" / "04_create_modeling_view.sql"

    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_file}\n"
            "Place the SF Crime train.csv file in data/raw/train.csv."
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building DuckDB database at: %s", database_path)

    with duckdb.connect(str(database_path)) as connection:
        run_sql_file(connection, sql_raw)
        run_sql_file(connection, sql_clean)
        run_sql_file(connection, sql_features)
        run_sql_file(connection, sql_modeling)

    row_count = connection.execute("SELECT COUNT(*) FROM sf_crime_modeling").fetchone()[
        0
    ]

    logger.info(
        "Database build complete. Modeling row count: %s",
        row_count,
    )


if __name__ == "__main__":
    build_database()