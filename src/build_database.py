from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from src.config import get_project_root, load_config
from src.logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class DatabaseBuildResult:
    """Summary returned after successfully building the database."""

    database_path: Path
    modeling_row_count: int
    executed_sql_files: tuple[Path, ...]


def run_sql_file(
    connection: duckdb.DuckDBPyConnection,
    sql_path: Path,
) -> None:
    """Execute one SQL file using an active DuckDB connection."""
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    logger.info("Running SQL file: %s", sql_path)

    sql = sql_path.read_text(encoding="utf-8")
    connection.execute(sql)


def get_database_sql_files(
    project_root: Path,
) -> tuple[Path, ...]:
    """Return database-build SQL files in required execution order."""
    sql_directory = project_root / "sql"

    return (
        sql_directory / "01_create_raw_table.sql",
        sql_directory / "02_create_clean_view.sql",
        sql_directory / "03_create_feature_view.sql",
        sql_directory / "04_create_modeling_view.sql",
    )


def resolve_database_paths(
    config: dict[str, Any],
    project_root: Path,
) -> tuple[Path, Path]:
    """Resolve configured raw-data and DuckDB paths."""
    raw_file = project_root / config["data"]["raw_file"]
    database_path = project_root / config["paths"]["database"]

    return raw_file, database_path


def validate_raw_data_file(raw_file: Path) -> None:
    """Raise an informative error when the configured raw file is absent."""
    if raw_file.exists():
        return

    raise FileNotFoundError(
        f"Raw data file not found: {raw_file}\n"
        f"Place the Kaggle training file at: {raw_file}"
    )


def build_database() -> DatabaseBuildResult:
    """Build all DuckDB tables and views from the configured raw dataset."""
    config = load_config()
    project_root = get_project_root()

    raw_file, database_path = resolve_database_paths(
        config=config,
        project_root=project_root,
    )

    sql_files = get_database_sql_files(project_root)

    validate_raw_data_file(raw_file)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building DuckDB database at: %s", database_path)

    with duckdb.connect(str(database_path)) as connection:
        for sql_path in sql_files:
            run_sql_file(connection, sql_path)

        row_count_result = connection.execute(
            "SELECT COUNT(*) FROM sf_crime_modeling"
        ).fetchone()

        if row_count_result is None:
            raise RuntimeError("DuckDB did not return a modeling row count.")

        modeling_row_count = int(row_count_result[0])

    result = DatabaseBuildResult(
        database_path=database_path,
        modeling_row_count=modeling_row_count,
        executed_sql_files=sql_files,
    )

    logger.info(
        "Database build complete. Modeling row count: %s",
        result.modeling_row_count,
    )

    return result


if __name__ == "__main__":
    build_database()


if __name__ == "__main__":
    build_database()