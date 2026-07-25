from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.build_database import (
    get_database_sql_files,
    resolve_database_paths,
    run_sql_file,
    validate_raw_data_file,
)


def test_run_sql_file_executes_file_contents(
    tmp_path: Path,
) -> None:
    sql_path = tmp_path / "example.sql"
    sql_path.write_text(
        "CREATE TABLE example AS SELECT 1 AS value;",
        encoding="utf-8",
    )

    connection = MagicMock()

    run_sql_file(connection, sql_path)

    connection.execute.assert_called_once_with(
        "CREATE TABLE example AS SELECT 1 AS value;"
    )


def test_run_sql_file_raises_when_file_is_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.sql"
    connection = MagicMock()

    with pytest.raises(
        FileNotFoundError,
        match="SQL file not found",
    ):
        run_sql_file(connection, missing_path)

    connection.execute.assert_not_called()


def test_database_sql_files_preserve_required_order(
    tmp_path: Path,
) -> None:
    result = get_database_sql_files(tmp_path)

    assert result == (
        tmp_path / "sql" / "01_create_raw_table.sql",
        tmp_path / "sql" / "02_create_clean_view.sql",
        tmp_path / "sql" / "03_create_feature_view.sql",
        tmp_path / "sql" / "04_create_modeling_view.sql",
    )


def test_resolve_database_paths_uses_project_root(
    tmp_path: Path,
) -> None:
    config = {
        "data": {
            "raw_file": "data/raw/train.csv",
        },
        "paths": {
            "database": "data/database/test.duckdb",
        },
    }

    raw_file, database_path = resolve_database_paths(
        config=config,
        project_root=tmp_path,
    )

    assert raw_file == tmp_path / "data/raw/train.csv"
    assert database_path == (tmp_path / "data/database/test.duckdb")


def test_validate_raw_data_file_accepts_existing_file(
    tmp_path: Path,
) -> None:
    raw_file = tmp_path / "train.csv"
    raw_file.write_text("example", encoding="utf-8")

    validate_raw_data_file(raw_file)


def test_validate_raw_data_file_reports_configured_path(
    tmp_path: Path,
) -> None:
    raw_file = tmp_path / "data" / "raw" / "san_francisco_crime_train.csv"

    with pytest.raises(
        FileNotFoundError,
        match="san_francisco_crime_train.csv",
    ):
        validate_raw_data_file(raw_file)