CREATE OR REPLACE TABLE sf_crime_raw AS
SELECT *
FROM read_csv_auto(
    'data/raw/train.csv',
    header = true,
    ignore_errors = true
);