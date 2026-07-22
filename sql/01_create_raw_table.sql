CREATE OR REPLACE TABLE sf_crime_raw AS
SELECT *
FROM read_csv_auto(
    'data/raw/san_francisco_crime_train.csv',
    header = true,
    ignore_errors = true
);