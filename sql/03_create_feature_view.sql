CREATE OR REPLACE VIEW sf_crime_features AS
WITH base_features AS (
    SELECT
        category,
        descript,
        day_of_week_name,
        pd_district,
        resolution,
        address,
        longitude,
        latitude,
        incident_timestamp,

        CAST(EXTRACT(year FROM incident_timestamp) AS INTEGER)
            AS incident_year,

        CAST(EXTRACT(month FROM incident_timestamp) AS INTEGER)
            AS incident_month,

        CAST(EXTRACT(day FROM incident_timestamp) AS INTEGER)
            AS incident_day,

        CAST(EXTRACT(hour FROM incident_timestamp) AS INTEGER)
            AS incident_hour,

        CAST(EXTRACT(minute FROM incident_timestamp) AS INTEGER)
            AS incident_minute,

        CAST(EXTRACT(dow FROM incident_timestamp) AS INTEGER)
            AS incident_day_of_week_num,

        CAST(EXTRACT(doy FROM incident_timestamp) AS INTEGER)
            AS incident_day_of_year,

        CAST(EXTRACT(week FROM incident_timestamp) AS INTEGER)
            AS incident_week_of_year,

        EPOCH(incident_timestamp) AS datetime_numeric,

        CASE
            WHEN day_of_week_name IN ('Saturday', 'Sunday') THEN 1
            ELSE 0
        END AS is_weekend,

        CASE
            WHEN address LIKE '%/%' THEN 1
            ELSE 0
        END AS is_intersection,

        TRY_CAST(
            NULLIF(
                REGEXP_EXTRACT(address, '^([0-9]+)', 1),
                ''
            ) AS INTEGER
        ) AS block_number,

        TRIM(
            REGEXP_REPLACE(
                SPLIT_PART(address, ' / ', 1),
                '^[0-9]+\\s+Block of\\s+',
                '',
                'i'
            )
        ) AS street_1,

        CASE
            WHEN address LIKE '%/%'
            THEN NULLIF(TRIM(SPLIT_PART(address, ' / ', 2)), '')
            ELSE NULL
        END AS street_2

    FROM sf_crime_clean
)
SELECT *
FROM base_features;