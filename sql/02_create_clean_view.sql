CREATE OR REPLACE VIEW sf_crime_clean AS
WITH standardized AS (
    SELECT
        TRIM(Category) AS category,
        Descript AS descript,
        TRIM(DayOfWeek) AS day_of_week_name,
        TRIM(PdDistrict) AS pd_district,
        Resolution AS resolution,
        TRIM(
            REGEXP_REPLACE(Address, '\\s*/\\s*', ' / ', 'g')
        ) AS address,
        CAST(X AS DOUBLE) AS longitude,
        CAST(Y AS DOUBLE) AS latitude,
        TRY_CAST(Dates AS TIMESTAMP) AS incident_timestamp
    FROM sf_crime_raw
)
SELECT
    category,
    descript,
    day_of_week_name,
    pd_district,
    resolution,
    address,
    longitude,
    latitude,
    incident_timestamp
FROM standardized
WHERE
    category IS NOT NULL
    AND category <> ''
    AND pd_district IS NOT NULL
    AND incident_timestamp IS NOT NULL
    AND longitude IS NOT NULL
    AND latitude IS NOT NULL

    -- Reproduce the original notebook's invalid-coordinate filter.
    AND longitude < -122;