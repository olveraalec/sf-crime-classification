CREATE OR REPLACE VIEW sf_crime_modeling AS
SELECT
    category AS target,
    incident_timestamp,

    -- Calendar features
    incident_year,
    incident_month,
    incident_day,
    incident_hour,
    incident_minute,
    incident_day_of_week_num,
    incident_day_of_year,
    incident_week_of_year,
    datetime_numeric,
    is_weekend,

    -- Spatial features
    pd_district,
    longitude,
    latitude,

    -- Address features
    address,
    street_1,
    street_2,
    block_number,
    is_intersection

FROM sf_crime_features
WHERE
    category IS NOT NULL
    AND incident_timestamp IS NOT NULL
    AND longitude IS NOT NULL
    AND latitude IS NOT NULL;