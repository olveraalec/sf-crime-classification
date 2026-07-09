CREATE OR REPLACE VIEW sf_crime_modeling AS
SELECT
    Category AS target,

    -- Time features
    incident_year,
    incident_month,
    incident_day,
    incident_hour,
    incident_day_of_week_num,
    DayOfWeek AS day_of_week,
    is_weekend,

    -- Location / district features
    PdDistrict AS pd_district,
    longitude,
    latitude,

    -- Address features
    Address AS address,
    is_intersection

FROM sf_crime_clean
WHERE
    target IS NOT NULL
    AND incident_timestamp IS NOT NULL
    AND longitude IS NOT NULL
    AND latitude IS NOT NULL;