CREATE OR REPLACE VIEW sf_crime_clean AS
SELECT
    Category,
    Descript,
    DayOfWeek,
    PdDistrict,
    Resolution,
    Address,
    X AS longitude,
    Y AS latitude,
    Dates,
    CAST(Dates AS TIMESTAMP) AS incident_timestamp,
    EXTRACT(year FROM CAST(Dates AS TIMESTAMP)) AS incident_year,
    EXTRACT(month FROM CAST(Dates AS TIMESTAMP)) AS incident_month,
    EXTRACT(day FROM CAST(Dates AS TIMESTAMP)) AS incident_day,
    EXTRACT(hour FROM CAST(Dates AS TIMESTAMP)) AS incident_hour,
    EXTRACT(dow FROM CAST(Dates AS TIMESTAMP)) AS incident_day_of_week_num,
    CASE
        WHEN DayOfWeek IN ('Saturday', 'Sunday') THEN 1
        ELSE 0
    END AS is_weekend,
    CASE
        WHEN Address LIKE '%/%' THEN 1
        ELSE 0
    END AS is_intersection
FROM sf_crime_raw
WHERE
    Category IS NOT NULL
    AND Dates IS NOT NULL
    AND X IS NOT NULL
    AND Y IS NOT NULL;