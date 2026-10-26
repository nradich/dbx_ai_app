-- Replace these examples with bound app parameters. The observation is an
-- instant; the weekly schedule is in SAN local time. Keep the session in UTC.
SET TIME ZONE 'UTC';

WITH observation AS (
  SELECT 'southwest' AS predicted_airline,
         TIMESTAMP'2026-10-27 15:20:00+00:00' AS observed_at
), local_day AS (
  SELECT *, CAST(from_utc_timestamp(observed_at, 'America/Los_Angeles') AS DATE) AS observed_local_date
  FROM observation
), possible_days AS (
  SELECT predicted_airline, observed_at,
         explode(array(date_sub(observed_local_date, 1), observed_local_date,
                       date_add(observed_local_date, 1))) AS local_date
  FROM local_day
), occurrences AS (
  SELECT a.airline_name, d.departure_id, d.flight_number, d.origin_iata,
         d.destination_iata, d.destination_city, d.day_of_week,
         d.scheduled_departure_local, p.local_date, p.observed_at,
         to_utc_timestamp(
           to_timestamp(concat(CAST(p.local_date AS STRING), ' ', d.scheduled_departure_local)),
           'America/Los_Angeles'
         ) AS scheduled_departure_utc
  FROM possible_days p
  JOIN dbx_ai_app.airplane_tail_id.demo_airlines a ON a.airline_id = p.predicted_airline
  JOIN dbx_ai_app.airplane_tail_id.demo_weekly_departures d ON d.airline_id = a.airline_id
  WHERE d.origin_iata = 'SAN'
    AND d.day_of_week = element_at(array('MON','TUE','WED','THU','FRI','SAT','SUN'),
                                  weekday(p.local_date) + 1)
)
SELECT airline_name, flight_number, origin_iata, destination_iata, destination_city,
       local_date, day_of_week, scheduled_departure_local, scheduled_departure_utc,
       (unix_timestamp(observed_at) - unix_timestamp(scheduled_departure_utc)) / 60.0
         AS minutes_after_scheduled_gate_departure
FROM occurrences
WHERE scheduled_departure_utc BETWEEN observed_at - INTERVAL 60 MINUTES
                                  AND observed_at + INTERVAL 10 MINUTES
ORDER BY abs(unix_timestamp(observed_at) - unix_timestamp(scheduled_departure_utc)),
         scheduled_departure_utc, departure_id;
