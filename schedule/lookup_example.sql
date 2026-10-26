-- Demo inputs: classifier label and actual observation instant with UTC offset.
-- Replace inputs with bound parameters in the application; do not interpolate
-- user strings into SQL. Return every candidate in the window, not only one.
SET TIME ZONE 'UTC';

WITH observation AS (
  SELECT 'southwest' AS predicted_airline,
         TIMESTAMP'2026-10-26 15:20:00+00:00' AS observed_at
)
SELECT a.airline_name, d.flight_number, d.origin_iata,
       d.destination_iata, d.destination_city, d.scheduled_departure_utc,
       (unix_timestamp(o.observed_at) - unix_timestamp(d.scheduled_departure_utc)) / 60.0
         AS minutes_after_scheduled_gate_departure
FROM observation o
JOIN dbx_ai_app.airplane_tail_id.demo_airlines a
  ON a.airline_id = o.predicted_airline
JOIN dbx_ai_app.airplane_tail_id.demo_departures d
  ON d.airline_id = a.airline_id
WHERE d.origin_iata = 'SAN'
  AND d.scheduled_departure_utc BETWEEN o.observed_at - INTERVAL 60 MINUTES
                                    AND o.observed_at + INTERVAL 10 MINUTES
ORDER BY abs(unix_timestamp(o.observed_at) - unix_timestamp(d.scheduled_departure_utc)),
         d.scheduled_departure_utc, d.departure_id;
