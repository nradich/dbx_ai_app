-- The first two queries should return 5 and 560 on a fresh fixture load.
SELECT count(*) AS airline_count FROM dbx_ai_app.airplane_tail_id.demo_airlines;
SELECT count(*) AS departure_count FROM dbx_ai_app.airplane_tail_id.demo_weekly_departures;

-- Each query below should return zero rows. Unlike NOT NULL, uniqueness and
-- references are not enforced by the demo tables.
SELECT airline_id, count(*) AS copies
FROM dbx_ai_app.airplane_tail_id.demo_airlines
GROUP BY airline_id HAVING count(*) > 1;

SELECT departure_id, count(*) AS copies
FROM dbx_ai_app.airplane_tail_id.demo_weekly_departures
GROUP BY departure_id HAVING count(*) > 1;

SELECT d.departure_id, d.airline_id
FROM dbx_ai_app.airplane_tail_id.demo_weekly_departures d
LEFT ANTI JOIN dbx_ai_app.airplane_tail_id.demo_airlines a
  ON d.airline_id = a.airline_id;

SELECT departure_id, day_of_week, scheduled_departure_local
FROM dbx_ai_app.airplane_tail_id.demo_weekly_departures
WHERE day_of_week NOT IN ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')
   OR NOT (scheduled_departure_local RLIKE '^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$');
