-- Demo tables in the same catalog and schema as the classifier. Requires
-- USE CATALOG, USE SCHEMA and CREATE TABLE privileges. No tables are replaced.
CREATE TABLE IF NOT EXISTS dbx_ai_app.airplane_tail_id.demo_airlines (
  airline_id STRING NOT NULL COMMENT 'Lowercase classifier output label',
  airline_name STRING NOT NULL,
  iata_code STRING NOT NULL,
  icao_code STRING NOT NULL
) USING DELTA
COMMENT 'Demo airline dimension; airline_id joins to the classifier output.';

CREATE TABLE IF NOT EXISTS dbx_ai_app.airplane_tail_id.demo_weekly_departures (
  departure_id STRING NOT NULL COMMENT 'Unique synthetic weekly schedule row ID',
  airline_id STRING NOT NULL COMMENT 'Operating carrier and classifier label',
  flight_number STRING NOT NULL COMMENT 'Synthetic operating flight number',
  origin_iata STRING NOT NULL,
  destination_iata STRING NOT NULL,
  destination_city STRING NOT NULL,
  day_of_week STRING NOT NULL COMMENT 'MON through SUN, in airport local time',
  scheduled_departure_local STRING NOT NULL COMMENT 'Local gate departure time, HH:mm:ss; not takeoff'
) USING DELTA
COMMENT 'Synthetic recurring weekly departure schedule, not a live flight feed.';
