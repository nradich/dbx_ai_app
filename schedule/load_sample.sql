-- Upload the two CSVs to the Volume paths below before running. COPY INTO
-- tracks loaded files, so rerunning with the same paths does not append again.
SET TIME ZONE 'UTC';

COPY INTO dbx_ai_app.airplane_tail_id.demo_airlines
FROM '/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_demo/airlines.csv'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'mode' = 'FAILFAST');

COPY INTO dbx_ai_app.airplane_tail_id.demo_departures
FROM '/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_demo/departures.csv'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'mode' = 'FAILFAST');
