# Illustrative San Diego departure schedule

This is a small, **synthetic** fixture for the tail-classifier demo. The flight
numbers, departures, and route assignments are invented; they are not a live,
historical, or authoritative flight schedule. They must not be used for travel
decisions. The fixture covers October 26, 2026, from 07:00 through 22:40 San
Diego local time, with five airlines staggered ten minutes apart in each hour.
SAN has one runway; the stagger is illustrative and does not model actual runway
capacity. See the [airport authority](https://www.san.org/about-the-authority/history-of-san/).

## Two Unity Catalog tables

`demo_airlines` maps `airline_id` directly to the model's lowercase `airline`
output. It also holds the airline name and IATA/ICAO codes. `demo_departures`
stores one operating flight per row, its synthetic unique ID, operating airline,
flight number, origin, destination code and city, and scheduled **gate** departure
as a `TIMESTAMP` instant. The CSVs use explicit UTC offsets. Operating carrier is
used because the classifier sees the painted aircraft, not a codeshare's marketing
airline. These are example managed Delta tables in `dbx_ai_app.airplane_tail_id`.

The supplied CSVs have unique IDs, unique airline labels and valid references;
`NOT NULL` is enforced by Delta. These tables do not enforce uniqueness or the
airline relationship. Validate incoming feeds before loading: Databricks [primary
and foreign keys are informational](https://docs.databricks.com/aws/en/tables/constraints).

## Load the sample

Use a Unity Catalog-enabled SQL warehouse or notebook with `USE CATALOG`,
`USE SCHEMA`, `CREATE TABLE`, table `MODIFY`, and the relevant Volume read/write
permissions. The existing `training_data` Volume and catalog/schema are used
to match this repository's setup; change all three-part names and Volume paths
consistently if your environment differs. Create the catalog, schema, and Volume
first if needed, as described in the main README. From the repository root:

```powershell
databricks fs mkdir dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_demo --profile YOUR_PROFILE
databricks fs cp schedule/airlines.csv dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_demo/airlines.csv --profile YOUR_PROFILE
databricks fs cp schedule/departures.csv dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_demo/departures.csv --profile YOUR_PROFILE
```

Run `create_tables.sql` and then `load_sample.sql` in a SQL editor or notebook.
`COPY INTO` remembers loaded file paths and can be rerun without duplicating the
same files. It is **not** a synchronization mechanism: it does not update existing
rows when a file changes. Use a separately designed ingestion/upsert process for
refreshable operational data. Both table names are deliberately demo-specific;
the scripts do not drop or overwrite tables. After loading, check:

Run `validate_tables.sql` to check counts, duplicate IDs, and orphaned airlines.
A fresh load should contain 5 airlines and 80 departures, with no duplicate IDs
or orphaned airlines. `COPY INTO` can load
CSV files from [Unity Catalog Volumes](https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/copy-into/unity-catalog).

## Example lookup and limits

Run `lookup_example.sql` for an example image classified as `southwest` and
observed at `2026-10-26T15:20:00Z` (08:20 Pacific). It finds SAN flights with
scheduled gate departures from 60 minutes before to 10 minutes after the
observation and ranks by absolute time difference. Those bounds are demo
assumptions, not calibrated taxi or delay estimates. This sample returns two
candidates, one scheduled 10 minutes after and one 50 minutes before the sighting.
A runway sighting is not a
gate departure: delays, early pushback, taxi time and similar departure times can
produce multiple plausible matches or none. Show all candidates and their offsets;
do not label the top result as a confirmed flight or destination. The model is
also uncalibrated and expects a tail crop, so its predicted airline can be wrong.

Store observed instants with explicit offsets or convert local time with the
IANA zone `America/Los_Angeles` (including daylight-saving transitions); do not
assume a fixed UTC offset. The SQL searches timestamps directly without filtering
on a UTC calendar date, so windows crossing midnight remain eligible. In the app,
bind the prediction, origin, observation instant and window as parameters rather
than interpolating user input. Replace the fixture with an appropriately licensed,
regularly refreshed schedule and, when available, real movement/status data.
