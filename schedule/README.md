# Illustrative weekly San Diego departures

This is a **synthetic** recurring schedule for the tail-classifier demo. Flight
numbers, times, and route assignments are invented; this is not an actual,
historical, or live flight schedule and must not be used for travel decisions.
Each of the five airlines has a departure in every hour from 07:00 through 22:40
San Diego local time, staggered by ten minutes. The same pattern repeats on all
seven days. SAN has one runway; the stagger does not model actual runway capacity.
See the [airport authority](https://www.san.org/about-the-authority/history-of-san/).

## Two Unity Catalog tables

`demo_airlines` maps `airline_id` directly to the model's lowercase `airline`
output and holds airline names and IATA/ICAO codes. `demo_weekly_departures`
contains one recurring departure per airline, flight and day of the week. Its
`day_of_week` is `MON`, `TUE`, `WED`, `THU`, `FRI`, `SAT`, or `SUN`;
`scheduled_departure_local` is an `HH:mm:ss` string in **America/Los_Angeles**
wall-clock time. It is a scheduled **gate** departure, not takeoff. Flight numbers
are intentionally reused across days. The sample's `departure_id` is unique per
weekly row. There is no date or fixed UTC offset in the schedule CSV.

For example, Tuesday's 16:00 departure repeats every Tuesday at 16:00 San Diego
time; its UTC instant changes with daylight saving time. All example rows operate
every day, although the weekday field permits different schedules on different
days. Operating airline is used because the classifier sees the painted aircraft,
not a codeshare's marketing airline. This first version assumes SAN for every row;
other airports would need their own IANA time zone in the data model.

The supplied CSVs have unique IDs and valid airline references; Delta enforces
`NOT NULL`. Uniqueness, relationships, valid weekday values and time formats are
not enforced by these tables. Validate inputs before loading. Databricks [primary
and foreign keys are informational](https://docs.databricks.com/aws/en/tables/constraints).

## Load the sample

Use a Unity Catalog-enabled SQL warehouse or notebook with `USE CATALOG`,
`USE SCHEMA`, `CREATE TABLE`, table `MODIFY`, and Volume read/write permissions.
The existing `training_data` Volume and catalog/schema match this repository's
setup; change all three-part names and Volume paths consistently if needed. Create
the catalog, schema, and Volume first if necessary, as in the main README.
From the repository root:

```powershell
databricks fs mkdir dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_demo --profile YOUR_PROFILE
databricks fs mkdir dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_weekly_demo --profile YOUR_PROFILE
databricks fs cp schedule/airlines.csv dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_demo/airlines.csv --profile YOUR_PROFILE
databricks fs cp schedule/departures.csv dbfs:/Volumes/dbx_ai_app/airplane_tail_id/training_data/schedule_weekly_demo/departures.csv --profile YOUR_PROFILE
```

Run `create_tables.sql`, then `load_sample.sql`, then `validate_tables.sql` in a
SQL editor or notebook. A fresh load should contain 5 airlines and 560 weekly
departure rows, with no duplicate IDs, orphaned airlines, or invalid weekdays or
times. The weekly table and path have new names so they do not overwrite the
previous dated demo table or CSV, if those were loaded separately. The airline
CSV retains its original path, so a prior `COPY INTO` is not duplicated.
If that airline CSV is already in the Volume at the shown path, skip its copy
command; the CLI does not overwrite an existing file without `--overwrite`.

`COPY INTO` remembers loaded file paths and can be rerun without duplicating the
same files. It does not synchronize changed rows; use a separately designed
ingestion/upsert process for a changing schedule. The scripts do not drop or
overwrite tables. See [loading from Volumes](https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/copy-into/unity-catalog).

## Lookup and limits

`lookup_example.sql` uses an observation of `2026-10-27T15:20:00Z` (Tuesday,
08:20 Pacific) and returns two Southwest candidates scheduled at 07:30 and 08:30
local time. The dates belong to the observation and calculated occurrences, not
the recurring schedule. For each sighting, the query checks the local observation
date and the days immediately before and after, constructs actual instants from
matching weekday/time rows using `America/Los_Angeles`, and ranks candidates
whose scheduled gate departures are from 60 minutes before to 10 minutes after
the sighting. The same Tuesday slots repeat the following Tuesday, even if the
UTC offset has changed. The query sets its session time zone to UTC; preserve that
assumption if adapting it. The demo's 07:00–22:40 slots avoid the ambiguous or
nonexistent local times around daylight-saving transitions; an expanded schedule
that includes those hours needs an explicit resolution policy.

The window is illustrative, not a calibrated taxi or delay estimate. A runway
sighting is not a gate departure: delays, early pushback and taxi time can produce
multiple plausible matches or none. Show all candidates and offsets, not a confirmed
flight. The classifier is uncalibrated, expects a tail crop, and can predict the
wrong airline. Bind prediction, origin, observation and window parameters in the
app instead of interpolating user input. Replace the fixture with an appropriately
licensed, regularly refreshed schedule and actual movement/status data when available.
