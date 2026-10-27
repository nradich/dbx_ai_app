"""San Diego sighting times and the recurring weekly departure window.

Pure logic with no Databricks imports. `match_departures` follows the reference
`match_occurrences` in tests/test_schedule_fixture.py and schedule/lookup_example.sql:
check the local observation date and its neighbours, turn matching weekday rows
into instants in America/Los_Angeles, and keep scheduled gate departures from 60
minutes before to 10 minutes after the sighting.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import re
from zoneinfo import ZoneInfo

AIRPORT_IATA = "SAN"
AIRPORT_NAME = "San Diego International Airport"
TIME_ZONE = "America/Los_Angeles"
PACIFIC = ZoneInfo(TIME_ZONE)
DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
WINDOW_BEFORE_MIN = 60
WINDOW_AFTER_MIN = 10
MAX_FUTURE = timedelta(minutes=10)  # tolerance for device clock skew

_LOCAL_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$")


def iso_z(moment: datetime) -> str:
    """Format an aware datetime as 'YYYY-MM-DDTHH:MM:SSZ' in UTC."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_local_string(now_utc: datetime) -> str:
    """Current San Diego wall-clock time with minute precision."""
    return now_utc.astimezone(PACIFIC).strftime("%Y-%m-%dT%H:%M")


def parse_observed_local(value: str, now_utc: datetime) -> tuple[str, datetime]:
    """Validate a San Diego wall-clock sighting time and return (normalized text, UTC instant).

    Repeated times at the end of daylight saving resolve to their first occurrence;
    skipped times at its start are rejected.
    """
    text = (value or "").strip()
    if not _LOCAL_PATTERN.match(text):
        raise ValueError("Enter the sighting time as a date and time, for example 2026-10-27T08:20.")
    try:
        naive = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError("The sighting time is not a valid date and time.") from None
    if not 2000 <= naive.year <= 2100:
        raise ValueError("The sighting year must be between 2000 and 2100.")
    local = naive.replace(tzinfo=PACIFIC, fold=0)
    observed_at = local.astimezone(timezone.utc)
    if observed_at.astimezone(PACIFIC).replace(tzinfo=None) != naive:
        raise ValueError("That time does not exist in San Diego because clocks moved forward. "
                         "Choose a time outside the skipped hour.")
    if observed_at - now_utc > MAX_FUTURE:
        raise ValueError("The sighting time is in the future. Enter when you saw the aircraft.")
    pattern = "%Y-%m-%dT%H:%M" if naive.second == 0 else "%Y-%m-%dT%H:%M:%S"
    return naive.strftime(pattern), observed_at


def match_departures(rows, observed_at: datetime) -> list[dict]:
    """Return scheduled departures near the sighting, closest first.

    `rows` are weekly schedule rows already filtered to one airline and origin.
    `minutes_after_scheduled_gate_departure` is the sighting minus the scheduled
    departure: negative means the departure is scheduled after the sighting.
    """
    observed_at = observed_at.astimezone(timezone.utc)
    observed_day = observed_at.astimezone(PACIFIC).date()
    earliest = observed_at - timedelta(minutes=WINDOW_BEFORE_MIN)
    latest = observed_at + timedelta(minutes=WINDOW_AFTER_MIN)
    matches = []
    for offset in (-1, 0, 1):
        local_day: date = observed_day + timedelta(days=offset)
        day_name = DAYS[local_day.weekday()]
        for row in rows:
            if row.get("day_of_week") != day_name:
                continue
            try:
                clock = time.fromisoformat(row["scheduled_departure_local"])
            except (KeyError, TypeError, ValueError):
                continue
            instant = datetime.combine(local_day, clock, PACIFIC).astimezone(timezone.utc)
            if earliest <= instant <= latest:
                matches.append((abs((observed_at - instant).total_seconds()), instant,
                                str(row.get("departure_id") or ""), local_day, row))
    matches.sort(key=lambda item: item[:3])
    return [{
        "flight_number": row.get("flight_number"),
        "destination_iata": row.get("destination_iata"),
        "destination_city": row.get("destination_city"),
        "day_of_week": row.get("day_of_week"),
        "local_date": local_day.isoformat(),
        "scheduled_departure_local": row.get("scheduled_departure_local"),
        "scheduled_departure_utc": iso_z(instant),
        "minutes_after_scheduled_gate_departure": (observed_at - instant).total_seconds() / 60,
    } for _, instant, _, local_day, row in matches]
