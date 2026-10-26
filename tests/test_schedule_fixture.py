"""Validate recurring CSVs and demonstrate their local-time semantics."""
import ast
import csv
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import unittest
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROOT = Path(__file__).resolve().parents[1]
DAYS = ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')
try:
    PACIFIC = ZoneInfo('America/Los_Angeles')
except ZoneInfoNotFoundError:
    PACIFIC = None  # Windows needs the optional tzdata package for IANA zones.


def read_csv(name):
    with (ROOT / 'schedule' / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def match_occurrences(rows, airline, observed):
    """Reference the documented window, including adjacent local calendar days."""
    observed_day = observed.astimezone(PACIFIC).date()
    result = []
    for delta in (-1, 0, 1):
        local_day = observed_day + timedelta(days=delta)
        for row in rows:
            if row['airline_id'] != airline or row['day_of_week'] != DAYS[local_day.weekday()]:
                continue
            local_time = time.fromisoformat(row['scheduled_departure_local'])
            instant = datetime.combine(local_day, local_time, PACIFIC).astimezone(timezone.utc)
            if observed - timedelta(minutes=60) <= instant <= observed + timedelta(minutes=10):
                result.append((row['flight_number'], instant))
    return sorted(result, key=lambda item: (abs((observed-item[1]).total_seconds()), item[1]))


class ScheduleFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.airlines = read_csv('airlines.csv')
        cls.departures = read_csv('departures.csv')

    def test_airlines_match_classifier_labels_and_codes(self):
        tree = ast.parse((ROOT / 'training' / 'prepare_data.py').read_text(encoding='utf-8'))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'CLASSES' for t in n.targets))
        ids = [a['airline_id'] for a in self.airlines]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(ast.literal_eval(assignment.value)))
        for code in ('iata_code', 'icao_code'):
            self.assertEqual(len({a[code] for a in self.airlines}), len(ids))

    def test_weekly_departure_integrity_and_hourly_coverage(self):
        self.assertEqual(len(self.departures), 560)
        ids = {a['airline_id']: a['iata_code'] for a in self.airlines}
        row_ids, flight_days, slots, coverage = set(), set(), set(), set()
        for row in self.departures:
            self.assertEqual(set(row), {'departure_id', 'airline_id', 'flight_number',
                                      'origin_iata', 'destination_iata', 'destination_city',
                                      'day_of_week', 'scheduled_departure_local'})
            self.assertTrue(all(row.values()))
            self.assertIn(row['airline_id'], ids)
            self.assertTrue(row['flight_number'].startswith(ids[row['airline_id']]))
            self.assertEqual(row['origin_iata'], 'SAN')
            self.assertRegex(row['destination_iata'], r'^[A-Z]{3}$')
            self.assertIn(row['day_of_week'], DAYS)
            self.assertRegex(row['scheduled_departure_local'], r'^\d{2}:\d{2}:\d{2}$')
            clock = time.fromisoformat(row['scheduled_departure_local'])
            self.assertIn(clock.hour, range(7, 23))
            row_ids.add(row['departure_id'])
            flight_days.add((row['day_of_week'], row['airline_id'], row['flight_number']))
            slots.add((row['day_of_week'], clock))
            coverage.add((row['day_of_week'], clock.hour, row['airline_id']))
        self.assertEqual(len(row_ids), 560)
        self.assertEqual(len(flight_days), 560)
        self.assertEqual(len(slots), 560)
        self.assertEqual(len(coverage), 7 * 16 * len(ids))

    @unittest.skipIf(PACIFIC is None, 'IANA zone data unavailable; install tzdata on Windows')
    def test_weekly_recurrence_across_daylight_saving(self):
        for local_date, expected_utc in ((date(2026, 10, 27), '2026-10-27T23:00:00+00:00'),
                                         (date(2026, 11, 3), '2026-11-04T00:00:00+00:00')):
            observed = datetime.combine(local_date, time(16, 10), PACIFIC).astimezone(timezone.utc)
            matches = match_occurrences(self.departures, 'alaska', observed)
            self.assertEqual([(flight, instant.isoformat()) for flight, instant in matches],
                             [('AS1010', expected_utc)])

    @unittest.skipIf(PACIFIC is None, 'IANA zone data unavailable; install tzdata on Windows')
    def test_example_and_adjacent_day_window(self):
        observed = datetime.fromisoformat('2026-10-27T15:20:00+00:00')
        result = match_occurrences(self.departures, 'southwest', observed)
        self.assertEqual([int((observed - instant).total_seconds() / 60) for _, instant in result], [-10, 50])
        previous_day = [{'airline_id': 'alaska', 'day_of_week': 'MON',
                         'flight_number': 'AS999', 'scheduled_departure_local': '23:55:00'}]
        after_midnight = datetime(2026, 10, 27, 0, 5, tzinfo=PACIFIC).astimezone(timezone.utc)
        self.assertEqual(match_occurrences(previous_day, 'alaska', after_midnight)[0][0], 'AS999')


if __name__ == '__main__':
    unittest.main()
