"""Validate the portable CSV fixture and its join to classifier labels."""
import ast
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read_csv(name):
    with (ROOT / 'schedule' / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


class ScheduleFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.airlines = read_csv('airlines.csv')
        cls.departures = read_csv('departures.csv')

    def test_airlines_match_classifier_labels_and_codes(self):
        tree = ast.parse((ROOT / 'training' / 'prepare_data.py').read_text(encoding='utf-8'))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'CLASSES' for t in n.targets))
        model_classes = set(ast.literal_eval(assignment.value))
        ids = [a['airline_id'] for a in self.airlines]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), model_classes)
        for code in ('iata_code', 'icao_code'):
            self.assertEqual(len({a[code] for a in self.airlines}), len(ids))

    def test_departure_integrity_and_hourly_coverage(self):
        self.assertEqual(len(self.departures), 80)
        ids = {a['airline_id']: a['iata_code'] for a in self.airlines}
        departure_ids, flight_numbers, instants, coverage = set(), set(), set(), set()
        pacific = timezone(timedelta(hours=-7))  # October 26 fixture is in daylight time.
        for row in self.departures:
            self.assertTrue(all(row.values()))
            self.assertIn(row['airline_id'], ids)
            self.assertTrue(row['flight_number'].startswith(ids[row['airline_id']]))
            self.assertEqual(row['origin_iata'], 'SAN')
            self.assertRegex(row['destination_iata'], r'^[A-Z]{3}$')
            self.assertNotEqual(row['destination_iata'], row['origin_iata'])
            self.assertTrue(row['scheduled_departure_utc'].endswith('Z'))
            instant = datetime.fromisoformat(row['scheduled_departure_utc'].replace('Z', '+00:00'))
            local = instant.astimezone(pacific)
            self.assertEqual(local.date().isoformat(), '2026-10-26')
            self.assertIn(local.hour, range(7, 23))
            departure_ids.add(row['departure_id'])
            flight_numbers.add(row['flight_number'])
            instants.add(instant)
            coverage.add((local.hour, row['airline_id']))
        self.assertEqual(len(departure_ids), 80)
        self.assertEqual(len(flight_numbers), 80)
        self.assertEqual(len(instants), 80)
        self.assertEqual(len(coverage), 16 * len(ids))

    def test_example_window_has_two_candidates(self):
        observed = datetime.fromisoformat('2026-10-26T15:20:00+00:00')
        matches = []
        for row in self.departures:
            when = datetime.fromisoformat(row['scheduled_departure_utc'].replace('Z', '+00:00'))
            if (row['airline_id'] == 'southwest' and row['origin_iata'] == 'SAN'
                    and observed - timedelta(minutes=60) <= when <= observed + timedelta(minutes=10)):
                matches.append((observed - when).total_seconds() / 60)
        self.assertEqual(sorted(matches), [-10, 50])


if __name__ == '__main__':
    unittest.main()
