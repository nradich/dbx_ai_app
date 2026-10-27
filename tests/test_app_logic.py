"""Pure-logic tests for the Tailspot Databricks App (no network, no Databricks).

Covers app/schedule.py, the upload preparation and settings in app/pipeline.py,
app/serving_client.py, and consistency between the app and its deployment files.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import unittest
from unittest import mock

from PIL import Image, ImageCms, ImageOps, features
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'tests'))

import schedule  # noqa: E402
import pipeline  # noqa: E402
import serving_client  # noqa: E402
from test_schedule_fixture import match_occurrences, read_csv  # noqa: E402

UTC = timezone.utc
PACIFIC = schedule.PACIFIC
AIRLINES = ('alaska', 'american', 'delta', 'southwest', 'united')
XMP_ORIENTATION_6 = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/>'
    b'</rdf:RDF></x:xmpmeta>')


# Helpers ---------------------------------------------------------------------
def utc(*args):
    return datetime(*args, tzinfo=UTC)


def pacific(*args):
    return datetime(*args, tzinfo=PACIFIC)


def patterned(size, mode='RGB'):
    """An image whose pixels all differ, so any rotation or flip is detectable."""
    width, height = size
    image = Image.new('RGB', size)
    for y in range(height):
        for x in range(width):
            image.putpixel((x, y), ((x * 37) % 256, (y * 53) % 256, (x * 11 + y * 29) % 256))
    return image if mode == 'RGB' else image.convert(mode)


def encode(image, fmt, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, fmt, **kwargs)
    return buffer.getvalue()


def with_orientation(orientation, **extra_tags):
    exif = Image.Exif()
    exif[0x0112] = orientation
    for tag, value in extra_tags.items():
        exif[int(tag)] = value
    return exif


def decode_rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        return image.convert('RGB')


def exif_transposed_rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def png_chunk_types(data):
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise AssertionError('not a PNG')
    position, types = 8, []
    while position < len(data):
        length = struct.unpack('>I', data[position:position + 4])[0]
        types.append(data[position + 4:position + 8].decode('ascii'))
        position += 12 + length
    return types


# Schedule --------------------------------------------------------------------
class ScheduleMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        departures = read_csv('departures.csv')
        cls.rows_by_airline = {airline: [row for row in departures if row['airline_id'] == airline]
                               for airline in AIRLINES}

    def test_southwest_example_from_api_contract(self):
        result = schedule.match_departures(self.rows_by_airline['southwest'], utc(2026, 10, 27, 15, 20))
        self.assertEqual(result, [
            {'flight_number': 'WN1002', 'destination_iata': 'PHX', 'destination_city': 'Phoenix',
             'day_of_week': 'TUE', 'local_date': '2026-10-27', 'scheduled_departure_local': '08:30:00',
             'scheduled_departure_utc': '2026-10-27T15:30:00Z',
             'minutes_after_scheduled_gate_departure': -10.0},
            {'flight_number': 'WN1001', 'destination_iata': 'LAS', 'destination_city': 'Las Vegas',
             'day_of_week': 'TUE', 'local_date': '2026-10-27', 'scheduled_departure_local': '07:30:00',
             'scheduled_departure_utc': '2026-10-27T14:30:00Z',
             'minutes_after_scheduled_gate_departure': 50.0},
        ])

    def test_equivalent_to_reference_across_both_dst_transitions(self):
        step = timedelta(minutes=7)
        mismatches, checked, matched = [], 0, 0
        for start, stop in ((utc(2026, 10, 30), utc(2026, 11, 3)),
                            (utc(2027, 3, 12), utc(2027, 3, 16))):
            instants = []
            moment = start
            while moment <= stop:
                instants.append(moment)
                moment += step
            for airline, rows in self.rows_by_airline.items():
                for observed in instants:
                    got = schedule.match_departures(rows, observed)
                    expected = match_occurrences(rows, airline, observed)
                    checked += 1
                    matched += bool(got)
                    got_pairs = [(c['flight_number'], c['scheduled_departure_utc']) for c in got]
                    expected_pairs = [(flight, schedule.iso_z(instant)) for flight, instant in expected]
                    got_minutes = [c['minutes_after_scheduled_gate_departure'] for c in got]
                    expected_minutes = [(observed - instant).total_seconds() / 60
                                        for _, instant in expected]
                    if got_pairs != expected_pairs or got_minutes != expected_minutes:
                        mismatches.append((airline, observed.isoformat(), got_pairs, expected_pairs))
        self.assertEqual(mismatches[:5], [])
        self.assertGreater(checked, 8000)
        self.assertGreater(matched, 1000)  # the sweep really exercises matches

    def test_weekly_recurrence_keeps_local_time_across_dst(self):
        for local_date, expected_utc in (((2026, 10, 27), '2026-10-27T23:00:00Z'),
                                         ((2026, 11, 3), '2026-11-04T00:00:00Z')):
            observed = pacific(*local_date, 16, 10)
            result = schedule.match_departures(self.rows_by_airline['alaska'], observed)
            self.assertEqual([(c['flight_number'], c['scheduled_departure_utc']) for c in result],
                             [('AS1010', expected_utc)])
            self.assertEqual(result[0]['minutes_after_scheduled_gate_departure'], 10.0)

    def test_adjacent_local_days_are_considered(self):
        monday_late = [{'departure_id': 'd1', 'flight_number': 'AS999', 'day_of_week': 'MON',
                        'scheduled_departure_local': '23:55:00'}]
        result = schedule.match_departures(monday_late, pacific(2026, 10, 27, 0, 5))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['flight_number'], 'AS999')
        self.assertEqual(result[0]['local_date'], '2026-10-26')
        self.assertEqual(result[0]['scheduled_departure_utc'], '2026-10-27T06:55:00Z')
        self.assertEqual(result[0]['minutes_after_scheduled_gate_departure'], 10.0)

        tuesday_early = [{'departure_id': 'd2', 'flight_number': 'AS998', 'day_of_week': 'TUE',
                          'scheduled_departure_local': '00:05:00'}]
        result = schedule.match_departures(tuesday_early, pacific(2026, 10, 26, 23, 58))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['local_date'], '2026-10-27')
        self.assertEqual(result[0]['minutes_after_scheduled_gate_departure'], -7.0)

    def test_malformed_rows_are_skipped(self):
        valid = {'departure_id': 'ok', 'flight_number': 'OK1', 'day_of_week': 'TUE',
                 'scheduled_departure_local': '08:30:00'}
        rows = [
            {'departure_id': 'a', 'flight_number': 'BAD1', 'day_of_week': 'TUE',
             'scheduled_departure_local': 'xx'},
            {'departure_id': 'b', 'flight_number': 'BAD2', 'day_of_week': 'TUE'},
            {'departure_id': 'c', 'flight_number': 'BAD3', 'day_of_week': 'FUNDAY',
             'scheduled_departure_local': '08:20:00'},
            {'departure_id': 'd', 'flight_number': 'BAD4', 'day_of_week': 'TUE',
             'scheduled_departure_local': None},
            {'departure_id': 'e', 'flight_number': 'BAD5', 'scheduled_departure_local': '08:20:00'},
            valid,
        ]
        result = schedule.match_departures(rows, utc(2026, 10, 27, 15, 20))
        self.assertEqual([c['flight_number'] for c in result], ['OK1'])

    def test_window_boundaries_are_inclusive(self):
        rows = [{'departure_id': 'early', 'flight_number': 'EARLY', 'day_of_week': 'TUE',
                 'scheduled_departure_local': '07:20:00'},
                {'departure_id': 'late', 'flight_number': 'LATE', 'day_of_week': 'TUE',
                 'scheduled_departure_local': '08:30:00'}]
        sighting = utc(2026, 10, 27, 15, 20)  # 08:20 PDT
        exact = schedule.match_departures(rows, sighting)
        self.assertEqual({c['flight_number']: c['minutes_after_scheduled_gate_departure'] for c in exact},
                         {'EARLY': 60.0, 'LATE': -10.0})
        one_second_later = schedule.match_departures(rows, sighting + timedelta(seconds=1))
        self.assertEqual([c['flight_number'] for c in one_second_later], ['LATE'])
        one_second_earlier = schedule.match_departures(rows, sighting - timedelta(seconds=1))
        self.assertEqual([c['flight_number'] for c in one_second_earlier], ['EARLY'])

    def test_ties_order_by_instant_then_departure_id(self):
        equidistant = [
            {'departure_id': 'a-after', 'flight_number': 'AFTER', 'day_of_week': 'TUE',
             'scheduled_departure_local': '08:25:00'},
            {'departure_id': 'z-before', 'flight_number': 'BEFORE', 'day_of_week': 'TUE',
             'scheduled_departure_local': '08:15:00'},
        ]
        result = schedule.match_departures(equidistant, utc(2026, 10, 27, 15, 20))
        self.assertEqual([c['flight_number'] for c in result], ['BEFORE', 'AFTER'])

        same_instant = [
            {'departure_id': 'id-2', 'flight_number': 'SECOND', 'day_of_week': 'TUE',
             'scheduled_departure_local': '08:30:00'},
            {'departure_id': 'id-1', 'flight_number': 'FIRST', 'day_of_week': 'TUE',
             'scheduled_departure_local': '08:30:00'},
        ]
        result = schedule.match_departures(same_instant, utc(2026, 10, 27, 15, 20))
        self.assertEqual([c['flight_number'] for c in result], ['FIRST', 'SECOND'])

    def test_iso_z_and_now_local_string(self):
        self.assertEqual(schedule.iso_z(pacific(2026, 10, 27, 8, 20, 5, 999999)), '2026-10-27T15:20:05Z')
        self.assertEqual(schedule.iso_z(datetime(2026, 10, 27, 17, 20, tzinfo=timezone(timedelta(hours=2)))),
                         '2026-10-27T15:20:00Z')
        self.assertEqual(schedule.iso_z(utc(2026, 10, 27, 15, 20)), '2026-10-27T15:20:00Z')
        self.assertEqual(schedule.now_local_string(utc(2026, 10, 27, 15, 20, 59)), '2026-10-27T08:20')
        self.assertEqual(schedule.now_local_string(utc(2026, 12, 1, 7, 59)), '2026-11-30T23:59')


# Sighting time parsing -------------------------------------------------------
class ObservedLocalTests(unittest.TestCase):
    NOW = utc(2026, 10, 27, 15, 25)

    def assertRejected(self, value, now=None):
        with self.assertRaises(ValueError) as caught:
            schedule.parse_observed_local(value, now or self.NOW)
        self.assertIsInstance(str(caught.exception), str)
        self.assertTrue(str(caught.exception).strip())
        return str(caught.exception)

    def test_valid_time_is_normalized_and_converted_to_utc(self):
        text, instant = schedule.parse_observed_local('2026-10-27T08:20', self.NOW)
        self.assertEqual(text, '2026-10-27T08:20')
        self.assertEqual(instant, utc(2026, 10, 27, 15, 20))
        self.assertEqual(instant.utcoffset(), timedelta(0))
        self.assertIs(instant.tzinfo, UTC)
        self.assertEqual(schedule.parse_observed_local('2026-10-27T08:20:00', self.NOW)[0],
                         '2026-10-27T08:20')
        text, instant = schedule.parse_observed_local(' 2026-10-27T08:20:05 ', self.NOW)
        self.assertEqual(text, '2026-10-27T08:20:05')
        self.assertEqual(instant, utc(2026, 10, 27, 15, 20, 5))

    def test_skipped_time_rejected_and_repeated_time_uses_first_occurrence(self):
        self.assertRejected('2027-03-14T02:30', now=utc(2027, 3, 15))
        self.assertEqual(schedule.parse_observed_local('2027-03-14T03:30', utc(2027, 3, 15))[1],
                         utc(2027, 3, 14, 10, 30))
        text, instant = schedule.parse_observed_local('2026-11-01T01:30', utc(2026, 11, 2))
        self.assertEqual(text, '2026-11-01T01:30')
        self.assertEqual(schedule.iso_z(instant), '2026-11-01T08:30:00Z')  # PDT, first occurrence

    def test_bad_formats_and_out_of_range_years_rejected(self):
        for value in ('', None, 'yesterday', '2026-10-27 08:20', '2026-13-01T00:00',
                      '2026-10-27T24:00', '2026-10-27T08:20Z', '2026-10-27T08:20+01:00',
                      '2026-02-30T10:00', '2026-10-27T08:20:60', '2026-10-27T8:20',
                      '\uff12\uff10\uff12\uff16-10-27T08:20'):
            with self.subTest(value=value):
                self.assertRejected(value)
        self.assertRejected('1999-12-31T23:59')
        message = self.assertRejected('2101-01-01T00:00', now=utc(2102, 1, 1))
        self.assertIn('2100', message)
        self.assertEqual(schedule.parse_observed_local('2000-01-01T00:00', self.NOW)[1],
                         utc(2000, 1, 1, 8))

    def test_future_limit_is_ten_minutes(self):
        text, instant = schedule.parse_observed_local('2026-10-27T08:35', self.NOW)
        self.assertEqual(instant - self.NOW, timedelta(minutes=10))
        self.assertIn('future', self.assertRejected('2026-10-27T08:36'))
        self.assertRejected('2026-10-27T08:35:01')


# Upload preparation ----------------------------------------------------------
class PrepareUploadTests(unittest.TestCase):
    def assertUserError(self, status, func, *args, **kwargs):
        with self.assertRaises(pipeline.UserError) as caught:
            func(*args, **kwargs)
        self.assertEqual(caught.exception.status_code, status)
        self.assertTrue(caught.exception.message.strip())
        return caught.exception

    def assertMetadataFree(self, png):
        self.assertTrue(set(png_chunk_types(png)) <= {'IHDR', 'IDAT', 'IEND'}, png_chunk_types(png))
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual(image.format, 'PNG')
            self.assertEqual(image.mode, 'RGB')
            image.load()
            for key in ('exif', 'icc_profile', 'xmp', 'XML:com.adobe.xmp', 'dpi'):
                self.assertNotIn(key, image.info)
            self.assertFalse(image.getexif())

    def test_exif_oriented_png_is_cropped_upright_and_stored_without_metadata(self):
        source = patterned((8, 6))
        data = encode(source, 'PNG', exif=with_orientation(6, **{'270': 'private metadata'}),
                      icc_profile=b'private profile')
        box = (0, 0, 1, 0.5)
        prepared = pipeline.prepare_upload(data, box)
        upright = exif_transposed_rgb(data)
        self.assertEqual(upright.size, (6, 8))

        self.assertEqual(decode_rgb(prepared.crop_png).tobytes(), upright.crop((0, 0, 6, 4)).tobytes())
        self.assertEqual(decode_rgb(prepared.original_png).size, (6, 8))
        self.assertEqual(decode_rgb(prepared.original_png).tobytes(), upright.tobytes())
        self.assertMetadataFree(prepared.crop_png)
        self.assertMetadataFree(prepared.original_png)
        self.assertEqual(prepared.crop_png,
                         serving_client.prepare_tail_png(data, box, pipeline.CROP_MAX_SIDE))
        self.assertEqual(prepared.crop_sha256, hashlib.sha256(prepared.crop_png).hexdigest())
        self.assertEqual(prepared.original_sha256, hashlib.sha256(prepared.original_png).hexdigest())
        self.assertEqual(prepared.upload_sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(prepared.upload_bytes, len(data))
        self.assertEqual(prepared.upload_format, 'PNG')
        self.assertEqual((prepared.original_width, prepared.original_height), (6, 8))
        self.assertEqual((prepared.crop_width, prepared.crop_height), (6, 4))
        self.assertFalse(prepared.crop_downscaled)
        self.assertEqual(prepared.crop_box, box)

    def test_seen_size_must_match_upright_size(self):
        data = encode(patterned((8, 6)), 'PNG', exif=with_orientation(6))
        prepared = pipeline.prepare_upload(data, (0, 0, 1, 1), (6, 8))
        self.assertEqual((prepared.original_width, prepared.original_height), (6, 8))
        self.assertUserError(422, pipeline.prepare_upload, data, (0, 0, 1, 1), (8, 6))

    def test_parse_seen_size(self):
        self.assertIsNone(pipeline.parse_seen_size(None, None))
        self.assertIsNone(pipeline.parse_seen_size('', ''))
        self.assertIsNone(pipeline.parse_seen_size(None, ''))
        self.assertEqual(pipeline.parse_seen_size('10', '20'), (10, 20))
        for width, height in (('10', None), (None, '10'), ('10', ''), ('a', '2'), ('0', '5'),
                              ('-1', '5'), ('5', '0'), ('1.5', '2')):
            with self.subTest(width=width, height=height):
                self.assertUserError(422, pipeline.parse_seen_size, width, height)

    def test_xmp_only_orientation_is_ignored_unlike_exif_transpose(self):
        data = encode(patterned((4, 2)), 'JPEG', xmp=XMP_ORIENTATION_6, quality=95)
        with Image.open(io.BytesIO(data)) as image:
            self.assertNotIn('exif', image.info)
            self.assertIn('xmp', image.info)
            self.assertEqual(serving_client.exif_orientation(image), 1)
        self.assertEqual(serving_client.oriented_rgb(data).size, (4, 2))
        self.assertEqual(serving_client.oriented_rgb(data).tobytes(), decode_rgb(data).tobytes())
        # Pillow would rotate it: documents the intended difference (browsers ignore XMP).
        self.assertEqual(exif_transposed_rgb(data).size, (2, 4))
        prepared = pipeline.prepare_upload(data, (0, 0, 1, 1), (4, 2))
        self.assertEqual((prepared.original_width, prepared.original_height), (4, 2))
        self.assertUserError(422, pipeline.prepare_upload, data, (0, 0, 1, 1), (2, 4))

    def test_all_exif_orientations_match_exif_transpose(self):
        source = patterned((5, 3))
        for fmt in ('JPEG', 'PNG'):
            for orientation in range(1, 9):
                with self.subTest(fmt=fmt, orientation=orientation):
                    data = encode(source, fmt, exif=with_orientation(orientation))
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        self.assertEqual(serving_client.exif_orientation(image), orientation)
                    got = serving_client.oriented_rgb(data)
                    expected = exif_transposed_rgb(data)
                    self.assertEqual(got.mode, 'RGB')
                    self.assertEqual(got.size, expected.size)
                    self.assertEqual(got.tobytes(), expected.tobytes())
                    self.assertEqual(got.info, {})

    def test_long_crop_is_downscaled_to_max_side(self):
        data = encode(patterned((3200, 200)), 'JPEG', quality=90)
        prepared = pipeline.prepare_upload(data, (0, 0, 1, 1))
        self.assertEqual((prepared.crop_width, prepared.crop_height), (1536, 96))
        self.assertTrue(prepared.crop_downscaled)
        self.assertEqual((prepared.original_width, prepared.original_height), (3200, 200))
        self.assertEqual(decode_rgb(prepared.crop_png).size, (1536, 96))
        self.assertEqual(prepared.crop_png,
                         serving_client.prepare_tail_png(data, (0, 0, 1, 1), pipeline.CROP_MAX_SIDE))
        self.assertEqual(prepared.upload_format, 'JPEG')

    def test_crop_at_exactly_max_side_is_not_downscaled(self):
        exact = pipeline.prepare_upload(encode(Image.new('RGB', (1536, 40), 'navy'), 'PNG'), (0, 0, 1, 1))
        self.assertEqual((exact.crop_width, exact.crop_height), (1536, 40))
        self.assertFalse(exact.crop_downscaled)
        over = pipeline.prepare_upload(encode(Image.new('RGB', (1537, 40), 'navy'), 'PNG'), (0, 0, 1, 1))
        self.assertEqual(over.crop_width, 1536)
        self.assertTrue(over.crop_downscaled)

    def test_jpeg_exif_gps_and_icc_profile_are_not_stored(self):
        exif = with_orientation(3, **{'270': 'private description', '271': 'PrivateMake',
                                      '272': 'PrivateModel'})
        gps = exif.get_ifd(0x8825)
        gps[1], gps[2] = 'N', (32.0, 44.0, 0.0)
        gps[3], gps[4] = 'W', (117.0, 11.0, 0.0)
        if features.check('littlecms2'):
            icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        else:
            icc = b'private icc profile'
        data = encode(patterned((20, 10)), 'JPEG', exif=exif, icc_profile=icc, quality=95)
        with Image.open(io.BytesIO(data)) as image:  # the fixture really carries the metadata
            self.assertEqual(dict(image.getexif().get_ifd(0x8825))[1], 'N')
            self.assertEqual(image.info.get('icc_profile'), icc)
        prepared = pipeline.prepare_upload(data, (0.1, 0.1, 0.9, 0.9))
        for png in (prepared.original_png, prepared.crop_png):
            self.assertMetadataFree(png)
            for secret in (b'private', b'Private', b'GPS', icc[:32]):
                self.assertNotIn(secret, png)
        self.assertEqual(decode_rgb(prepared.original_png).tobytes(), exif_transposed_rgb(data).tobytes())

    def test_palette_rgba_gray_cmyk_and_mpo_are_accepted_as_rgb(self):
        palette = Image.new('P', (6, 4))
        palette.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (768 - 9))
        for x in range(6):
            for y in range(4):
                palette.putpixel((x, y), (x + y) % 3)
        rgba = Image.new('RGBA', (6, 4), (10, 20, 30, 0))
        rgba.putpixel((0, 0), (200, 100, 50, 255))
        cases = {
            'P': (encode(palette, 'PNG'), palette.convert('RGB'), 'PNG'),
            'RGBA': (encode(rgba, 'PNG'), rgba.convert('RGB'), 'PNG'),
            'L': (encode(patterned((6, 4), 'L'), 'PNG'), patterned((6, 4), 'L').convert('RGB'), 'PNG'),
            'LA': (encode(patterned((6, 4), 'LA'), 'PNG'), patterned((6, 4), 'LA').convert('RGB'), 'PNG'),
        }
        cmyk = encode(patterned((6, 4), 'CMYK'), 'JPEG', quality=95)
        cases['CMYK'] = (cmyk, decode_rgb(cmyk), 'JPEG')
        mpo = encode(patterned((6, 4)), 'MPO', save_all=True,
                     append_images=[Image.new('RGB', (6, 4), 'blue')], quality=95)
        with Image.open(io.BytesIO(mpo)) as first_frame:
            cases['MPO'] = (mpo, first_frame.convert('RGB'), 'MPO')
        for name, (data, expected, fmt) in cases.items():
            with self.subTest(mode=name):
                prepared = pipeline.prepare_upload(data, (0, 0, 1, 1))
                self.assertEqual(prepared.upload_format, fmt)
                self.assertEqual(decode_rgb(prepared.original_png).tobytes(), expected.tobytes())
                self.assertEqual(decode_rgb(prepared.crop_png).tobytes(), expected.tobytes())
                self.assertMetadataFree(prepared.original_png)

    def test_16bit_grayscale_png_keeps_its_brightness(self):
        # BUG: Pillow clips I;16 -> RGB instead of scaling, so a mid-gray 16-bit PNG
        # (displayed mid-gray by browsers) is stored and sent to the model as white.
        image = Image.new('I;16', (4, 4), 32768)
        data = encode(image, 'PNG')
        with Image.open(io.BytesIO(data)) as decoded:
            self.assertIn(decoded.mode, ('I;16', 'I'))
        prepared = pipeline.prepare_upload(data, (0, 0, 1, 1))
        red, green, blue = decode_rgb(prepared.crop_png).getpixel((0, 0))
        self.assertAlmostEqual(red, 128, delta=2)
        self.assertEqual(red, green)
        self.assertEqual(green, blue)

    def test_invalid_uploads_rejected(self):
        small_png = encode(Image.new('RGB', (10, 10), 'white'), 'PNG')
        self.assertUserError(422, pipeline.prepare_upload, b'', (0, 0, 1, 1))
        with mock.patch.object(pipeline, 'MAX_UPLOAD_BYTES', len(small_png) - 1):
            self.assertUserError(413, pipeline.prepare_upload, small_png, (0, 0, 1, 1))
        with mock.patch.object(pipeline, 'MAX_UPLOAD_BYTES', len(small_png)):
            pipeline.prepare_upload(small_png, (0, 0, 1, 1))  # exactly the limit is allowed
        other_formats = {'GIF': encode(Image.new('RGB', (10, 10)), 'GIF'),
                         'BMP': encode(Image.new('RGB', (10, 10)), 'BMP'),
                         'TIFF': encode(Image.new('RGB', (10, 10)), 'TIFF')}
        if features.check('webp'):
            other_formats['WEBP'] = encode(Image.new('RGB', (10, 10)), 'WEBP')
        for name, data in other_formats.items():
            with self.subTest(fmt=name):
                self.assertUserError(422, pipeline.prepare_upload, data, (0, 0, 1, 1))
        self.assertUserError(422, pipeline.prepare_upload, b'this is not an image at all' * 10, (0, 0, 1, 1))
        with mock.patch.object(pipeline, 'MAX_PIXELS', 99):
            error = self.assertUserError(422, pipeline.prepare_upload, small_png, (0, 0, 1, 1))
            self.assertIn('pixels', error.message)
        with mock.patch.object(pipeline, 'MAX_PIXELS', 100):
            pipeline.prepare_upload(small_png, (0, 0, 1, 1))
        jpeg = encode(patterned((64, 64)), 'JPEG', quality=95)
        self.assertUserError(422, pipeline.prepare_upload, jpeg[:len(jpeg) // 2], (0, 0, 1, 1))
        for box in ((0, 0, 0, 1), (-0.1, 0, 1, 1), (0, 0, 1.1, 1), (0.5, 0.5, 0.5001, 0.5001),
                    (0.6, 0, 0.4, 1), (0, 0, 0.1, 0.1)):
            with self.subTest(box=box):
                self.assertUserError(422, pipeline.prepare_upload, small_png, box)

    def test_parse_crop_box(self):
        self.assertEqual(pipeline.parse_crop_box('0.1', ' 0.2 ', '0.9', '1'), (0.1, 0.2, 0.9, 1.0))
        self.assertTrue(all(type(v) is float for v in pipeline.parse_crop_box('0', '0', '1', '1')))
        for bad in ('a', 'nan', 'inf', '-inf', '1e400', None, ''):
            with self.subTest(value=bad):
                self.assertUserError(422, pipeline.parse_crop_box, bad, '0', '1', '1')
                self.assertUserError(422, pipeline.parse_crop_box, '0', '0', '1', bad)


# Settings and storage paths --------------------------------------------------
class SettingsTests(unittest.TestCase):
    ENV_KEYS = ('SERVING_ENDPOINT', 'DATABRICKS_WAREHOUSE_ID', 'TAILSPOT_SCHEMA', 'TAILSPOT_VOLUME_PATH')

    def test_invalid_schema_and_volume_rejected(self):
        for schema in ('x;drop', 'a.b.c', '', 'a', 'a.`b`', 'a. b'):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                pipeline.Settings(schema=schema)
        for path in ('/Volumes/a/b', '/tmp/x', '/Volumes/a/b/c/../d', '/Volumes/a/b/c/', 'Volumes/a/b/c',
                     '/Volumes/a/b/c d'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                pipeline.Settings(volume_path=path)
        with self.assertRaises(ValueError):
            pipeline.Settings(serving_endpoint='')

    def test_trailing_newline_is_rejected(self):
        # BUG: `$` in _SCHEMA_PATTERN/_VOLUME_PATTERN also matches before a final "\n".
        for kwargs in ({'schema': 'dbx_ai_app.airplane_tail_id\n'},
                       {'volume_path': '/Volumes/dbx_ai_app/airplane_tail_id/submitted_photos\n'}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                pipeline.Settings(**kwargs)

    def test_table_name_is_backtick_quoted(self):
        self.assertEqual(pipeline.Settings().table('demo_airlines'),
                         '`dbx_ai_app`.`airplane_tail_id`.`demo_airlines`')
        self.assertEqual(pipeline.Settings(schema='c1.s_2').table('t'), '`c1`.`s_2`.`t`')

    def test_from_env_reads_values_and_falls_back_on_empty(self):
        values = {'SERVING_ENDPOINT': 'other-endpoint', 'DATABRICKS_WAREHOUSE_ID': 'abc123',
                  'TAILSPOT_SCHEMA': 'cat.sch', 'TAILSPOT_VOLUME_PATH': '/Volumes/cat/sch/vol'}
        with mock.patch.dict(os.environ, values, clear=False):
            settings = pipeline.Settings.from_env()
        self.assertEqual(settings, pipeline.Settings(serving_endpoint='other-endpoint', warehouse_id='abc123',
                                                     schema='cat.sch', volume_path='/Volumes/cat/sch/vol'))
        defaults = pipeline.Settings()
        with mock.patch.dict(os.environ, {key: '' for key in self.ENV_KEYS}, clear=False):
            self.assertEqual(pipeline.Settings.from_env(), defaults)
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in self.ENV_KEYS:
                os.environ.pop(key, None)
            self.assertEqual(pipeline.Settings.from_env(), defaults)
        self.assertIsNone(defaults.warehouse_id)
        self.assertEqual(defaults.serving_endpoint, 'airplane-tail-identifier')
        with mock.patch.dict(os.environ, {'TAILSPOT_SCHEMA': 'bad;schema'}, clear=False):
            with self.assertRaises(ValueError):
                pipeline.Settings.from_env()

    def test_storage_paths_use_utc_date(self):
        original, crop = pipeline.storage_paths(pipeline.Settings(), 'abc',
                                                pacific(2026, 10, 27, 23, 30))
        folder = '/Volumes/dbx_ai_app/airplane_tail_id/submitted_photos/2026/10/28/abc'
        self.assertEqual((original, crop), (f'{folder}/original.png', f'{folder}/tail_crop.png'))
        original, _ = pipeline.storage_paths(pipeline.Settings(), 'x', utc(2026, 1, 2, 0, 0))
        self.assertIn('/2026/01/02/x/', original)


# Repository consistency ------------------------------------------------------
class RepositoryConsistencyTests(unittest.TestCase):
    def test_app_serving_client_is_byte_identical_copy(self):
        self.assertEqual((ROOT / 'app' / 'serving_client.py').read_bytes(),
                         (ROOT / 'serving' / 'client.py').read_bytes())

    def test_submission_columns_match_ddl(self):
        lines = (ROOT / 'app_setup' / 'create_storage.sql').read_text(encoding='utf-8').splitlines()
        start = next(i for i, line in enumerate(lines)
                     if 'CREATE TABLE' in line and 'photo_submissions' in line and line.rstrip().endswith('('))
        columns, not_null = [], set()
        for line in lines[start + 1:]:
            if line.lstrip().startswith(')'):
                break
            match = re.match(r'^\s*([a-z_][a-z0-9_]*)\s+([A-Z]+)\b(.*)$', line)
            self.assertIsNotNone(match, line)
            name, sql_type, rest = match.groups()
            columns.append((name, sql_type))
            if re.search(r'\bNOT NULL\b', rest.split('COMMENT')[0]):
                not_null.add(name)
        self.assertEqual(columns, list(pipeline.SUBMISSION_COLUMNS))
        self.assertEqual(not_null, {
            'submission_id', 'submitted_at', 'status', 'airport_iata', 'observed_local', 'observed_at',
            'upload_format', 'upload_bytes', 'upload_sha256', 'original_path', 'original_sha256',
            'original_width', 'original_height', 'crop_left', 'crop_top', 'crop_right', 'crop_bottom',
            'crop_path', 'crop_sha256', 'crop_width', 'crop_height', 'crop_downscaled',
            'serving_endpoint', 'inference_attempts'})

    def test_app_yaml_matches_settings(self):
        config = yaml.safe_load((ROOT / 'app' / 'app.yaml').read_text(encoding='utf-8'))
        self.assertEqual(config['command'], ['python', 'main.py'])
        env = {item['name']: item for item in config['env']}
        self.assertEqual(set(env), {'SERVING_ENDPOINT', 'DATABRICKS_WAREHOUSE_ID', 'TAILSPOT_SCHEMA',
                                    'TAILSPOT_VOLUME_PATH', 'TAILSPOT_TRAINING_EXPERIMENT_ID'})
        self.assertEqual(env['SERVING_ENDPOINT'].get('valueFrom'), 'serving-endpoint')
        self.assertEqual(env['DATABRICKS_WAREHOUSE_ID'].get('valueFrom'), 'sql-warehouse')
        self.assertEqual(env['TAILSPOT_TRAINING_EXPERIMENT_ID'].get('valueFrom'), 'training-experiment')
        defaults = pipeline.Settings()
        self.assertEqual(env['TAILSPOT_SCHEMA'].get('value'), defaults.schema)
        self.assertEqual(env['TAILSPOT_VOLUME_PATH'].get('value'), defaults.volume_path)

    def test_bundle_matches_app_yaml_and_storage(self):
        bundle = yaml.safe_load((ROOT / 'databricks.yml').read_text(encoding='utf-8'))
        self.assertEqual(bundle['sync']['paths'], ['./app'])
        app = bundle['resources']['apps']['tailspot']
        self.assertEqual(app['name'], 'tailspot')
        self.assertEqual(app['source_code_path'], './app')
        resources = {item['name']: item for item in app['resources']}
        self.assertEqual(len(resources), len(app['resources']), 'resource names must be unique')
        self.assertEqual(resources['serving-endpoint']['serving_endpoint'],
                         {'name': pipeline.Settings().serving_endpoint, 'permission': 'CAN_QUERY'})
        self.assertEqual(resources['sql-warehouse']['sql_warehouse']['permission'], 'CAN_USE')

        app_yaml = yaml.safe_load((ROOT / 'app' / 'app.yaml').read_text(encoding='utf-8'))
        value_from = {item['valueFrom'] for item in app_yaml['env'] if 'valueFrom' in item}
        self.assertTrue(value_from <= set(resources), value_from - set(resources))

        securables = {}
        for item in app['resources']:
            if 'uc_securable' in item:
                spec = item['uc_securable']
                entry = securables.setdefault(spec['securable_full_name'], (spec['securable_type'], set()))
                self.assertEqual(entry[0], spec['securable_type'])
                entry[1].add(spec['permission'])

        def effective(permissions):
            # Databricks Apps: a table resource with MODIFY implicitly includes SELECT
            # (docs.databricks.com/aws/en/dev-tools/databricks-apps/tables).
            return permissions | ({'SELECT'} if 'MODIFY' in permissions else set())

        schema = pipeline.Settings().schema
        volume = '.'.join(pipeline.Settings().volume_path.split('/')[2:])
        self.assertEqual(securables[volume][0], 'VOLUME')
        self.assertIn('WRITE_VOLUME', securables[volume][1])
        for table in ('demo_airlines', 'demo_weekly_departures'):
            kind, permissions = securables[f'{schema}.{table}']
            self.assertEqual(kind, 'TABLE')
            self.assertIn('SELECT', effective(permissions))
        kind, permissions = securables[f'{schema}.photo_submissions']
        self.assertEqual(kind, 'TABLE')
        self.assertTrue({'MODIFY', 'SELECT'} <= effective(permissions), permissions)

    def test_files_referenced_by_bundle_comments_exist(self):
        text = (ROOT / 'databricks.yml').read_text(encoding='utf-8')
        comments = ' '.join(line.split('#', 1)[1] for line in text.splitlines() if '#' in line)
        referenced = set(re.findall(r'\b[\w-]+(?:/[\w.-]+)+\.(?:sql|md|py|yml|yaml)\b', comments))
        self.assertTrue(referenced)
        missing = sorted(path for path in referenced if not (ROOT / path).is_file())
        self.assertEqual(missing, [])

    def test_requirements_are_pinned(self):
        lines = [line.split('#', 1)[0].strip()
                 for line in (ROOT / 'app' / 'requirements.txt').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(sorted(line for line in lines if line), sorted([
            'fastapi==0.162.0', 'uvicorn==0.54.0', 'python-multipart==0.0.32',
            'databricks-sdk==0.157.0', 'pillow==13.0.0', 'tzdata==2026.5',
            'opentelemetry-sdk==1.45.1']))

    def test_constants_match_api_contract(self):
        api = (ROOT / 'app' / 'API.md').read_text(encoding='utf-8')
        config = json.loads(re.search(r'## GET /api/config.*?```json\n(.*?)```', api, re.S).group(1))
        self.assertEqual(config['max_upload_bytes'], pipeline.MAX_UPLOAD_BYTES)
        self.assertEqual(config['max_upload_bytes'], 20 * 1024 * 1024)
        self.assertEqual(config['window_minutes_before'], schedule.WINDOW_BEFORE_MIN)
        self.assertEqual(config['window_minutes_after'], schedule.WINDOW_AFTER_MIN)
        self.assertEqual(config['airport_iata'], schedule.AIRPORT_IATA)
        self.assertEqual(config['airport_name'], schedule.AIRPORT_NAME)
        self.assertEqual(config['time_zone'], schedule.TIME_ZONE)
        self.assertIn('40 megapixels', api)
        self.assertEqual(pipeline.MAX_PIXELS, 40_000_000)
        self.assertIn('1536 px', api)
        self.assertEqual(pipeline.CROP_MAX_SIDE, 1536)
        self.assertEqual(schedule.MAX_FUTURE, timedelta(minutes=10))


if __name__ == '__main__':
    unittest.main()
