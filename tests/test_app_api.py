"""HTTP-layer tests for app/main.py with a fake WorkspaceClient, executor and clock (no network)."""
import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import uuid
import warnings

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
warnings.filterwarnings('ignore', message='Using `httpx`')

from databricks.sdk.service.sql import StatementState  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
import pipeline  # noqa: E402

logging.getLogger('tailspot').disabled = True
logging.getLogger('httpx').setLevel(logging.WARNING)

NOW = datetime(2026, 10, 27, 15, 25, tzinfo=timezone.utc)
BOX = (0.1, 0.1, 0.9, 0.9)
SCHEDULE_COLUMNS = ['airline_name', 'iata_code', 'departure_id', 'flight_number',
                    'destination_iata', 'destination_city', 'day_of_week',
                    'scheduled_departure_local']


def _read_csv(name):
    with open(ROOT / 'schedule' / name, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


AIRLINES, DEPARTURES = _read_csv('airlines.csv'), _read_csv('departures.csv')


def png_bytes(size=(200, 100), fmt='PNG'):
    buffer = io.BytesIO()
    Image.new('RGB', size, (200, 30, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


class SyncExecutor:
    def submit(self, fn, *args):
        fn(*args)

    def shutdown(self, **kwargs):
        pass


class ManualExecutor(SyncExecutor):
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run_all(self):
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)


class FailingExecutor(SyncExecutor):
    def submit(self, fn, *args):
        raise RuntimeError('executor shut down')


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps, self.t = self.sleeps + [seconds], self.t + seconds


class FakeWorkspace:
    def __init__(self, fail_on=None):
        self.fail_on, self.uploads, self.deletes, self.sql, self.queries, self.cancels = \
            fail_on, [], [], [], [], []
        self.fallback_rows, self.fallback_error, self.fallback_pending = [], None, False
        self.files = NS(upload=self._upload, delete=self.deletes.append)
        self.statement_execution = NS(execute_statement=self._execute,
                                      get_statement=self._get_statement,
                                      cancel_execution=self.cancels.append)
        self.serving_endpoints = NS(query=self._query, get=self._get_endpoint)

    def _upload(self, path, fileobj, overwrite=False):
        if self.fail_on == len(self.uploads) + 1:
            self.uploads.append(None)  # count the failed attempt
            raise RuntimeError('upload failed')
        self.uploads.append((path, fileobj.read(), overwrite))

    def _response(self, rows, columns=None, state=StatementState.SUCCEEDED):
        columns = columns or (list(rows[0]) if rows else ['x'])
        return NS(statement_id='st-1', status=NS(state=state, error=None),
                  manifest=NS(schema=NS(columns=[NS(name=c) for c in columns])),
                  result=NS(data_array=[[row.get(c) for c in columns] for row in rows],
                            next_chunk_index=None))

    def _execute(self, statement, warehouse_id, wait_timeout, parameters):
        params = {p.name: p.value for p in parameters or []}
        self.sql.append((statement, params, wait_timeout))
        if 'demo_weekly_departures' in statement:
            rows = []
            for airline in (a for a in AIRLINES if a['airline_id'] == params['airline']):
                deps = [d for d in DEPARTURES if d['airline_id'] == airline['airline_id']
                        and d['origin_iata'] == params['origin']] or [{}]
                rows += [{**airline, **dep} for dep in deps]
            return self._response(rows, SCHEDULE_COLUMNS)
        if statement.startswith('MERGE INTO'):
            return self._response([])
        if 'photo_submissions` s' in statement:
            if self.fallback_error:
                raise self.fallback_error
            if self.fallback_pending:
                return self._response([], state=StatementState.PENDING)
            return self._response(self.fallback_rows)
        raise AssertionError(f'unexpected SQL: {statement}')

    def _get_statement(self, statement_id):
        return self._response([], state=StatementState.PENDING)

    def _query(self, name, dataframe_records):
        self.queries.append((name, dataframe_records))
        probs = {'alaska': 0.02, 'american': 0.03, 'delta': 0.02, 'southwest': 0.9, 'united': 0.03}
        return NS(predictions=[{'airline': 'southwest', 'confidence': 0.9,
                                'probabilities_json': json.dumps(probs)}],
                  served_model_name='airplane-tail-v1')

    def _get_endpoint(self, name):
        return NS(config=NS(served_entities=[NS(
            name='airplane-tail-v1', entity_name='dbx_ai_app.airplane_tail_id.airplane_tail_identifier',
            entity_version='1')]))

    def merges(self):
        return [s for s in self.sql if s[0].startswith('MERGE INTO')]


def make(executor=None, fake=None):
    """Purpose 'files' gets only the Files API, 'api' only SQL + serving, so misuse fails loudly."""
    fake, clock, purposes = fake or FakeWorkspace(), Clock(), []
    views = {'files': NS(files=fake.files),
             'api': NS(statement_execution=fake.statement_execution,
                       serving_endpoints=fake.serving_endpoints)}
    service = pipeline.SubmissionService(
        pipeline.Settings(warehouse_id='wh'),
        client_factory=lambda purpose: (purposes.append(purpose), views[purpose])[1],
        executor=executor or SyncExecutor(), sleep=clock.sleep, monotonic=clock.monotonic,
        now=lambda: NOW)
    service.purposes = purposes
    return TestClient(main.create_app(service)), service, fake, clock


def post(client, image_bytes, filename='p.png', content_type='image/png', **overrides):
    form = {'crop_left': '0.1', 'crop_top': '0.1', 'crop_right': '0.9', 'crop_bottom': '0.9',
            'observed_local': '2026-10-27T08:20', **overrides}
    form = {k: v for k, v in form.items() if v is not None}
    files = None if image_bytes is None else {'photo': (filename, image_bytes, content_type)}
    return client.post('/api/submissions', data=form, files=files)


class HappyPathTests(unittest.TestCase):
    def test_submit_and_status(self):
        client, service, fake, _ = make()
        data = png_bytes()
        response = post(client, data)
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        sid = body['submission_id']
        self.assertEqual(uuid.UUID(sid).version, 4)
        self.assertEqual(body, {'submission_id': sid, 'status': 'queued',
                                'status_url': f'/api/submissions/{sid}'})
        prepared = pipeline.prepare_upload(data, BOX)
        folder = f'/Volumes/dbx_ai_app/airplane_tail_id/submitted_photos/2026/10/27/{sid}'
        self.assertEqual(fake.uploads, [(f'{folder}/original.png', prepared.original_png, True),
                                        (f'{folder}/tail_crop.png', prepared.crop_png, True)])
        self.assertEqual(set(service.purposes), {'files', 'api'})
        self.assertEqual(fake.queries, [('airplane-tail-identifier', [
            {'image_base64': base64.b64encode(prepared.crop_png).decode('ascii')}])])
        schedule_sql = [s for s in fake.sql if 'demo_weekly_departures' in s[0]]
        self.assertEqual([s[1] for s in schedule_sql], [{'origin': 'SAN', 'airline': 'southwest'}])

        merges = fake.merges()
        self.assertEqual(len(merges), 1)
        statement, params, _ = merges[0]
        self.assertIn('WHEN NOT MATCHED THEN INSERT *', statement)
        self.assertEqual(list(params), [n for n, _ in pipeline.SUBMISSION_COLUMNS])
        sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
        expected = {'status': 'succeeded', 'observed_at': '2026-10-27T15:20:00+00:00',
                    'submitted_at': '2026-10-27T15:25:00+00:00',
                    'original_sha256': sha(fake.uploads[0][1]), 'crop_sha256': sha(fake.uploads[1][1]),
                    'crop_downscaled': 'false', 'upload_format': 'PNG',
                    'predicted_airline': 'southwest', 'candidate_count': '2', 'model_version': '1',
                    'inference_attempts': '1', 'review_label': None}
        self.assertEqual({k: params[k] for k in expected}, expected)
        self.assertEqual(float(params['crop_left']), 0.1)

        status = client.get(f'/api/submissions/{sid}')
        self.assertEqual((status.status_code, status.headers['cache-control']), (200, 'no-store'))
        s = status.json()
        self.assertEqual((s['status'], s['stage'], s['recorded'], s['error'], s['schedule_lookup'],
                          s['prediction']['airline_name'], s['prediction']['iata_code']),
                         ('succeeded', 'done', True, None, 'ok', 'Southwest Airlines', 'WN'))
        self.assertEqual(json.loads(params['candidates_json']), s['candidates'])
        self.assertEqual([(c['flight_number'], c['minutes_after_scheduled_gate_departure'])
                          for c in s['candidates']], [('WN1002', -10.0), ('WN1001', 50.0)])
        self.assertEqual(set(s['model']), {'endpoint', 'served_entity', 'model_name', 'model_version'})

    def test_no_user_values_in_sql_text(self):
        client, _, fake, _ = make()
        sid = post(client, png_bytes()).json()['submission_id']
        client.get(f'/api/submissions/{uuid.uuid4()}')  # exercises the fallback statement
        self.assertGreaterEqual(len(fake.sql), 3)
        for statement, _, _ in fake.sql:
            for value in ('southwest', '2026-10-27T08:20', sid, '/Volumes/'):
                self.assertNotIn(value, statement)

    def test_queued_status_before_processing(self):
        executor = ManualExecutor()
        client, _, _, _ = make(executor)
        sid = post(client, png_bytes()).json()['submission_id']
        s = client.get(f'/api/submissions/{sid}').json()
        self.assertEqual((s['status'], s['stage'], s['schedule_lookup'], s['model'],
                          s['prediction'], s['recorded']),
                         ('queued', 'queued', 'pending', None, None, False))
        executor.run_all()
        self.assertEqual(client.get(f'/api/submissions/{sid}').json()['status'], 'succeeded')


class SubmitErrorTests(unittest.TestCase):
    def assertDetail(self, response, status, contains=''):
        self.assertEqual(response.status_code, status, response.text)
        detail = response.json()['detail']
        self.assertIsInstance(detail, str)
        self.assertIn(contains, detail)

    def test_upload_failures_return_502(self):
        client, service, fake, _ = make(fake=FakeWorkspace(fail_on=1))
        self.assertDetail(post(client, png_bytes()), 502)
        self.assertEqual((fake.queries, fake.merges(), service._active), ([], [], 0))

        client, service, fake, _ = make(fake=FakeWorkspace(fail_on=2))
        self.assertDetail(post(client, png_bytes()), 502)
        self.assertEqual(fake.deletes, [fake.uploads[0][0]])
        self.assertTrue(fake.deletes[0].endswith('/original.png'))
        self.assertEqual((fake.queries, fake.merges(), service._active, len(service._jobs)),
                         ([], [], 0, 0))

    def test_busy_future_time_orientation_and_executor_failure(self):
        client, service, fake, _ = make()
        service._active = pipeline.MAX_IN_FLIGHT
        self.assertDetail(post(client, png_bytes()), 503)
        self.assertEqual((service._active, fake.uploads), (pipeline.MAX_IN_FLIGHT, []))

        client, service, fake, _ = make()
        self.assertDetail(post(client, png_bytes(), observed_local='2026-10-27T09:00'), 422, 'future')
        self.assertEqual((service._active, fake.uploads), (0, []))
        self.assertDetail(post(client, png_bytes(), image_width='100', image_height='200'), 422,
                          'orientation')
        self.assertEqual((service._active, fake.uploads), (0, []))
        self.assertEqual(post(client, png_bytes(), image_width='200',
                              image_height='100').status_code, 202)

        client, service, fake, _ = make(FailingExecutor())
        self.assertDetail(post(client, png_bytes()), 503)
        self.assertEqual((dict(service._jobs), service._active), ({}, 0))

    def test_validation_errors(self):
        client, _, fake, _ = make()
        self.assertDetail(post(client, None), 422, 'photo')
        self.assertDetail(post(client, png_bytes(), observed_local=None), 422, 'observed_local')
        self.assertDetail(post(client, png_bytes(), crop_left='abc'), 422)
        self.assertDetail(post(client, png_bytes(), image_width='200'), 422)
        gif = png_bytes(fmt='GIF')
        self.assertDetail(post(client, gif, 'p.gif', 'image/gif'), 422, 'JPEG or PNG')
        self.assertEqual(fake.uploads, [])

    def test_body_size_limit(self):
        client, _, fake, _ = make()
        response = client.post('/api/submissions', content=b'x' * 10,
                               headers={'content-length': str(main.MAX_REQUEST_BYTES + 1),
                                        'content-type': 'multipart/form-data; boundary=b'})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), main.TOO_LARGE)

        def chunks():
            for _ in range(22):
                yield b'\0' * (1024 * 1024)

        response = client.post('/api/submissions', content=chunks(),
                               headers={'content-type': 'multipart/form-data; boundary=b'})
        self.assertNotIn('content-length', {k.lower() for k in response.request.headers})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), main.TOO_LARGE)
        self.assertEqual(fake.uploads, [])
        self.assertEqual(post(client, png_bytes()).status_code, 202)


def fallback_row(sid, **changes):
    row = {'submission_id': sid, 'status': 'succeeded', 'error': None, 'airport_iata': 'SAN',
           'observed_local': '2026-10-27T08:20', 'submitted_epoch': '1793114700',
           'observed_epoch': '1793114400', 'original_path': '/Volumes/a/b/c/o.png',
           'crop_path': '/Volumes/a/b/c/t.png', 'crop_width': '10', 'crop_height': '12',
           'crop_downscaled': 'false', 'serving_endpoint': 'airplane-tail-identifier',
           'served_entity': 'airplane-tail-v1', 'model_name': 'm', 'model_version': '1',
           'inference_attempts': '2', 'predicted_airline': 'southwest', 'confidence': '0.9',
           'probabilities_json': '{"southwest": 0.9}', 'candidates_json': '[]',
           'airline_name': 'Southwest Airlines', 'iata_code': 'WN'}
    return {**row, **changes}


class StatusFallbackTests(unittest.TestCase):
    def fallback_sql(self, fake):
        return [s for s in fake.sql if 'photo_submissions` s' in s[0]]

    def test_unknown_id_not_found_cache_and_case(self):
        client, _, fake, clock = make()
        sid = str(uuid.uuid4())
        response = client.get(f'/api/submissions/{sid}')
        self.assertEqual((response.status_code, response.json(), response.headers['cache-control']),
                         (404, {'detail': 'Submission not found'}, 'no-store'))
        self.assertEqual([(s[1], s[2]) for s in self.fallback_sql(fake)],
                         [({'submission_id': sid}, '10s')])
        for step, expected_calls in ((pipeline.NOT_FOUND_CACHE_S - 1, 1), (2, 2)):
            clock.t += step
            self.assertEqual(client.get(f'/api/submissions/{sid}').status_code, 404)
            self.assertEqual(len(self.fallback_sql(fake)), expected_calls)

        before = len(fake.sql)
        self.assertEqual(client.get('/api/submissions/abc').status_code, 404)
        self.assertEqual(len(fake.sql), before)

        known = post(client, png_bytes()).json()['submission_id']
        before = len(fake.sql)
        upper = client.get(f'/api/submissions/{known.upper()}')
        self.assertEqual((upper.status_code, upper.json()['submission_id']), (200, known))
        self.assertEqual(len(fake.sql), before)

    def test_recorded_row_maps_to_api_shape(self):
        client, _, fake, _ = make()
        sid = str(uuid.uuid4())
        fake.fallback_rows = [fallback_row(sid)]
        s = client.get(f'/api/submissions/{sid}').json()
        expected = {
            'status': 'succeeded', 'stage': 'done', 'recorded': True, 'schedule_lookup': 'ok',
            'submitted_at': '2026-10-27T15:25:00Z', 'observed_at': '2026-10-27T15:20:00Z',
            'inference_attempts': 2, 'candidates': [], 'error': None, 'message': 'Done.',
            'storage': {'original_path': '/Volumes/a/b/c/o.png', 'crop_path': '/Volumes/a/b/c/t.png',
                        'crop_width': 10, 'crop_height': 12, 'crop_downscaled': False},
            'prediction': {'airline': 'southwest', 'airline_name': 'Southwest Airlines',
                           'iata_code': 'WN', 'confidence': 0.9, 'probabilities': {'southwest': 0.9}},
            'model': {'endpoint': 'airplane-tail-identifier', 'served_entity': 'airplane-tail-v1',
                      'model_name': 'm', 'model_version': '1'}}
        self.assertEqual({k: s[k] for k in expected}, expected)
        calls = len(fake.sql)
        self.assertEqual(client.get(f'/api/submissions/{sid}').json(), s)
        self.assertEqual(len(fake.sql), calls)

    def test_recorded_row_variants(self):
        client, _, fake, _ = make()
        skipped, failed_lookup = str(uuid.uuid4()), str(uuid.uuid4())
        fake.fallback_rows = [fallback_row(skipped, status='failed', error='timeout',
                                           predicted_airline=None, candidates_json=None)]
        s = client.get(f'/api/submissions/{skipped}').json()
        self.assertEqual((s['schedule_lookup'], s['prediction'], s['status'], s['message']),
                         ('skipped', None, 'failed', 'timeout'))
        fake.fallback_rows = [fallback_row(failed_lookup, candidates_json=None)]
        s = client.get(f'/api/submissions/{failed_lookup}').json()
        self.assertEqual((s['schedule_lookup'], s['candidates']), ('failed', []))
        self.assertEqual(s['prediction']['airline'], 'southwest')

    def test_fallback_errors_return_503(self):
        client, _, fake, _ = make()
        fake.fallback_error = RuntimeError('warehouse down')
        response = client.get(f'/api/submissions/{uuid.uuid4()}')
        self.assertEqual((response.status_code, response.headers['cache-control']), (503, 'no-store'))
        self.assertIsInstance(response.json()['detail'], str)

        client, _, fake, _ = make()
        fake.fallback_pending, sid = True, str(uuid.uuid4())
        response = client.get(f'/api/submissions/{sid}')
        self.assertEqual(response.status_code, 503)
        self.assertIsInstance(response.json()['detail'], str)
        self.assertEqual(fake.cancels, ['st-1'])
        fake.fallback_pending = False  # a transient failure is not cached as "not found"
        fake.fallback_rows = [fallback_row(sid)]
        self.assertEqual(client.get(f'/api/submissions/{sid}').status_code, 200)


class ConfigAndStaticTests(unittest.TestCase):
    def test_health_and_config(self):
        client, _, fake, _ = make()
        self.assertEqual(client.get('/api/health').json(), {'status': 'ok'})
        response = client.get('/api/config')
        self.assertEqual(response.headers['cache-control'], 'no-store')
        config = response.json()
        self.assertRegex(config.pop('now_local'), r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$')
        self.assertEqual(config, {'airport_iata': 'SAN', 'airport_name': 'San Diego International Airport',
                                  'time_zone': 'America/Los_Angeles', 'max_upload_bytes': 20971520,
                                  'window_minutes_before': 60, 'window_minutes_after': 10})
        self.assertEqual((fake.sql, fake.uploads, fake.queries), ([], [], []))

    def test_static_files_and_traversal(self):
        client, _, _, _ = make()
        index = client.get('/')
        self.assertEqual(index.status_code, 200)
        self.assertTrue(index.headers['content-type'].startswith('text/html'))
        self.assertIn('tailspot', index.text.lower())
        for path, code in (('/app.js', 200), ('/styles.css', 200), ('/api/nope', 404),
                           ('/API.md', 404), ('/main.py', 404)):
            self.assertEqual(client.get(path).status_code, code, path)
        for path in ('/../main.py', '/%2e%2e/main.py', '/..%2fmain.py', '/static/../main.py'):
            response = client.get(path)
            self.assertNotEqual(response.status_code, 200, path)
            self.assertNotIn('create_app', response.text, path)

    def test_unexpected_submit_error_returns_500(self):
        client, service, _, _ = make()
        service.submit = lambda *a, **k: {}['internal']  # raises KeyError
        response = post(client, png_bytes())
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('internal', response.json()['detail'])

    def test_status_snapshot_isolation(self):
        client, service, _, _ = make()
        sid = post(client, png_bytes()).json()['submission_id']
        snapshot = service.get(sid)
        snapshot['status'], snapshot['storage']['crop_width'] = 'tampered', -1
        snapshot['candidates'].clear()
        snapshot['prediction']['probabilities']['southwest'] = 0
        fresh = service.get(sid)
        self.assertEqual((fresh['status'], len(fresh['candidates']),
                          fresh['prediction']['probabilities']['southwest']), ('succeeded', 2, 0.9))
        self.assertGreater(fresh['storage']['crop_width'], 0)
        self.assertEqual(client.get(f'/api/submissions/{sid}').json(), fresh)


if __name__ == '__main__':
    unittest.main()
