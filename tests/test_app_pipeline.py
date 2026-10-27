"""Background-processing tests for app/pipeline.py using fakes (no network, no Databricks)."""
import csv
from datetime import datetime, timezone
import io
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from PIL import Image
from databricks.sdk import errors
from databricks.sdk.service.sql import StatementState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))

import pipeline  # noqa: E402

pipeline.log.disabled = True  # failures are logged with tracebacks by design
UTC = timezone.utc
ENDPOINT = 'airplane-tail-identifier'
COLS = ('airline_name', 'iata_code', 'departure_id', 'flight_number', 'destination_iata',
        'destination_city', 'day_of_week', 'scheduled_departure_local')
PROBS = {'alaska': 0.025, 'american': 0.025, 'delta': 0.025, 'southwest': 0.9, 'united': 0.025}


def read_csv(name):
    with open(ROOT / 'schedule' / name, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


AIRLINES = {row['airline_id']: row for row in read_csv('airlines.csv')}
DEPARTURES = read_csv('departures.csv')


def now():
    return datetime(2026, 10, 27, 15, 25, tzinfo=UTC)


def resp(state=StatementState.SUCCEEDED, rows=None, cols=(), msg=None, sid='s1'):
    return NS(statement_id=sid, status=NS(state=state, error=NS(message=msg) if msg else None),
              manifest=NS(schema=NS(columns=[NS(name=c) for c in cols])),
              result=None if rows is None else NS(data_array=rows, next_chunk_index=None))


def prediction(airline='southwest', confidence=0.9):
    return {'airline': airline, 'confidence': confidence, 'probabilities_json': json.dumps(PROBS)}


class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


class SyncExecutor:
    def submit(self, fn, *args):
        fn(*args)

    def shutdown(self, **kwargs):
        pass


class ManualExecutor:
    def __init__(self):
        self.jobs, self.shutdown_kwargs = [], None

    def submit(self, fn, *args):
        self.jobs.append((fn, args))

    def run_all(self):
        while self.jobs:
            fn, args = self.jobs.pop(0)
            fn(*args)

    def shutdown(self, **kwargs):
        self.shutdown_kwargs = kwargs


class FakeWorkspace:
    """Records statements; scripted schedule, MERGE and serving outcomes."""

    def __init__(self, clock):
        self.clock, self.uploads, self.deletes, self.statements = clock, [], [], []
        self.schedule_fail, self.merge_script, self.get_error = False, [], None
        self.serving, self.query_cost, self.query_starts, self.on_query = iter([]), 0, [], None
        self.polls, self.cancelled, self.chunks = [], [], {}
        self.files = NS(upload=lambda path, data, overwrite=False: self.uploads.append(path),
                        delete=self.deletes.append)
        self.statement_execution = NS(
            execute_statement=self.execute_statement, cancel_execution=self.cancelled.append,
            get_statement=lambda sid: self.polls.pop(0),
            get_statement_result_chunk_n=lambda sid, idx: self.chunks[idx])
        self.serving_endpoints = NS(query=self.query, get=self.get)

    def script(self, *items):
        self.serving = iter(items)

    def execute_statement(self, statement, warehouse_id, wait_timeout, parameters):
        params = {p.name: p.value for p in parameters}
        self.statements.append((statement, params))
        if 'demo_weekly_departures' in statement:
            if self.schedule_fail:
                return resp(StatementState.FAILED, msg='boom')
            airline = AIRLINES.get(params['airline'])
            if airline is None:  # LEFT JOIN from demo_airlines: unknown airline -> no rows
                return resp(rows=[], cols=COLS)
            deps = [d for d in DEPARTURES if d['airline_id'] == params['airline']
                    and d['origin_iata'] == params['origin']] or [dict.fromkeys(COLS)]
            return resp(rows=[[airline['airline_name'], airline['iata_code']]
                              + [d[c] for c in COLS[2:]] for d in deps], cols=COLS)
        if statement.startswith('MERGE INTO'):
            outcome = self.merge_script.pop(0) if self.merge_script else 'ok'
            if outcome == 'fail':
                return resp(StatementState.FAILED, msg='boom')
            if outcome == 'concurrent':
                return resp(StatementState.FAILED,
                            msg='[DELTA_CONCURRENT_APPEND] ConcurrentAppendException')
            return resp(rows=[])
        return resp(rows=[], cols=('submission_id',))

    def query(self, name, dataframe_records):
        self.query_starts.append(self.clock.t)
        if self.on_query:
            self.on_query()
        self.clock.t += self.query_cost
        item = next(self.serving, 'default')
        if isinstance(item, BaseException):
            raise item
        item = prediction() if item == 'default' else item
        return NS(predictions=item if isinstance(item, list) else [item],
                  served_model_name='airplane-tail-v1')

    def get(self, name):
        if self.get_error:
            raise self.get_error
        return NS(config=NS(served_entities=[NS(
            name='airplane-tail-v1', entity_name='dbx_ai_app.airplane_tail_id.airplane_tail_identifier',
            entity_version='1')]))


def make(executor=None):
    clock = Clock()
    fake = FakeWorkspace(clock)
    service = pipeline.SubmissionService(
        pipeline.Settings(warehouse_id='wh'), client_factory=lambda purpose: fake,
        executor=executor or SyncExecutor(), sleep=clock.sleep, monotonic=clock.monotonic, now=now)
    return service, fake, clock


def png(w=40, h=30):
    buffer = io.BytesIO()
    Image.new('RGB', (w, h), (200, 30, 30)).save(buffer, 'PNG')
    return buffer.getvalue()


def submit(service, observed='2026-10-27T08:20'):
    return service.submit(png(), (0.1, 0.1, 0.9, 0.9), observed)['submission_id']


def merges(fake):
    return [params for statement, params in fake.statements if statement.startswith('MERGE INTO')]


class InferenceTests(unittest.TestCase):
    def test_cold_start_retries_then_succeeds(self):
        service, fake, clock = make()
        cold = errors.TemporarilyUnavailable('cold')
        fake.script(cold, cold, 'default')
        seen = []
        fake.on_query = lambda: seen.append(service.get(next(reversed(service._jobs))))
        job = service.get(submit(service))
        self.assertEqual((job['status'], job['inference_attempts'], job['recorded'], job['error']),
                         ('succeeded', 3, True, None))
        self.assertEqual(clock.sleeps, [5, 10])
        self.assertEqual([s['message'] for s in seen],
                         [pipeline.FIRST_ATTEMPT_MESSAGE] + [pipeline.RETRY_MESSAGE] * 2)
        self.assertEqual([(s['inference_attempts'], s['status'], s['stage']) for s in seen],
                         [(n, 'processing', 'waiting_for_model') for n in (1, 2, 3)])
        self.assertEqual((job['prediction']['airline_name'], job['prediction']['iata_code'],
                          job['schedule_lookup']), ('Southwest Airlines', 'WN', 'ok'))
        self.assertEqual([c['flight_number'] for c in job['candidates']], ['WN1002', 'WN1001'])
        self.assertEqual(job['model']['model_version'], '1')
        row = merges(fake)[0]
        self.assertEqual((row['inference_attempts'], row['status'], row['error'],
                          row['predicted_airline'], row['model_version'], row['candidate_count']),
                         ('3', 'succeeded', None, 'southwest', '1', '2'))
        self.assertEqual(json.loads(row['probabilities_json']), PROBS)
        self.assertEqual(json.loads(row['candidates_json']), job['candidates'])
        self.assertEqual((row['observed_local'], row['observed_at']),
                         ('2026-10-27T08:20', '2026-10-27T15:20:00+00:00'))
        self.assertEqual(service._active, 0)

    def test_deadline_gives_up_and_records_failure(self):
        service, fake, clock = make()
        fake.serving = itertools.repeat(errors.TemporarilyUnavailable('cold'))
        job = service.get(submit(service))
        self.assertEqual((job['status'], job['error'], job['message'], job['prediction'],
                          job['schedule_lookup'], job['recorded']),
                         ('failed', pipeline.TIMEOUT_MESSAGE, pipeline.TIMEOUT_MESSAGE, None,
                          'skipped', True))
        self.assertEqual(job['model'], {'endpoint': ENDPOINT, 'served_entity': None,
                                        'model_name': None, 'model_version': None})
        self.assertEqual(job['inference_attempts'], len(fake.query_starts))
        rows = merges(fake)
        self.assertEqual([r['status'] for r in rows], ['failed'])
        for key in ('predicted_airline', 'confidence', 'probabilities_json', 'candidate_count',
                    'candidates_json', 'model_version'):
            self.assertIsNone(rows[0][key], key)
        float(rows[0]['inference_seconds'])
        self.assertLessEqual(sum(clock.sleeps) + pipeline.MIN_ATTEMPT_WINDOW_S,
                             pipeline.INFERENCE_DEADLINE_S)
        self.assertFalse([s for s, _ in fake.statements if 'demo_weekly_departures' in s])
        self.assertEqual(service._active, 0)

    def test_deadline_with_slow_attempts(self):
        service, fake, clock = make()
        fake.serving = itertools.repeat(errors.TemporarilyUnavailable('cold'))
        fake.query_cost = 150
        job = service.get(submit(service))
        self.assertEqual(job['error'], pipeline.TIMEOUT_MESSAGE)
        self.assertGreater(len(fake.query_starts), 1)
        self.assertLess(fake.query_starts[-1],
                        pipeline.INFERENCE_DEADLINE_S - pipeline.MIN_ATTEMPT_WINDOW_S)
        self.assertLessEqual(clock.t, pipeline.INFERENCE_DEADLINE_S + 150)

    def test_non_retryable_failures(self):
        failed, unavailable = pipeline.MODEL_FAILED_MESSAGE, pipeline.MODEL_UNAVAILABLE_MESSAGE
        cases = [(errors.BadRequest('x'), failed), (errors.PermissionDenied('x'), unavailable),
                 (errors.NotFound('x'), unavailable), (errors.Unauthenticated('x'), unavailable),
                 ({'airline': 'x', 'confidence': 'high', 'probabilities_json': '{}'}, failed),
                 (dict(prediction(), probabilities_json='not json'), failed),
                 (dict(prediction(), probabilities_json='[1,2]'), failed),
                 (prediction(airline=''), failed), (prediction(confidence=float('nan')), failed),
                 (prediction(confidence=True), failed), ({'airline': 'southwest'}, failed),
                 ([], failed)]
        for item, message in cases:
            with self.subTest(item=repr(item)):
                service, fake, clock = make()
                fake.script(item)
                job = service.get(submit(service))
                self.assertEqual((job['status'], job['error'], job['message'],
                                  job['inference_attempts'], len(fake.query_starts), clock.sleeps),
                                 ('failed', message, message, 1, 1, []))
                self.assertEqual(merges(fake)[0]['status'], 'failed')

    def test_retryable_generic_errors(self):
        for exc in (ConnectionError('x'), TimeoutError('x'), errors.InternalError('x'),
                    errors.ResourceExhausted('x'), errors.DeadlineExceeded('x')):
            with self.subTest(exc=type(exc).__name__):
                service, fake, clock = make()
                fake.script(exc, 'default')
                job = service.get(submit(service))
                self.assertEqual((job['status'], job['inference_attempts'], clock.sleeps),
                                 ('succeeded', 2, [5]))

    def test_metadata_failure_still_succeeds(self):
        service, fake, _ = make()
        fake.get_error = errors.PermissionDenied('no')
        job = service.get(submit(service))
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(job['model'], {'endpoint': ENDPOINT, 'served_entity': 'airplane-tail-v1',
                                        'model_name': None, 'model_version': None})
        self.assertEqual((merges(fake)[0]['model_version'], merges(fake)[0]['served_entity']),
                         (None, 'airplane-tail-v1'))


class ScheduleTests(unittest.TestCase):
    def test_schedule_failure(self):
        service, fake, _ = make()
        fake.schedule_fail = True
        job = service.get(submit(service))
        self.assertEqual((job['status'], job['schedule_lookup'], job['candidates'], job['error'],
                          job['message'], job['recorded']),
                         ('succeeded', 'failed', [], pipeline.SCHEDULE_FAILED_MESSAGE, 'Done.', True))
        row = merges(fake)[0]
        self.assertEqual((row['error'], row['candidate_count'], row['candidates_json'],
                          row['predicted_airline']),
                         (pipeline.SCHEDULE_FAILED_MESSAGE, None, None, 'southwest'))
        self.assertEqual(service._active, 0)

    def test_unknown_airline(self):
        service, fake, _ = make()
        fake.script(prediction('klm'))
        job = service.get(submit(service))
        self.assertEqual((job['status'], job['schedule_lookup'], job['candidates']),
                         ('succeeded', 'ok', []))
        self.assertEqual((job['prediction']['airline_name'], job['prediction']['iata_code']),
                         (None, None))
        self.assertEqual((merges(fake)[0]['candidate_count'], merges(fake)[0]['candidates_json']),
                         ('0', '[]'))

    def test_known_airline_no_departures_in_window(self):
        service, fake, _ = make()
        job = service.get(submit(service, observed='2026-10-27T03:00'))
        self.assertEqual((job['status'], job['schedule_lookup'], job['candidates'],
                          job['prediction']['iata_code']), ('succeeded', 'ok', [], 'WN'))
        self.assertEqual(merges(fake)[0]['candidate_count'], '0')


class RecordingAndLifecycleTests(unittest.TestCase):
    def test_merge_failure_not_recorded(self):
        service, fake, _ = make()
        fake.merge_script = ['fail']
        job = service.get(submit(service))
        self.assertEqual((job['status'], job['recorded'], job['error']),
                         ('succeeded', False, pipeline.NOT_RECORDED_MESSAGE))
        self.assertEqual(len(merges(fake)), 1)  # non-concurrent failures are not retried
        self.assertEqual(service._active, 0)

        service, fake, _ = make()
        fake.schedule_fail, fake.merge_script = True, ['fail']
        job = service.get(submit(service))
        self.assertEqual((job['recorded'], job['error']), (False, pipeline.SCHEDULE_FAILED_MESSAGE
                                                           + ' ' + pipeline.NOT_RECORDED_MESSAGE))

    def test_concurrent_merge_retried(self):
        service, fake, clock = make()
        fake.merge_script = ['concurrent', 'ok']
        job = service.get(submit(service))
        self.assertTrue(job['recorded'])
        self.assertEqual(len(merges(fake)), 2)
        self.assertIn(2, clock.sleeps)
        self.assertEqual(merges(fake)[0], merges(fake)[1])  # same idempotent statement

        service, fake, _ = make()
        fake.merge_script = ['concurrent'] * 3
        job = service.get(submit(service))
        self.assertEqual((job['recorded'], len(merges(fake)), job['error']),
                         (False, 3, pipeline.NOT_RECORDED_MESSAGE))

    def test_unexpected_exception(self):
        service, fake, _ = make()
        with mock.patch.object(pipeline, 'infer', side_effect=RuntimeError('bug')):
            sid = submit(service)
        job = service.get(sid)
        self.assertEqual((job['status'], job['stage'], job['message'], job['recorded']),
                         ('failed', 'done', pipeline.UNEXPECTED_MESSAGE, True))
        self.assertTrue(job['error'].startswith(pipeline.UNEXPECTED_MESSAGE))
        rows = merges(fake)
        self.assertEqual([(r['status'], r['error'], r['serving_endpoint']) for r in rows],
                         [('failed', pipeline.UNEXPECTED_MESSAGE, ENDPOINT)])
        self.assertEqual(service._active, 0)
        self.assertIsNone(service._jobs[sid]['private'])

        service, fake, _ = make()
        fake.merge_script = ['fail']
        with mock.patch.object(pipeline, 'infer', side_effect=RuntimeError('bug')):
            job = service.get(submit(service))
        self.assertEqual((job['recorded'], job['error']),
                         (False, pipeline.UNEXPECTED_MESSAGE + ' ' + pipeline.NOT_RECORDED_MESSAGE))

    def test_failure_after_recording_reports_recorded(self):
        # The row was already written; an exception afterwards must not claim it was not.
        service, fake, _ = make()
        real_update = service._update

        def update(sid, **changes):
            if changes.get('stage') == 'done' and changes.get('status') == 'succeeded':
                raise RuntimeError('bug')
            real_update(sid, **changes)
        service._update = update
        job = service.get(submit(service))
        self.assertEqual(len(merges(fake)), 1)  # not recorded twice
        self.assertEqual((job['status'], job['recorded'], job['error']),
                         ('failed', True, pipeline.UNEXPECTED_MESSAGE))

    def test_active_accounting_with_queued_jobs(self):
        executor = ManualExecutor()
        service, fake, _ = make(executor)
        sid = submit(service)
        self.assertEqual(service._active, 1)
        job = service.get(sid)
        self.assertEqual((job['status'], job['stage'], job['recorded']), ('queued', 'queued', False))
        self.assertEqual((len(fake.uploads), fake.query_starts), (2, []))
        executor.run_all()
        self.assertEqual((service._active, service.get(sid)['status']), (0, 'succeeded'))

        for _ in range(pipeline.MAX_IN_FLIGHT):
            submit(service)
        with self.assertRaises(pipeline.UserError) as ctx:
            submit(service)
        self.assertEqual((ctx.exception.status_code, service._active), (503, pipeline.MAX_IN_FLIGHT))

    def test_shutdown_fails_queued_jobs(self):
        executor = ManualExecutor()
        service, _, _ = make(executor)
        done = submit(service)
        executor.run_all()
        sid = submit(service)
        service.shutdown()
        self.assertEqual(executor.shutdown_kwargs.get('cancel_futures'), True)
        job = service.get(sid)
        self.assertEqual((job['status'], job['stage'], job['message'], job['error']),
                         ('failed', 'done', pipeline.RESTARTED_MESSAGE, pipeline.RESTARTED_MESSAGE))
        self.assertEqual(service.get(done)['status'], 'succeeded')

    def test_eviction_keeps_newest(self):
        service, _, _ = make()
        with mock.patch.object(pipeline, 'STORE_LIMIT', 3):
            ids = [submit(service) for _ in range(5)]
            self.assertEqual(list(service._jobs), ids[-3:])

    def test_get_returns_deep_copy(self):
        service, _, _ = make()
        sid = submit(service)
        job = service.get(sid)
        job['prediction']['probabilities']['southwest'] = -1
        job['candidates'].clear()
        job['model']['endpoint'] = 'x'
        fresh = service.get(sid)
        self.assertEqual((fresh['prediction']['probabilities']['southwest'],
                          len(fresh['candidates']), fresh['model']['endpoint']), (0.9, 2, ENDPOINT))


class SqlHelperTests(unittest.TestCase):
    def sql_fake(self, first, polls=(), chunks=None):
        calls, polls = NS(executed=[], cancelled=[]), list(polls)
        api = NS(execute_statement=lambda **kw: calls.executed.append(kw) or first,
                 get_statement=lambda sid: polls.pop(0), cancel_execution=calls.cancelled.append,
                 get_statement_result_chunk_n=lambda sid, idx: (chunks or {})[idx])
        return NS(statement_execution=api), calls

    def test_polls_until_succeeded(self):
        w, _ = self.sql_fake(resp(StatementState.PENDING), [
            resp(StatementState.RUNNING), resp(rows=[['1', 'a']], cols=('n', 's'))])
        clock = Clock()
        rows = pipeline.execute_sql(w, 'wh', 'SELECT 1', sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual((rows, clock.sleeps), ([{'n': '1', 's': 'a'}], [2, 2]))

    def test_timeout_cancels(self):
        w, calls = self.sql_fake(resp(StatementState.PENDING),
                                 [resp(StatementState.RUNNING) for _ in range(10)])
        clock = Clock()
        with self.assertRaises(TimeoutError):
            pipeline.execute_sql(w, 'wh', 'SELECT 1', max_wait_s=5, sleep=clock.sleep,
                                 monotonic=clock.monotonic)
        self.assertEqual(calls.cancelled, ['s1'])
        self.assertLessEqual(sum(clock.sleeps), 6)

    def test_failed_and_missing_warehouse_and_empty_result(self):
        w, _ = self.sql_fake(resp(StatementState.FAILED, msg='syntax oops'))
        with self.assertRaisesRegex(RuntimeError, 'FAILED: syntax oops'):
            pipeline.execute_sql(w, 'wh', 'SELECT 1')
        w, calls = self.sql_fake(resp(rows=[]))
        for warehouse in (None, ''):
            with self.assertRaises(RuntimeError):
                pipeline.execute_sql(w, warehouse, 'SELECT 1')
        self.assertEqual(calls.executed, [])
        w, _ = self.sql_fake(resp(rows=None, cols=('n',)))
        self.assertEqual(pipeline.execute_sql(w, 'wh', 'SELECT 1'), [])

    def test_chunks_and_parameters(self):
        first = resp(rows=[['1']], cols=('n',))
        first.result.next_chunk_index = 1
        w, calls = self.sql_fake(first, chunks={1: NS(data_array=[['2'], ['3']], next_chunk_index=2),
                                                2: NS(data_array=None, next_chunk_index=None)})
        rows = pipeline.execute_sql(w, 'wh', 'SELECT :a, :b', {'a': None, 'b': 5})
        self.assertEqual(rows, [{'n': '1'}, {'n': '2'}, {'n': '3'}])
        params = calls.executed[0]['parameters']
        self.assertEqual([(p.name, p.value) for p in params], [('a', None), ('b', '5')])
        self.assertEqual(calls.executed[0]['warehouse_id'], 'wh')

    @staticmethod
    def full_values(**overrides):
        return {**dict.fromkeys(name for name, _ in pipeline.SUBMISSION_COLUMNS), **overrides}

    def test_insert_submission_rejects_wrong_columns(self):
        fake, settings = FakeWorkspace(Clock()), pipeline.Settings(warehouse_id='wh')
        missing = self.full_values()
        del missing['reviewed_at']
        for values in (missing, self.full_values(extra='x')):
            with self.assertRaises(ValueError):
                pipeline.insert_submission(fake, settings, values)
        self.assertEqual(fake.statements, [])

    def test_insert_submission_formats(self):
        fake = FakeWorkspace(Clock())
        moment = datetime(2026, 10, 27, 8, 25, tzinfo=pipeline.schedule.PACIFIC)
        pipeline.insert_submission(fake, pipeline.Settings(warehouse_id='wh'), self.full_values(
            submitted_at=moment, crop_downscaled=True, crop_left=0.1, crop_width=5,
            upload_bytes=7, status='zz-status'))
        statement, params = fake.statements[0]
        self.assertEqual(params['submitted_at'], '2026-10-27T15:25:00+00:00')
        self.assertEqual((params['crop_downscaled'], params['crop_left'], params['crop_width'],
                          params['upload_bytes'], params['status']), ('true', '0.1', '5', '7', 'zz-status'))
        self.assertEqual((params['completed_at'], params['confidence']), (None, None))
        self.assertIn('`dbx_ai_app`.`airplane_tail_id`.`photo_submissions`', statement)
        self.assertNotIn('zz-status', statement)

    def test_lookup_schedule_uses_parameters(self):
        fake, settings = FakeWorkspace(Clock()), pipeline.Settings(warehouse_id='wh')
        observed = datetime(2026, 10, 27, 15, 20, tzinfo=UTC)
        name, code, candidates = pipeline.lookup_schedule(fake, settings, 'southwest', observed)
        self.assertEqual((name, code), ('Southwest Airlines', 'WN'))
        self.assertEqual([c['flight_number'] for c in candidates], ['WN1002', 'WN1001'])
        statement, params = fake.statements[0]
        self.assertEqual(params, {'origin': 'SAN', 'airline': 'southwest'})
        self.assertNotIn('southwest', statement.lower())
        self.assertNotIn('SAN', statement)
        self.assertEqual(pipeline.lookup_schedule(fake, settings, 'klm', observed), (None, None, []))


if __name__ == '__main__':
    unittest.main()
