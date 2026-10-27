"""Tests for app/performance.py and the /api/performance/* routes (fakes only, no network)."""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
warnings.filterwarnings('ignore', message='Using `httpx`')

from databricks.sdk import errors  # noqa: E402
from databricks.sdk.service.ml import (GetRunResponse, Metric, Run, RunData, RunInfo,  # noqa: E402
                                       RunInfoStatus, RunTag)
from databricks.sdk.service.sql import StatementState  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
import performance  # noqa: E402
import pipeline  # noqa: E402

logging.getLogger('tailspot').disabled = True
logging.getLogger('httpx').setLevel(logging.WARNING)

EXPERIMENT = '350719556651226'
RUN_A, RUN_B = 'a' * 32, 'b' * 32
EVAL_NEW, EVAL_OLD = 'e' * 32, 'f' * 32
EPOCH = int(datetime(2026, 10, 27, 20, 0, tzinfo=timezone.utc).timestamp())
TABLE_PREFIX = '`dbx_ai_app`.`airplane_tail_id`.'


class Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def run_row(version, **changes):
    row = {'eval_run_id': EVAL_NEW, 'created_epoch': str(EPOCH), 'model_version': version,
           'recipe': 'improved', 'dataset_version': 'v2', 'training_run_id': RUN_A,
           'val_accuracy': '0.9230769230769231', 'val_loss': '0.31', 'aliases': '["demo"]',
           'serving': 'false', 'selected': 'false'}
    return {**row, **changes}


def metric_row(version, eval_set, variant, metric='accuracy', value='1.0', n='10', run=EVAL_NEW):
    return {'eval_run_id': run, 'model_version': version, 'eval_set': eval_set, 'variant': variant,
            'metric': metric, 'value': value, 'n': n}


def prediction_row(version, item_id, eval_set='v1_test', variant='tail_crop', run=EVAL_NEW, **changes):
    row = {'eval_run_id': run, 'model_version': version, 'eval_set': eval_set, 'variant': variant,
           'item_id': item_id, 'actual': 'delta', 'predicted': 'delta', 'confidence': '0.8',
           'true_prob': '0.8', 'correct': 'true', 'probabilities_json': '{}',
           'source': '/Volumes/x/y/z/secret.png'}
    return {**row, **changes}


def sdk_run(run_id, status=RunInfoStatus.FINISHED, experiment_id=EXPERIMENT, metrics=(), tags=None,
            start=EPOCH * 1000, end=(EPOCH + 600) * 1000, name='train-run', lifecycle='active'):
    return Run(info=RunInfo(run_id=run_id, run_name=name, status=status, start_time=start,
                            end_time=end, experiment_id=experiment_id, lifecycle_stage=lifecycle),
               data=RunData(metrics=[Metric(key=k, value=v, step=s, timestamp=1) for k, v, s in metrics],
                            tags=[RunTag(key=k, value=v) for k, v in (tags or {}).items()]))


class FakeWorkspace:
    """SQL warehouse, serving endpoint and MLflow experiments, emulating the queries' semantics."""

    def __init__(self):
        self.sql, self.sql_error = [], None
        self.eval_runs, self.eval_metrics, self.eval_predictions = [], [], []
        self.groups, self.recent = [], []
        self.endpoint_error, self.endpoint_gets = None, 0
        self.entities = [NS(name='airplane-tail-v2', entity_name='m', entity_version='2')]
        self.runs, self.histories, self.search_calls, self.get_run_calls = [], {}, [], []
        self.history_calls, self.experiment_error, self.yielded = [], None, 0
        self.statement_execution = NS(execute_statement=self._execute, get_statement=self._get,
                                      cancel_execution=lambda statement_id: None)
        self.serving_endpoints = NS(get=self._endpoint)
        self.experiments = NS(search_runs=self._search_runs, get_run=self._get_run,
                              get_history=self._get_history)

    # SQL -------------------------------------------------------------------
    @staticmethod
    def _response(rows, columns):
        return NS(statement_id='st-1', status=NS(state=StatementState.SUCCEEDED, error=None),
                  manifest=NS(schema=NS(columns=[NS(name=c) for c in columns])),
                  result=NS(data_array=[[row.get(c) for c in columns] for row in rows],
                            next_chunk_index=None))

    def _get(self, statement_id):
        return NS(statement_id=statement_id, status=NS(state=StatementState.PENDING, error=None))

    def _latest(self):
        # Emulates: ORDER BY created_at DESC, eval_run_id DESC LIMIT 1
        if not self.eval_runs:
            return None
        return max(self.eval_runs, key=lambda r: (int(r['created_epoch']), r['eval_run_id']))['eval_run_id']

    def _execute(self, statement, warehouse_id, wait_timeout, parameters):
        params = {p.name: p.value for p in parameters or []}
        self.sql.append((statement, params, wait_timeout, warehouse_id))
        if self.sql_error:
            raise self.sql_error
        latest = self._latest()
        if 'model_eval_predictions' in statement and 'eval_run_id' in params:
            rows = [r for r in self.eval_predictions
                    if r['eval_run_id'] == params['eval_run_id'] and r['eval_set'] == params['eval_set']
                    and r['variant'] == params['variant']]
            return self._response(rows, ['model_version', 'item_id', 'actual', 'predicted',
                                         'confidence', 'true_prob', 'correct'])
        if 'model_eval_predictions' in statement:
            rows = [{**r, 'latest_run_id': latest} for r in self.eval_predictions
                    if r['eval_run_id'] == latest and r['eval_set'] == params['eval_set']
                    and r['variant'] == params['variant']]
            if latest and not rows:
                rows = [{'latest_run_id': latest}]
            return self._response(rows, ['latest_run_id', 'model_version', 'item_id', 'actual',
                                         'predicted', 'confidence', 'true_prob', 'correct'])
        if 'model_eval_metrics' in statement:
            rows = [r for r in self.eval_metrics if r['eval_run_id'] == params['eval_run_id']]
            return self._response(rows, ['model_version', 'eval_set', 'variant', 'metric', 'value', 'n'])
        if 'model_eval_runs' in statement:
            rows = [r for r in self.eval_runs if r['eval_run_id'] == latest]
            return self._response(rows, ['eval_run_id', 'created_epoch', 'model_version', 'recipe',
                                         'dataset_version', 'training_run_id', 'val_accuracy',
                                         'val_loss', 'aliases', 'serving', 'selected'])
        if "'status' AS kind" in statement:
            return self._response(self.groups, ['kind', 'label', 'n'])
        if 'photo_submissions' in statement:
            return self._response(self.recent, ['submission_id', 'submitted_epoch', 'predicted_airline',
                                                'confidence', 'review_label', 'model_version'])
        raise AssertionError(f'unexpected SQL: {statement}')

    # Serving ---------------------------------------------------------------
    def _endpoint(self, name):
        self.endpoint_gets += 1
        if self.endpoint_error:
            raise self.endpoint_error
        return NS(config=NS(served_entities=self.entities))

    # MLflow ----------------------------------------------------------------
    def _search_runs(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.experiment_error:
            raise self.experiment_error
        for run in self.runs:  # an SDK iterator keeps paging; count what was consumed
            self.yielded += 1
            yield run

    def _get_run(self, run_id):
        self.get_run_calls.append(run_id)
        if self.experiment_error:
            raise self.experiment_error
        for run in self.runs:
            if run.info.run_id == run_id:
                return GetRunResponse(run=run)
        raise errors.ResourceDoesNotExist(f'Run {run_id} not found')

    def _get_history(self, metric_key, run_id):
        self.history_calls.append((metric_key, run_id))
        return iter(self.histories.get((run_id, metric_key), []))


def make(experiment_id=EXPERIMENT, warehouse_id='wh'):
    fake, clock, created = FakeWorkspace(), Clock(), []
    perf = performance.PerformanceService(
        pipeline.Settings(warehouse_id=warehouse_id), lambda: (created.append(1), fake)[1],
        training_experiment_id=experiment_id, sleep=clock.sleep, monotonic=clock.monotonic)
    service = pipeline.SubmissionService(pipeline.Settings(warehouse_id='wh'),
                                         client_factory=lambda purpose: fake,
                                         executor=NS(submit=None, shutdown=lambda **k: None))
    client = TestClient(main.create_app(service, perf))
    return NS(client=client, fake=fake, clock=clock, created=created, perf=perf)


class Base(unittest.TestCase):
    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        return response.json()

    def error(self, response, status):
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        detail = response.json()['detail']
        self.assertIsInstance(detail, str)
        return detail


# Summary ------------------------------------------------------------------------
class SummaryTests(Base):
    def test_unavailable_when_nothing_published(self):
        t = make()
        body = self.ok(t.client.get('/api/performance/summary'))
        self.assertEqual(body, {'available': False, 'eval_run_id': None, 'evaluated_at': None,
                                'serving': {'endpoint': 'airplane-tail-identifier', 'model_version': '2'},
                                'models': [], 'eval_sets': [], 'metrics': []})
        self.assertEqual(len(t.fake.sql), 1)  # no metrics query without a comparison

    def test_latest_comparison_models_and_metrics(self):
        t = make()
        t.fake.eval_runs = [
            run_row('1', eval_run_id=EVAL_OLD, created_epoch=str(EPOCH - 86400)),
            run_row('10', aliases='["prod", "demo"]', selected='true', val_loss=None,
                    created_epoch=str(EPOCH - 5)),
            run_row('2', aliases='not json', serving='true', recipe=None),
            run_row('1', aliases='["demo"]', recipe='v1', dataset_version='v1')]
        t.fake.eval_metrics = [
            metric_row('2', 'context_sensitivity', 'context_full', value='0.5'),
            metric_row('2', 'context_sensitivity', 'context_1.5'),
            metric_row('2', 'context_sensitivity', 'context_1.25'),
            metric_row('2', 'full_plane_stress', 'tail_crop'),
            metric_row('2', 'full_plane_stress', 'whole_image', value=None),
            metric_row('10', 'v1_test', 'tail_crop', metric='macro_f1', value='0.8'),
            metric_row('2', 'v1_test', 'tail_crop'),
            metric_row('2', 'v1_test', 'whole_image'),        # not a v1_test variant: dropped
            metric_row('2', 'mystery_set', 'tail_crop'),      # unknown set: dropped
            metric_row('2', 'v2_val', 'tail_crop', n='13'),
            metric_row('1', 'v2_test', 'tail_crop', run=EVAL_OLD)]  # older comparison
        body = self.ok(t.client.get('/api/performance/summary'))
        self.assertEqual((body['available'], body['eval_run_id'], body['evaluated_at']),
                         (True, EVAL_NEW, '2026-10-27T20:00:00Z'))
        self.assertEqual(body['serving'], {'endpoint': 'airplane-tail-identifier', 'model_version': '2'})
        self.assertEqual([m['model_version'] for m in body['models']], ['1', '2', '10'])
        one, two, ten = body['models']
        self.assertEqual(one, {'model_version': '1', 'recipe': 'v1', 'dataset_version': 'v1',
                               'training_run_id': RUN_A, 'val_accuracy': 0.9230769230769231,
                               'val_loss': 0.31, 'aliases': ['demo'], 'serving': False,
                               'selected': False})
        self.assertEqual((two['aliases'], two['serving'], two['recipe']), ([], True, None))
        self.assertEqual((ten['aliases'], ten['selected'], ten['val_loss']), (['prod', 'demo'], True, None))

        self.assertEqual([s['eval_set'] for s in body['eval_sets']],
                         ['v2_val', 'v1_test', 'full_plane_stress', 'context_sensitivity'])
        sets = {s['eval_set']: s for s in body['eval_sets']}
        self.assertEqual(sets['full_plane_stress']['variants'], ['whole_image', 'tail_crop'])
        self.assertEqual(sets['context_sensitivity']['variants'],
                         ['context_1.25', 'context_1.5', 'context_full'])
        self.assertEqual(sets['v1_test']['variants'], ['tail_crop'])
        for entry in body['eval_sets']:
            self.assertEqual(set(entry), {'eval_set', 'label', 'description', 'variants'})
            self.assertTrue(entry['label'] and len(entry['description']) > 40)

        metrics = body['metrics']
        self.assertEqual(len(metrics), 8)
        self.assertEqual(metrics[0], {'model_version': '2', 'eval_set': 'v2_val', 'variant': 'tail_crop',
                                      'metric': 'accuracy', 'value': 1.0, 'n': 13})
        self.assertEqual(metrics[-1]['model_version'], '10')
        whole = next(m for m in metrics if m['variant'] == 'whole_image')
        self.assertIsNone(whole['value'])
        self.assertNotIn('mystery_set', json.dumps(body))

        runs_sql, metrics_sql = t.fake.sql
        self.assertIn('ORDER BY created_at DESC, eval_run_id DESC LIMIT 1', runs_sql[0])
        self.assertEqual(runs_sql[1], {})
        self.assertEqual(metrics_sql[1], {'eval_run_id': EVAL_NEW})
        self.assertEqual((metrics_sql[2], metrics_sql[3]), ('20s', 'wh'))

    def test_catalogue_matches_evaluation_sets(self):
        sys.path.insert(0, str(ROOT / 'evaluation'))
        import eval_metrics
        self.assertEqual(tuple(e['eval_set'] for e in performance.EVAL_SETS), eval_metrics.EVAL_SETS)
        context = performance._CATALOGUE['context_sensitivity']['variants']
        self.assertEqual(context[:-1], tuple(eval_metrics.context_variant(s) for s in (1.0, 1.5, 2.0, 3.0)))
        text = ' '.join(e['description'] for e in performance.EVAL_SETS)
        for phrase in ('Acroterion', 'BWI', 'original 10 held-out', '6 new', 'select'):
            self.assertIn(phrase, text)

    def test_serving_version_null_on_error_or_ambiguity(self):
        t = make()
        t.fake.endpoint_error = errors.PermissionDenied('no')
        self.assertIsNone(self.ok(t.client.get('/api/performance/summary'))['serving']['model_version'])
        t = make()
        t.fake.entities = t.fake.entities * 2  # traffic split: no single version
        self.assertIsNone(self.ok(t.client.get('/api/performance/summary'))['serving']['model_version'])

    def test_summary_is_cached_for_30_seconds(self):
        t = make()
        t.fake.eval_runs = [run_row('1')]
        first = self.ok(t.client.get('/api/performance/summary'))
        calls, gets = len(t.fake.sql), t.fake.endpoint_gets
        t.clock.t += 29.5
        first['models'].clear()  # responses are copies
        self.assertEqual(len(self.ok(t.client.get('/api/performance/summary'))['models']), 1)
        self.assertEqual((len(t.fake.sql), t.fake.endpoint_gets), (calls, gets))
        t.clock.t += 0.5
        self.ok(t.client.get('/api/performance/summary'))
        self.assertEqual((len(t.fake.sql), t.fake.endpoint_gets), (2 * calls, gets + 1))

    def test_sql_failure_is_503_and_not_cached(self):
        t = make()
        t.fake.sql_error = RuntimeError('warehouse down')
        detail = self.error(t.client.get('/api/performance/summary'), 503)
        self.assertNotIn('warehouse down', detail)
        t.fake.sql_error = None
        self.assertFalse(self.ok(t.client.get('/api/performance/summary'))['available'])

    def test_missing_warehouse_is_503(self):
        t = make(warehouse_id=None)
        self.error(t.client.get('/api/performance/summary'), 503)
        self.assertEqual(t.fake.sql, [])


# Predictions -------------------------------------------------------------------
class PredictionTests(Base):
    URL = '/api/performance/predictions'

    def test_validation(self):
        t = make()
        for params in ({}, {'eval_set': 'v1_test'}, {'variant': 'tail_crop'},
                       {'eval_set': 'nope', 'variant': 'tail_crop'},
                       {'eval_set': 'v1_test', 'variant': 'whole_image'},
                       {'eval_set': 'context_sensitivity', 'variant': 'context_9'},
                       {'eval_set': 'context_sensitivity', 'variant': 'context_1.0\n'},
                       {'eval_set': 'v1_test', 'variant': "tail_crop' OR '1'='1"},
                       {'eval_set': "v1_test'; DROP TABLE x; --", 'variant': 'tail_crop'}):
            with self.subTest(params=params):
                self.error(t.client.get(self.URL, params=params), 422)
        self.assertEqual(t.fake.sql, [])
        self.assertEqual(t.created, [])
        detail = self.error(t.client.get(self.URL, params={'eval_set': 'x', 'variant': 'y'}), 422)
        self.assertIn('v2_val', detail)

    def test_rows_for_latest_comparison(self):
        t = make()
        t.fake.eval_runs = [run_row('1'), run_row('1', eval_run_id=EVAL_OLD, created_epoch='1')]
        t.fake.eval_predictions = [
            prediction_row('10', 'b'), prediction_row('2', 'z'),
            prediction_row('2', 'a', predicted='united', correct='false', confidence='0.55',
                           true_prob='0.1'),
            prediction_row('2', 'old', run=EVAL_OLD),
            prediction_row('2', 'other-variant', variant='whole_image')]
        body = self.ok(t.client.get(self.URL, params={'eval_set': 'v1_test', 'variant': 'tail_crop'}))
        self.assertEqual(body['eval_run_id'], EVAL_NEW)
        self.assertEqual([(r['model_version'], r['item_id']) for r in body['rows']],
                         [('2', 'a'), ('2', 'z'), ('10', 'b')])
        self.assertEqual(body['rows'][0], {'model_version': '2', 'item_id': 'a', 'actual': 'delta',
                                           'predicted': 'united', 'confidence': 0.55, 'true_prob': 0.1,
                                           'correct': False})
        self.assertNotIn('/Volumes', json.dumps(body))
        statement, params, _, _ = t.fake.sql[0]
        self.assertEqual(params, {'eval_set': 'v1_test', 'variant': 'tail_crop'})
        self.assertIn('ORDER BY created_at DESC, eval_run_id DESC LIMIT 1', statement)

        context = self.ok(t.client.get(self.URL, params={'eval_set': 'context_sensitivity',
                                                         'variant': 'context_1.25'}))
        self.assertEqual((context['eval_run_id'], context['rows']), (EVAL_NEW, []))

    def test_no_comparison_and_cache_per_key(self):
        t = make()
        query = {'eval_set': 'v2_val', 'variant': 'tail_crop'}
        self.assertEqual(self.ok(t.client.get(self.URL, params=query)), {'eval_run_id': None, 'rows': []})
        t.clock.t += 29
        self.ok(t.client.get(self.URL, params=query))
        self.assertEqual(len(t.fake.sql), 1)
        self.ok(t.client.get(self.URL, params={'eval_set': 'v1_test', 'variant': 'tail_crop'}))
        self.assertEqual(len(t.fake.sql), 2)
        t.clock.t += 1
        self.ok(t.client.get(self.URL, params=query))
        self.assertEqual(len(t.fake.sql), 3)

    def test_failure_is_503(self):
        t = make()
        t.fake.sql_error = TimeoutError('slow')
        self.error(t.client.get(self.URL, params={'eval_set': 'v2_val', 'variant': 'tail_crop'}), 503)

    def test_eval_run_id_selects_that_comparison(self):
        t = make()
        t.fake.eval_runs = [run_row('1'), run_row('1', eval_run_id=EVAL_OLD, created_epoch='1')]
        t.fake.eval_predictions = [prediction_row('2', 'new'), prediction_row('2', 'old', run=EVAL_OLD),
                                   prediction_row('1', 'old-v1', run=EVAL_OLD)]
        query = {'eval_set': 'v1_test', 'variant': 'tail_crop'}
        body = self.ok(t.client.get(self.URL, params={**query, 'eval_run_id': EVAL_OLD}))
        self.assertEqual(body['eval_run_id'], EVAL_OLD)
        self.assertEqual([(r['model_version'], r['item_id']) for r in body['rows']],
                         [('1', 'old-v1'), ('2', 'old')])
        statement, params, _, _ = t.fake.sql[0]
        self.assertEqual(params, {**query, 'eval_run_id': EVAL_OLD})
        self.assertNotIn(EVAL_OLD, statement)
        self.assertNotIn('model_eval_runs', statement)
        # Unknown (but well-formed) run: empty rows, the ID echoed.
        unknown = self.ok(t.client.get(self.URL, params={**query, 'eval_run_id': '0' * 32}))
        self.assertEqual(unknown, {'eval_run_id': '0' * 32, 'rows': []})

    def test_eval_run_id_validation(self):
        t = make()
        for run_id in ('', 'E' * 32, 'e' * 31, 'e' * 33, 'g' * 32, "e' OR '1'='1", 'e' * 32 + '\n'):
            with self.subTest(run_id=run_id):
                detail = self.error(t.client.get(self.URL, params={
                    'eval_set': 'v1_test', 'variant': 'tail_crop', 'eval_run_id': run_id}), 422)
                self.assertIn('eval_run_id', detail)
        self.assertEqual((t.fake.sql, t.created), ([], []))

    def test_eval_run_id_is_part_of_the_cache_key(self):
        t = make()
        t.fake.eval_runs = [run_row('1')]
        query = {'eval_set': 'v1_test', 'variant': 'tail_crop'}
        self.ok(t.client.get(self.URL, params=query))
        self.ok(t.client.get(self.URL, params={**query, 'eval_run_id': EVAL_NEW}))
        self.ok(t.client.get(self.URL, params={**query, 'eval_run_id': EVAL_OLD}))
        self.assertEqual(len(t.fake.sql), 3)
        self.ok(t.client.get(self.URL, params={**query, 'eval_run_id': EVAL_OLD}))
        self.ok(t.client.get(self.URL, params=query))
        self.assertEqual(len(t.fake.sql), 3)


# Training runs -----------------------------------------------------------------
class TrainingRunTests(Base):
    URL = '/api/performance/training-runs'

    def test_no_experiment_configured(self):
        t = make(experiment_id=None)
        self.assertEqual(self.ok(t.client.get(self.URL)), {'experiment_id': None, 'runs': []})
        self.error(t.client.get(f'{self.URL}/{RUN_A}/history'), 404)
        self.assertEqual((t.created, t.fake.search_calls), ([], []))

    def test_runs_are_mapped(self):
        t = make()
        t.fake.runs = [
            sdk_run(RUN_B, status=RunInfoStatus.RUNNING, end=None, name='improved-v2',
                    metrics=[('val_accuracy', 0.75, 7), ('train_loss', 0.4, 7)],
                    tags={'recipe': 'improved', 'dataset_version': 'v2'}),
            sdk_run(RUN_A, metrics=[('best_val_accuracy', 1.0, 0), ('val_accuracy', 0.9, 12),
                                    ('test_accuracy', 0.9, 0), ('test_macro_f1', float('nan'), 0)],
                    tags={'registered_model_version': '1', 'mlflow.runName': 'tag-name'}, name=None)]
        body = self.ok(t.client.get(self.URL))
        self.assertEqual(body['experiment_id'], EXPERIMENT)
        running, finished = body['runs']
        self.assertEqual(running, {
            'run_id': RUN_B, 'run_name': 'improved-v2', 'status': 'RUNNING',
            'start_time': '2026-10-27T20:00:00Z', 'end_time': None, 'recipe': 'improved',
            'dataset_version': 'v2', 'registered_model_version': None, 'best_val_accuracy': 0.75,
            'test_accuracy': None, 'test_macro_f1': None, 'epochs_completed': 7})
        self.assertEqual(finished, {
            'run_id': RUN_A, 'run_name': 'tag-name', 'status': 'FINISHED',
            'start_time': '2026-10-27T20:00:00Z', 'end_time': '2026-10-27T20:10:00Z', 'recipe': 'v1',
            'dataset_version': 'v1', 'registered_model_version': '1', 'best_val_accuracy': 1.0,
            'test_accuracy': 0.9, 'test_macro_f1': None, 'epochs_completed': 12})
        self.assertEqual(t.fake.search_calls, [{'experiment_ids': [EXPERIMENT], 'max_results': 20,
                                                'order_by': ['attributes.start_time DESC']}])

    def test_val_accuracy_fallback_only_while_running(self):
        t = make()
        metrics = [('val_accuracy', 0.6, 4)]
        t.fake.runs = [sdk_run(f'{i:032x}', status=status, metrics=metrics)
                       for i, status in enumerate((RunInfoStatus.RUNNING, RunInfoStatus.FINISHED,
                                                   RunInfoStatus.FAILED, RunInfoStatus.KILLED,
                                                   RunInfoStatus.SCHEDULED))]
        t.fake.runs.append(sdk_run('f' * 32, status=RunInfoStatus.RUNNING,
                                   metrics=metrics + [('best_val_accuracy', 0.7, 0)]))
        runs = self.ok(t.client.get(self.URL))['runs']
        self.assertEqual([(r['status'], r['best_val_accuracy']) for r in runs],
                         [('RUNNING', 0.6), ('FINISHED', None), ('FAILED', None), ('KILLED', None),
                          ('SCHEDULED', None), ('RUNNING', 0.7)])
        self.assertEqual({r['epochs_completed'] for r in runs}, {4})

    def test_at_most_twenty_and_cached_ten_seconds(self):
        t = make()
        t.fake.runs = [sdk_run(f'{i:032x}') for i in range(45)]
        self.assertEqual(len(self.ok(t.client.get(self.URL))['runs']), 20)
        self.assertEqual(t.fake.yielded, 20)  # did not page through the experiment
        t.clock.t += 9
        self.ok(t.client.get(self.URL))
        self.assertEqual(len(t.fake.search_calls), 1)
        t.clock.t += 1
        self.ok(t.client.get(self.URL))
        self.assertEqual(len(t.fake.search_calls), 2)

    def test_errors_are_503(self):
        t = make()
        t.fake.experiment_error = errors.PermissionDenied('no access')
        self.assertNotIn('no access', self.error(t.client.get(self.URL), 503))


class HistoryTests(Base):
    def url(self, run_id):
        return f'/api/performance/training-runs/{run_id}/history'

    def test_invalid_or_foreign_runs_are_404(self):
        t = make()
        t.fake.runs = [sdk_run(RUN_A, experiment_id='999'),
                       sdk_run(RUN_B, lifecycle='deleted')]
        for run_id in ('abc', 'A' * 32, 'a' * 33, 'g' * 32, 'a' * 31 + '%0A'):
            self.error(t.client.get(self.url(run_id)), 404)
        self.assertEqual(t.fake.get_run_calls, [])
        self.error(t.client.get(self.url(RUN_A)), 404)   # other experiment
        self.error(t.client.get(self.url(RUN_B)), 404)   # deleted
        self.error(t.client.get(self.url('c' * 32)), 404)  # SDK NotFound
        self.assertEqual(t.fake.history_calls, [])

    def test_history_sorted_deduplicated_and_complete(self):
        t = make()
        t.fake.runs = [sdk_run(RUN_A)]
        t.fake.histories = {
            (RUN_A, 'val_accuracy'): [Metric(key='val_accuracy', value=0.8, step=2, timestamp=5),
                                      Metric(key='val_accuracy', value=0.5, step=1, timestamp=3),
                                      Metric(key='val_accuracy', value=0.7, step=2, timestamp=9)],
            (RUN_A, 'lr'): [Metric(key='lr', value=float('inf'), step=1, timestamp=1),
                            Metric(key='lr', value=0.001, step=None, timestamp=1)]}
        body = self.ok(t.client.get(self.url(RUN_A)))
        self.assertEqual(body, {'run_id': RUN_A, 'status': 'FINISHED', 'metrics': {
            'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [[1, 0.5], [2, 0.7]],
            'val_macro_f1': [], 'lr': [[0, 0.001], [1, None]]}})
        self.assertEqual(t.fake.history_calls, [(k, RUN_A) for k in performance.HISTORY_KEYS])

    def test_cache_depends_on_status(self):
        for status, ttl in ((RunInfoStatus.RUNNING, 10), (RunInfoStatus.FINISHED, 300),
                            (RunInfoStatus.FAILED, 300)):
            with self.subTest(status=status):
                t = make()
                t.fake.runs = [sdk_run(RUN_A, status=status)]
                self.ok(t.client.get(self.url(RUN_A)))
                t.clock.t += ttl - 1
                self.ok(t.client.get(self.url(RUN_A)))
                self.assertEqual(len(t.fake.get_run_calls), 1)
                t.clock.t += 1
                self.ok(t.client.get(self.url(RUN_A)))
                self.assertEqual(len(t.fake.get_run_calls), 2)

    def test_errors_are_503(self):
        t = make()
        t.fake.experiment_error = errors.TemporarilyUnavailable('busy')
        self.error(t.client.get(self.url(RUN_A)), 503)


# Submissions -------------------------------------------------------------------
class SubmissionStatsTests(Base):
    URL = '/api/performance/submissions'

    def test_aggregates_and_recent(self):
        t = make()
        t.fake.groups = [
            {'kind': 'status', 'label': 'succeeded', 'n': '7'},
            {'kind': 'status', 'label': 'failed', 'n': '2'},
            {'kind': 'airline', 'label': 'united', 'n': '2'},
            {'kind': 'airline', 'label': 'delta', 'n': '5'},
            {'kind': 'airline', 'label': 'alaska', 'n': '2'},
            {'kind': 'airline', 'label': None, 'n': '9'},
            {'kind': 'bucket', 'label': '1', 'n': '3'},
            {'kind': 'bucket', 'label': '4', 'n': '4'},
            {'kind': 'review', 'label': 'correct', 'n': '3'},
            {'kind': 'review', 'label': 'incorrect', 'n': '1'}]
        t.fake.recent = [
            {'submission_id': 's1', 'submitted_epoch': str(EPOCH), 'predicted_airline': 'delta',
             'confidence': '0.91', 'review_label': 'delta', 'model_version': '2'},
            {'submission_id': 's2', 'submitted_epoch': str(EPOCH - 60), 'predicted_airline': None,
             'confidence': None, 'review_label': None, 'model_version': None}]
        body = self.ok(t.client.get(self.URL))
        self.assertEqual({k: body[k] for k in ('total', 'succeeded', 'failed', 'reviewed',
                                               'reviewed_correct')},
                         {'total': 9, 'succeeded': 7, 'failed': 2, 'reviewed': 4, 'reviewed_correct': 3})
        self.assertEqual(body['by_predicted_airline'], [{'airline': 'delta', 'count': 5},
                                                        {'airline': 'alaska', 'count': 2},
                                                        {'airline': 'united', 'count': 2}])
        self.assertEqual(body['confidence_buckets'], [
            {'lower': 0.0, 'upper': 0.2, 'count': 0}, {'lower': 0.2, 'upper': 0.4, 'count': 3},
            {'lower': 0.4, 'upper': 0.6, 'count': 0}, {'lower': 0.6, 'upper': 0.8, 'count': 0},
            {'lower': 0.8, 'upper': 1.0, 'count': 4}])
        self.assertEqual(body['recent'], [
            {'submission_id': 's1', 'submitted_at': '2026-10-27T20:00:00Z', 'predicted_airline': 'delta',
             'confidence': 0.91, 'review_label': 'delta', 'model_version': '2'},
            {'submission_id': 's2', 'submitted_at': '2026-10-27T19:59:00Z', 'predicted_airline': None,
             'confidence': None, 'review_label': None, 'model_version': None}])
        aggregate, recent = t.fake.sql
        self.assertIn(performance.CONFIDENCE_BUCKET_SQL, aggregate[0])
        self.assertIn('LIMIT 10', recent[0])
        for statement in (aggregate[0], recent[0]):
            self.assertIn(f'{TABLE_PREFIX}`photo_submissions`', statement)
            self.assertNotIn('_path', statement)

    def test_empty_table(self):
        t = make()
        body = self.ok(t.client.get(self.URL))
        self.assertEqual({k: v for k, v in body.items() if k != 'confidence_buckets'},
                         {'total': 0, 'succeeded': 0, 'failed': 0, 'reviewed': 0, 'reviewed_correct': 0,
                          'by_predicted_airline': [], 'recent': []})
        self.assertEqual([b['count'] for b in body['confidence_buckets']], [0] * 5)

    def test_bucket_expression_edges(self):
        # The CASE expression is portable SQL; evaluate the exact text with sqlite.
        db = sqlite3.connect(':memory:')
        sql = f'SELECT {performance.CONFIDENCE_BUCKET_SQL} FROM (SELECT ? AS confidence)'
        cases = {None: None, 0.0: 0, 0.19999999: 0, 0.2: 1, 0.3999: 1, 0.4: 2, 0.6: 3, 0.79: 3,
                 0.8: 4, 0.99: 4, 1.0: 4}
        self.assertEqual({v: db.execute(sql, (v,)).fetchone()[0] for v in cases}, cases)

    def test_reviewed_correct_needs_a_prediction(self):
        # Evaluate the exact review CASE used by the aggregate query with sqlite.
        self.assertIn('predicted_airline IS NOT NULL AND review_label = predicted_airline',
                      performance.REVIEW_SQL)
        db = sqlite3.connect(':memory:')
        db.execute('CREATE TABLE s (predicted_airline TEXT, review_label TEXT)')
        db.executemany('INSERT INTO s VALUES (?, ?)', [
            ('delta', 'delta'), ('delta', 'united'), (None, 'delta'), (None, None), ('delta', None)])
        counts = dict(db.execute(f'SELECT review, COUNT(*) FROM (SELECT {performance.REVIEW_SQL} '
                                 'AS review FROM s) WHERE review IS NOT NULL GROUP BY review'))
        self.assertEqual(counts, {'correct': 1, 'incorrect': 2})  # reviewed 3, correct 1
        t = make()
        t.client.get(self.URL)
        self.assertIn(performance.REVIEW_SQL, t.fake.sql[0][0])

    def test_cached_fifteen_seconds_and_503(self):
        t = make()
        self.ok(t.client.get(self.URL))
        t.clock.t += 14
        self.ok(t.client.get(self.URL))
        self.assertEqual(len(t.fake.sql), 2)
        t.clock.t += 1
        t.fake.sql_error = RuntimeError('SQL statement FAILED: permission denied')
        self.error(t.client.get(self.URL), 503)


# Wiring and safety -------------------------------------------------------------
class WiringTests(Base):
    def test_no_databricks_calls_until_requested(self):
        t = make()
        self.assertEqual(t.created, [])
        self.assertEqual(t.client.get('/api/health').json(), {'status': 'ok'})
        self.assertEqual(t.created, [])

    def test_user_input_never_in_sql_text(self):
        t = make()
        t.fake.eval_runs = [run_row('1')]
        t.client.get('/api/performance/summary')
        t.client.get('/api/performance/predictions',
                     params={'eval_set': 'context_sensitivity', 'variant': 'context_1.5'})
        t.client.get('/api/performance/submissions')
        self.assertEqual(len(t.fake.sql), 5)
        for statement, params, wait, _ in t.fake.sql:
            self.assertEqual(wait, '20s')
            for value in ('context_1.5', 'context_sensitivity', EVAL_NEW):
                self.assertNotIn(value, statement)
            for table in ('model_eval_runs', 'model_eval_metrics', 'model_eval_predictions',
                          'photo_submissions'):
                if table in statement:
                    self.assertIn(f'{TABLE_PREFIX}`{table}`', statement)

    def test_default_wiring_shares_the_submission_client_and_reads_env(self):
        fake, purposes = FakeWorkspace(), []
        service = pipeline.SubmissionService(
            pipeline.Settings(warehouse_id='wh'),
            client_factory=lambda purpose: (purposes.append(purpose), fake)[1],
            executor=NS(submit=None, shutdown=lambda **k: None))
        fake.runs = [sdk_run(RUN_A)]
        with mock.patch.dict(os.environ, {performance.EXPERIMENT_ENV: EXPERIMENT}):
            client = TestClient(main.create_app(service))
        self.assertEqual(purposes, [])
        self.assertEqual(self.ok(client.get('/api/performance/training-runs'))['runs'][0]['run_id'], RUN_A)
        self.assertEqual(self.ok(client.get('/api/performance/summary'))['available'], False)
        self.assertEqual(purposes, ['api'])
        self.assertIs(service.client('api'), fake)

    def test_sql_waits_stay_below_the_browser_timeout(self):
        self.assertEqual((performance.SQL_WAIT, performance.SQL_MAX_WAIT_S), ('20s', 25))
        t = make()
        with mock.patch.object(pipeline, 'execute_sql', wraps=pipeline.execute_sql) as spy:
            self.ok(t.client.get('/api/performance/submissions'))
        self.assertEqual(spy.call_count, 2)
        for call in spy.call_args_list:
            self.assertEqual((call.kwargs['wait_timeout'], call.kwargs['max_wait_s']), ('20s', 25))
        js = (ROOT / 'app' / 'static' / 'performance.js').read_text(encoding='utf-8')
        self.assertIn('const REQUEST_TIMEOUT_MS = 60000;', js)

    def test_experiment_id_from_env(self):
        for value, expected in (('', None), ('  ', None), (EXPERIMENT, EXPERIMENT),
                                (f' {EXPERIMENT} ', EXPERIMENT), ('1 OR 1', None), ('a/b', None)):
            with self.subTest(value=value), mock.patch.dict(os.environ, {performance.EXPERIMENT_ENV: value}):
                self.assertEqual(performance.experiment_id_from_env(), expected)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(performance.EXPERIMENT_ENV, None)
            self.assertIsNone(performance.experiment_id_from_env())

    def test_unknown_performance_path_is_404(self):
        t = make()
        self.assertEqual(t.client.get('/api/performance/nope').status_code, 404)
        self.assertEqual(t.client.get(f'/api/performance/training-runs/{RUN_A}').status_code, 404)

    def test_bundle_declares_read_access(self):
        import yaml
        bundle = yaml.safe_load((ROOT / 'databricks.yml').read_text(encoding='utf-8'))
        app_yaml = yaml.safe_load((ROOT / 'app' / 'app.yaml').read_text(encoding='utf-8'))
        env = {item['name']: item for item in app_yaml['env']}
        key = env[performance.EXPERIMENT_ENV]['valueFrom']
        resources = {r['name']: r for r in bundle['resources']['apps']['tailspot']['resources']}
        self.assertEqual(resources[key]['experiment'],
                         {'experiment_id': '${var.training_experiment_id}', 'permission': 'CAN_READ'})
        self.assertEqual(bundle['variables']['training_experiment_id']['default'], EXPERIMENT)
        schema = pipeline.Settings().schema
        securables = {r['uc_securable']['securable_full_name']: r['uc_securable']
                      for r in resources.values() if 'uc_securable' in r}
        for table in ('model_eval_runs', 'model_eval_metrics', 'model_eval_predictions'):
            spec = securables[f'{schema}.{table}']
            self.assertEqual((spec['securable_type'], spec['permission']), ('TABLE', 'SELECT'))
            self.assertIn(f'CREATE TABLE IF NOT EXISTS {schema}.{table} ',
                          (ROOT / 'app_setup' / 'create_eval_tables.sql').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
