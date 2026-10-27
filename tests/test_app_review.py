"""Tests for the "Did we get the airline right?" feedback (POST /api/submissions/{id}/review)."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'tests', ROOT / 'app'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from databricks.sdk.service.sql import StatementState  # noqa: E402

import pipeline  # noqa: E402
from test_app_api import FakeWorkspace, ManualExecutor, fallback_row, make, png_bytes, post  # noqa: E402


class ReviewWorkspace(FakeWorkspace):
    """Adds UPDATE handling: affected rows and failures are scripted per test."""

    def __init__(self):
        super().__init__()
        self.affected, self.update_errors = '1', []

    def _execute(self, statement, warehouse_id, wait_timeout, parameters):
        if statement.startswith('UPDATE'):
            params = {p.name: p.value for p in parameters or []}
            self.sql.append((statement, params, wait_timeout))
            if self.update_errors:
                message = self.update_errors.pop(0)
                return pipeline_failed(message)
            return self._response([{'num_affected_rows': self.affected}])
        return super()._execute(statement, warehouse_id, wait_timeout, parameters)

    def updates(self):
        return [s for s in self.sql if s[0].startswith('UPDATE')]


def pipeline_failed(message):
    from types import SimpleNamespace as NS
    return NS(statement_id='st-x', status=NS(state=StatementState.FAILED, error=NS(message=message)),
              manifest=None, result=None)


def submitted():
    fake = ReviewWorkspace()
    client, service, fake, clock = make(fake=fake)
    response = post(client, png_bytes())
    assert response.status_code == 202, response.text
    return client, service, fake, clock, response.json()['submission_id']


class ReviewTests(unittest.TestCase):
    def test_correct_answer_records_predicted_airline(self):
        client, _, fake, _, sid = submitted()
        response = client.post(f'/api/submissions/{sid}/review', json={'correct': True})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(response.json()['review'], {'label': 'southwest', 'correct': True,
                                                     'source': 'app_user',
                                                     'reviewed_at': '2026-10-27T15:25:00Z'})
        (statement, params, _), = fake.updates()
        self.assertEqual(params, {'label': 'southwest', 'reviewed_at': '2026-10-27T15:25:00+00:00',
                                  'source': 'app_user', 'submission_id': sid, 'predicted': 'southwest'})
        self.assertIn('(review_label IS NULL OR review_source = :source)', statement)
        for value in (sid, 'southwest', 'app_user'):
            self.assertNotIn(value, statement)
        self.assertEqual(client.get(f'/api/submissions/{sid}').json()['review']['label'], 'southwest')

    def test_wrong_answer_records_chosen_airline_and_can_be_changed(self):
        client, _, fake, _, sid = submitted()
        response = client.post(f'/api/submissions/{sid}/review', json={'correct': False, 'airline': 'delta'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual((response.json()['review']['label'], response.json()['review']['correct']),
                         ('delta', False))
        changed = client.post(f'/api/submissions/{sid}/review', json={'correct': False, 'airline': 'other'})
        self.assertEqual(changed.json()['review']['label'], 'other')
        self.assertEqual([u[1]['label'] for u in fake.updates()], ['delta', 'other'])

    def test_new_submission_has_no_review(self):
        client, _, _, _, sid = submitted()
        self.assertIsNone(client.get(f'/api/submissions/{sid}').json()['review'])

    def test_invalid_answers_are_rejected_without_sql(self):
        client, _, fake, _, sid = submitted()
        cases = [({'correct': False}, 422), ({'correct': False, 'airline': 'jetblue'}, 422),
                 ({'correct': False, 'airline': 'southwest'}, 422),
                 ({'correct': True, 'airline': 'delta'}, 422), ({'correct': 'yes'}, 422),
                 ({'airline': 'delta'}, 422), ({'correct': True, 'extra': 1}, 422)]
        for body, code in cases:
            with self.subTest(body=body):
                response = client.post(f'/api/submissions/{sid}/review', json=body)
                self.assertEqual(response.status_code, code)
                self.assertIsInstance(response.json()['detail'], str)
        self.assertEqual(fake.updates(), [])

    def test_unknown_or_invalid_submission_is_404(self):
        client, _, fake, _, _ = submitted()
        self.assertEqual(client.post('/api/submissions/not-a-uuid/review',
                                     json={'correct': True}).status_code, 404)
        self.assertEqual(client.post('/api/submissions/00000000-0000-4000-8000-000000000000/review',
                                     json={'correct': True}).status_code, 404)
        self.assertEqual(fake.updates(), [])

    def test_not_finished_failed_or_unrecorded_cannot_be_reviewed(self):
        executor = ManualExecutor()
        client, service, fake, _ = make(executor=executor, fake=ReviewWorkspace())
        sid = post(client, png_bytes()).json()['submission_id']
        self.assertEqual(client.post(f'/api/submissions/{sid}/review', json={'correct': True}).status_code, 409)
        executor.run_all()
        with service._lock:
            service._jobs[sid]['public']['recorded'] = False
        response = client.post(f'/api/submissions/{sid}/review', json={'correct': True})
        self.assertEqual(response.status_code, 409)
        self.assertIn('not saved', response.json()['detail'])
        with service._lock:
            service._jobs[sid]['public'].update(recorded=True, status='failed', prediction=None)
        self.assertEqual(client.post(f'/api/submissions/{sid}/review', json={'correct': True}).status_code, 409)
        self.assertEqual(fake.updates(), [])

    def test_reviewer_label_is_not_overwritten(self):
        fake = ReviewWorkspace()
        client, _, fake, _ = make(fake=fake)
        sid = '11111111-2222-4333-8444-555555555555'
        fake.fallback_rows = [fallback_row(sid, review_label='delta', review_source='reviewer',
                                           reviewed_epoch='1793114500')]
        status = client.get(f'/api/submissions/{sid}').json()
        self.assertEqual(status['review'], {'label': 'delta', 'correct': False, 'source': 'reviewer',
                                            'reviewed_at': '2026-10-27T15:21:40Z'})
        response = client.post(f'/api/submissions/{sid}/review', json={'correct': True})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(fake.updates(), [])

    def test_legacy_label_without_source_counts_as_reviewer(self):
        fake = ReviewWorkspace()
        client, _, fake, _ = make(fake=fake)
        sid = '11111111-2222-4333-8444-666666666666'
        fake.fallback_rows = [fallback_row(sid, review_label='southwest', review_source=None,
                                           reviewed_epoch=None)]
        self.assertEqual(client.get(f'/api/submissions/{sid}').json()['review']['source'], 'reviewer')
        self.assertEqual(client.post(f'/api/submissions/{sid}/review', json={'correct': True}).status_code, 409)

    def test_restarted_app_reviews_recorded_submission(self):
        fake = ReviewWorkspace()
        client, _, fake, _ = make(fake=fake)
        sid = '11111111-2222-4333-8444-777777777777'
        fake.fallback_rows = [fallback_row(sid)]
        response = client.post(f'/api/submissions/{sid}/review', json={'correct': False, 'airline': 'united'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['review']['label'], 'united')

    def test_zero_rows_updated_is_conflict(self):
        client, _, fake, _, sid = submitted()
        fake.affected = '0'
        response = client.post(f'/api/submissions/{sid}/review', json={'correct': True})
        self.assertEqual(response.status_code, 409)
        self.assertIsNone(client.get(f'/api/submissions/{sid}').json()['review'])

    def test_concurrent_conflict_is_retried_and_other_failures_are_503(self):
        client, _, fake, clock, sid = submitted()
        fake.update_errors = ['[DELTA_CONCURRENT_APPEND] ConcurrentAppendException']
        self.assertEqual(client.post(f'/api/submissions/{sid}/review', json={'correct': True}).status_code, 200)
        self.assertEqual(len(fake.updates()), 2)
        self.assertIn(2, clock.sleeps)
        fake.update_errors = ['permission denied']
        response = client.post(f'/api/submissions/{sid}/review', json={'correct': False, 'airline': 'delta'})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['detail'], 'Your answer could not be saved. Please try again.')
        self.assertEqual(client.get(f'/api/submissions/{sid}').json()['review']['label'], 'southwest')

    def test_review_choices_match_classifier_classes(self):
        import ast
        tree = ast.parse((ROOT / 'training' / 'prepare_data.py').read_text(encoding='utf-8'))
        classes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                       and any(getattr(t, 'id', None) == 'CLASSES' for t in n.targets))
        self.assertEqual(pipeline.REVIEW_CHOICES, (*classes, 'other'))
        js = (ROOT / 'app' / 'static' / 'app.js').read_text(encoding='utf-8')
        for value in pipeline.REVIEW_CHOICES:
            self.assertIn(f"['{value}', ", js)


if __name__ == '__main__':
    unittest.main()
