"""Tests for GET /api/submissions/{id}/photo (photos on the Model performance tab)."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'tests', ROOT / 'app'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from databricks.sdk import errors  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import main  # noqa: E402
import performance  # noqa: E402
import pipeline  # noqa: E402
from test_app_performance import EPOCH, FakeWorkspace  # noqa: E402

VOL = '/Volumes/dbx_ai_app/airplane_tail_id/submitted_photos'
SID = '11111111-2222-4333-8444-0000000000aa'
OTHER = '11111111-2222-4333-8444-0000000000bb'


def png(size, colour=(10, 120, 200)):
    buffer = io.BytesIO()
    Image.new('RGB', size, colour).save(buffer, 'PNG')
    return buffer.getvalue()


class Contents(io.BytesIO):
    def __init__(self, data, fail=False):
        super().__init__(data)
        self.fail, self.closed_by_app = fail, False

    def read(self, size=-1):
        if self.fail:
            raise ConnectionError('stream broke')
        return super().read(size)

    def close(self):
        self.closed_by_app = True
        super().close()


class FakeFiles:
    def __init__(self):
        self.store, self.downloads, self.error, self.opened = {}, [], None, []

    def download(self, file_path):
        self.downloads.append(file_path)
        if self.error:
            raise self.error
        if file_path not in self.store:
            raise errors.NotFound(f'{file_path} not found')
        contents = Contents(self.store[file_path])
        self.opened.append(contents)
        return NS(contents=contents, content_length=len(self.store[file_path]), content_type='image/png')


def row(sid, crop=None, original=None):
    folder = f'{VOL}/2026/10/27/{sid}'
    return {'submission_id': sid, 'submitted_epoch': str(EPOCH), 'status': 'succeeded',
            'predicted_airline': 'delta', 'confidence': '0.9', 'review_label': None, 'review_source': None,
            'model_version': '3', 'crop_path': crop or f'{folder}/tail_crop.png',
            'original_path': original if original is not None else f'{folder}/original.png'}


def make():
    fake, files = FakeWorkspace(), FakeFiles()
    perf = performance.PerformanceService(pipeline.Settings(warehouse_id='wh'), lambda: fake,
                                          files_client_factory=lambda: NS(files=files))
    service = pipeline.SubmissionService(pipeline.Settings(warehouse_id='wh'), client_factory=lambda p: fake,
                                         executor=NS(submit=None, shutdown=lambda **k: None))
    return TestClient(main.create_app(service, perf)), perf, fake, files


class PhotoTests(unittest.TestCase):
    def setUp(self):
        self.client, self.perf, self.fake, self.files = make()
        self.crop, self.original = png((900, 600)), png((1800, 1200), (200, 30, 30))
        r = row(SID)
        self.fake.photo_rows = [r]
        self.files.store = {r['crop_path']: self.crop, r['original_path']: self.original}

    def lookups(self):
        return [s for s in self.fake.sql if 'WHERE submission_id = :submission_id' in s[0]]

    def test_crop_and_original_are_streamed_unchanged(self):
        for kind, expected in (('crop', self.crop), ('original', self.original)):
            with self.subTest(kind=kind):
                response = self.client.get(f'/api/submissions/{SID}/photo', params={'kind': kind})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, expected)
                self.assertEqual(response.headers['content-type'], 'image/png')
                self.assertEqual(response.headers['cache-control'], 'private, max-age=300')
                self.assertEqual(response.headers['x-content-type-options'], 'nosniff')
                self.assertEqual(response.headers['content-length'], str(len(expected)))
        self.assertTrue(all(c.closed_by_app for c in self.files.opened))
        statement, params, _, _ = self.lookups()[0]
        self.assertEqual(params, {'submission_id': SID})
        self.assertNotIn(SID, statement)
        self.assertIn('`dbx_ai_app`.`airplane_tail_id`.`photo_submissions`', statement)

    def test_thumbnail_is_small_png_and_cached(self):
        response = self.client.get(f'/api/submissions/{SID}/photo')  # default kind=thumb
        self.assertEqual(response.status_code, 200)
        with Image.open(io.BytesIO(response.content)) as image:
            self.assertEqual((image.format, image.size), ('PNG', (256, 171)))
        self.assertEqual(self.client.get(f'/api/submissions/{SID}/photo?kind=thumb').content, response.content)
        self.assertEqual(len(self.files.downloads), 1)
        self.assertEqual(len(self.lookups()), 1)

    def test_paths_from_recent_list_avoid_extra_queries_and_never_leak(self):
        self.fake.recent, self.fake.photo_rows = [row(SID)], []
        body = self.client.get('/api/performance/submissions').json()
        self.assertTrue(body['recent'][0]['has_photo'])
        self.assertNotIn('/Volumes', str(body))
        self.assertEqual(self.client.get(f'/api/submissions/{SID}/photo?kind=crop').status_code, 200)
        self.assertEqual(self.lookups(), [])

    def test_upper_case_id_is_normalised(self):
        self.assertEqual(self.client.get(f'/api/submissions/{SID.upper()}/photo?kind=crop').status_code, 200)

    def test_invalid_kind_is_422_and_bad_ids_are_404_without_queries(self):
        response = self.client.get(f'/api/submissions/{SID}/photo?kind=../../secret')
        self.assertEqual(response.status_code, 422)
        self.assertIsInstance(response.json()['detail'], str)
        for bad in ('abc', '..%2F..%2Fetc', SID + 'x', "1' OR '1'='1"):
            with self.subTest(bad=bad):
                self.assertEqual(self.client.get(f'/api/submissions/{bad}/photo').status_code, 404)
        self.assertEqual(self.fake.sql, [])
        self.assertEqual(self.files.downloads, [])

    def test_unknown_submission_and_missing_file_are_404(self):
        self.assertEqual(self.client.get(f'/api/submissions/{OTHER}/photo?kind=crop').status_code, 404)
        self.files.store.pop(row(SID)['original_path'])
        self.assertEqual(self.client.get(f'/api/submissions/{SID}/photo?kind=original').status_code, 404)

    def test_paths_outside_the_volume_or_for_another_submission_are_refused(self):
        cases = [row(SID, crop='/Volumes/other/schema/vol/2026/10/27/' + SID + '/tail_crop.png'),
                 row(SID, crop=f'{VOL}/2026/10/27/{OTHER}/tail_crop.png'),
                 row(SID, crop=f'{VOL}/2026/10/27/{SID}/../../{OTHER}/tail_crop.png'),
                 row(SID, crop=f'{VOL}/2026/10/27/{SID}/secret.txt'),
                 row(SID, original='')]
        for case in cases:
            with self.subTest(case=case['crop_path']):
                client, _, fake, files = make()
                fake.photo_rows = [case]
                files.store = {case['crop_path']: self.crop}
                kind = 'original' if case['original_path'] == '' else 'crop'
                self.assertEqual(client.get(f'/api/submissions/{SID}/photo?kind={kind}').status_code, 404)
                self.assertEqual(files.downloads, [])

    def test_storage_and_lookup_failures_are_503(self):
        self.files.error = errors.PermissionDenied('no')
        response = self.client.get(f'/api/submissions/{SID}/photo?kind=crop')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['detail'], 'The photo is temporarily unavailable.')
        client, _, fake, _ = make()
        fake.sql_error = RuntimeError('SQL statement FAILED: warehouse stopped')
        self.assertEqual(client.get(f'/api/submissions/{SID}/photo').status_code, 503)

    def test_unreadable_image_thumbnail_is_503(self):
        self.files.store[row(SID)['crop_path']] = b'not a png'
        self.assertEqual(self.client.get(f'/api/submissions/{SID}/photo').status_code, 503)

    def test_files_client_is_the_submission_files_client_by_default(self):
        purposes, fake, files = [], FakeWorkspace(), FakeFiles()
        fake.files = files
        fake.photo_rows = [row(SID)]
        files.store = {row(SID)['crop_path']: self.crop}
        service = pipeline.SubmissionService(
            pipeline.Settings(warehouse_id='wh'), client_factory=lambda p: (purposes.append(p), fake)[1],
            executor=NS(submit=None, shutdown=lambda **k: None))
        client = TestClient(main.create_app(service))
        self.assertEqual(client.get(f'/api/submissions/{SID}/photo?kind=crop').status_code, 200)
        self.assertEqual(sorted(set(purposes)), ['api', 'files'])


if __name__ == '__main__':
    unittest.main()
