"""Contract/privacy check using a fake serving transport; no network calls."""
import base64
import io
import json
import os
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

from PIL import Image, ImageOps

import client
from client import identify_tail


class ClientTests(unittest.TestCase):
    def test_upload_metadata_removed_and_prediction_parsed(self):
        original = Image.new("RGB", (24, 32), (80, 100, 120))
        exif = Image.Exif()
        exif[270] = "private upload metadata"
        buffer = io.BytesIO()
        original.save(buffer, format="PNG", exif=exif, icc_profile=b"private profile")

        def query(**kwargs):
            self.assertEqual(kwargs["name"], "airplane-tail-identifier")
            values = kwargs["dataframe_records"]
            self.assertEqual(len(values), 1)
            with Image.open(io.BytesIO(base64.b64decode(values[0]["image_base64"]))) as submitted:
                self.assertEqual(submitted.size, original.size)
                self.assertEqual(submitted.convert("RGB").tobytes(), original.tobytes())
                self.assertFalse(submitted.getexif())
                self.assertNotIn("icc_profile", submitted.info)
                self.assertNotIn("exif", submitted.info)
            return SimpleNamespace(predictions=[{
                "airline": "alaska", "confidence": 0.8,
                "probabilities_json": json.dumps({"alaska": 0.8, "united": 0.2}),
            }])

        fake = SimpleNamespace(serving_endpoints=SimpleNamespace(query=query))
        prediction = identify_tail(buffer.getvalue(), client=fake)
        self.assertEqual(prediction["airline"], "alaska")
        self.assertEqual(prediction["probabilities"]["alaska"], 0.8)

    def test_empty_upload_rejected_before_client_creation(self):
        with self.assertRaises(ValueError):
            identify_tail(b"")

    def test_normalized_crop_is_applied_after_orientation_before_submission(self):
        image = Image.new('RGB', (8, 6), 'red')
        for y in range(6):
            for x in range(4, 8):
                image.putpixel((x, y), (0, 0, 255))
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 degrees clockwise when displayed
        exif[270] = 'private metadata'
        buffer = io.BytesIO()
        image.save(buffer, 'PNG', exif=exif)
        with Image.open(io.BytesIO(buffer.getvalue())) as source:
            oriented = ImageOps.exif_transpose(source).convert('RGB')
        expected = oriented.crop((0, 0, 6, 4))

        def query(**kwargs):
            payload = kwargs['dataframe_records'][0]['image_base64']
            with Image.open(io.BytesIO(base64.b64decode(payload))) as submitted:
                self.assertEqual(submitted.size, (6, 4))
                self.assertEqual(submitted.convert('RGB').tobytes(), expected.tobytes())
                self.assertFalse(submitted.getexif())
            return SimpleNamespace(predictions=[{'airline': 'delta', 'confidence': 0.8,
                                                  'probabilities_json': '{"delta": 0.8}'}])

        fake = SimpleNamespace(serving_endpoints=SimpleNamespace(query=query))
        self.assertEqual(identify_tail(buffer.getvalue(), client=fake,
                                       crop_box=(0, 0, 1, 0.5))['airline'], 'delta')

    def test_invalid_or_tiny_crop_is_rejected_before_query(self):
        buffer = io.BytesIO()
        Image.new('RGB', (10, 10)).save(buffer, 'PNG')
        def query(**_):
            self.fail('invalid crop must not invoke serving')
        fake = SimpleNamespace(serving_endpoints=SimpleNamespace(query=query))
        for box in ((0, 0, 0, 1), (-0.1, 0, 1, 1), (0, 0, 1.1, 1),
                    (0, 0, 1, float('nan')), (False, 0, 1, 1), (0, 0, .01, .01)):
            with self.subTest(box=box), self.assertRaises(ValueError):
                identify_tail(buffer.getvalue(), client=fake, crop_box=box)


class ClientHelperTests(unittest.TestCase):
    @staticmethod
    def fake_client(sent):
        def query(**kwargs):
            sent.append(kwargs)
            return SimpleNamespace(predictions=[{'airline': 'united', 'confidence': 0.7,
                                                 'probabilities_json': '{"united": 0.7}'}])
        return SimpleNamespace(serving_endpoints=SimpleNamespace(query=query))

    @staticmethod
    def png(image, **kwargs):
        buffer = io.BytesIO()
        image.save(buffer, 'PNG', **kwargs)
        return buffer.getvalue()

    def test_crop_pixels_rounds_outward(self):
        self.assertEqual(client.crop_pixels((10, 10), (0.1, 0.1, 0.55, 0.55)), (1, 1, 6, 6))
        self.assertEqual(client.crop_pixels((10, 10), (0.15, 0.25, 0.35, 0.45)), (1, 2, 4, 5))
        self.assertEqual(client.crop_pixels((10, 10), [0, 0, 1, 1]), (0, 0, 10, 10))

    def test_tail_crop_max_side_preserves_aspect_ratio(self):
        wide = Image.new('RGB', (3000, 1000))
        self.assertEqual(client.tail_crop(wide, None, 1536).size, (1536, 512))
        self.assertEqual(client.tail_crop(Image.new('RGB', (1000, 3000)), None, 1536).size, (512, 1536))
        self.assertEqual(client.tail_crop(wide, (0, 0, 0.5, 1), 1536).size, (1500, 1000))
        self.assertEqual(client.tail_crop(wide, (0, 0, 0.8, 0.9), 1536).size, (1536, 576))
        self.assertEqual(client.tail_crop(wide, None, 3000).size, (3000, 1000))
        self.assertEqual(client.tail_crop(wide).size, (3000, 1000))

    def test_prepare_tail_png_is_exactly_what_identify_tail_sends(self):
        image = Image.new('RGB', (12, 9), (1, 2, 3))
        image.putpixel((3, 4), (250, 0, 0))
        exif = Image.Exif()
        exif[274] = 8
        data = self.png(image, exif=exif)
        for box in (None, (0.25, 0, 1, 0.5)):
            with self.subTest(box=box):
                sent = []
                identify_tail(data, client=self.fake_client(sent), crop_box=box)
                self.assertEqual(base64.b64decode(sent[0]['dataframe_records'][0]['image_base64']),
                                 client.prepare_tail_png(data, box))

    def test_query_tail_png_endpoint_from_env_or_argument(self):
        with mock.patch.dict(os.environ, {'SERVING_ENDPOINT': 'env-endpoint'}, clear=False):
            sent = []
            result = client.query_tail_png(b'png', self.fake_client(sent))
            self.assertEqual(sent[0]['name'], 'env-endpoint')
            self.assertEqual(sent[0]['dataframe_records'], [{'image_base64': base64.b64encode(b'png').decode()}])
            self.assertEqual(result, {'airline': 'united', 'confidence': 0.7, 'probabilities': {'united': 0.7}})
            sent = []
            client.query_tail_png(b'png', self.fake_client(sent), endpoint='explicit-endpoint')
            self.assertEqual(sent[0]['name'], 'explicit-endpoint')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SERVING_ENDPOINT', None)
            sent = []
            client.query_tail_png(b'png', self.fake_client(sent))
            self.assertEqual(sent[0]['name'], client.DEFAULT_ENDPOINT)

    def test_oriented_rgb_rejects_empty_bytes(self):
        with self.assertRaises(ValueError):
            client.oriented_rgb(b'')

    def test_exif_orientation_defaults_to_one(self):
        self.assertEqual(client.exif_orientation(Image.new('RGB', (2, 2))), 1)
        with Image.open(io.BytesIO(self.png(Image.new('RGB', (2, 2))))) as plain:
            self.assertEqual(client.exif_orientation(plain), 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for raw in (b'garbage', b'Exif\x00\x00garbage', b'MM\x00*\x00\x00\x00\x08\xff\xff'):
                with self.subTest(raw=raw):
                    image = Image.new('RGB', (2, 2))
                    image.info['exif'] = raw
                    self.assertEqual(client.exif_orientation(image), 1)
        exif = Image.Exif()
        exif[274] = 9  # outside the valid 1-8 range
        with Image.open(io.BytesIO(self.png(Image.new('RGB', (2, 2)), exif=exif))) as invalid:
            invalid.load()
            self.assertEqual(client.exif_orientation(invalid), 1)


if __name__ == "__main__":
    unittest.main()
