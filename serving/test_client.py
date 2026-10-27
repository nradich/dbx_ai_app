"""Contract/privacy check using a fake serving transport; no network calls."""
import base64
import io
import json
from types import SimpleNamespace
import unittest

from PIL import Image, ImageOps

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


if __name__ == "__main__":
    unittest.main()
