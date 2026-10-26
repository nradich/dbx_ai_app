"""Contract/privacy check using a fake serving transport; no network calls."""
import base64
import io
import json
from types import SimpleNamespace
import unittest

from PIL import Image

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


if __name__ == "__main__":
    unittest.main()
