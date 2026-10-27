"""Server-side example for a Databricks App using its managed SDK credentials."""
import base64
import io
import json
import math
import os

from PIL import Image, ImageOps


def identify_tail(image_bytes: bytes, client=None, *, crop_box=None) -> dict:
    """Send a tail crop; optional normalized box refers to the EXIF-oriented image.

    crop_box is (left, top, right, bottom), each coordinate in [0, 1]. If it
    is omitted, image_bytes must already contain a user-selected tail crop.
    """
    if not image_bytes:
        raise ValueError("Provide a tail-focused image")
    with Image.open(io.BytesIO(image_bytes)) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
        if crop_box is not None:
            if (not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in crop_box)):
                raise ValueError("crop_box must contain four finite normalized coordinates")
            left, top, right, bottom = crop_box
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("crop_box must be within the image and have positive area")
            box = (math.floor(left * rgb.width), math.floor(top * rgb.height),
                   math.ceil(right * rgb.width), math.ceil(bottom * rgb.height))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                raise ValueError("crop_box must cover at least two pixels in each dimension")
            rgb = rgb.crop(box)
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    if client is None:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
    endpoint = os.environ.get("SERVING_ENDPOINT", "airplane-tail-identifier")
    response = client.serving_endpoints.query(
        name=endpoint,
        dataframe_records=[{"image_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}],
    )
    prediction = response.predictions[0]
    return {"airline": prediction["airline"], "confidence": prediction["confidence"],
            "probabilities": json.loads(prediction["probabilities_json"])}
