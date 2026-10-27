"""Server-side example for a Databricks App using its managed SDK credentials.

`app/serving_client.py` must remain a byte-identical copy of this file; a
repository test enforces that, because the app deploys only its own folder.
"""
import base64
import io
import json
import math
import os

from PIL import Image

DEFAULT_ENDPOINT = "airplane-tail-identifier"
EXIF_ORIENTATION_TAG = 0x0112
_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
_SIXTEEN_BIT_GRAY = frozenset({"I", "I;16", "I;16L", "I;16B", "I;16N"})


def exif_orientation(image: Image.Image) -> int:
    """Orientation from the image's EXIF block only (1 if absent or unreadable).

    Browsers ignore XMP-only orientation, so unlike ImageOps.exif_transpose this
    does not fall back to XMP; the crop box then refers to what the user saw.
    """
    raw = image.info.get("exif")
    if not raw:
        return 1
    exif = Image.Exif()
    try:
        exif.load(raw)
        value = exif.get(EXIF_ORIENTATION_TAG, 1)
    except Exception:
        return 1
    return value if value in _TRANSPOSE else 1


def oriented_rgb(image_bytes: bytes) -> Image.Image:
    """Decode, apply EXIF orientation and return a metadata-free RGB image."""
    if not image_bytes:
        raise ValueError("Provide a tail-focused image")
    with Image.open(io.BytesIO(image_bytes)) as original:
        original.load()  # PNG eXIf chunks can follow the image data
        method = _TRANSPOSE.get(exif_orientation(original))
        image = original if method is None else original.transpose(method)
        if image.mode in _SIXTEEN_BIT_GRAY:
            # convert("RGB") would clip 16-bit values; scale 0-65535 to 0-255 first.
            image = image.convert("I").point(lambda value: value / 257).convert("L")
        rgb = image.convert("RGB")
    rgb.info = {}
    return rgb


def crop_pixels(size, crop_box):
    """Validate a normalized (left, top, right, bottom) box; return outward-rounded pixels."""
    if (not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in crop_box)):
        raise ValueError("crop_box must contain four finite normalized coordinates")
    left, top, right, bottom = crop_box
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("crop_box must be within the image and have positive area")
    width, height = size
    box = (math.floor(left * width), math.floor(top * height),
           math.ceil(right * width), math.ceil(bottom * height))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError("crop_box must cover at least two pixels in each dimension")
    return box


def png_bytes(image: Image.Image) -> bytes:
    """Encode as PNG without carrying over any metadata."""
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    return buffer.getvalue()


def tail_crop(rgb: Image.Image, crop_box=None, max_side=None) -> Image.Image:
    """Apply the optional normalized crop, then optionally limit the longest side."""
    if crop_box is not None:
        rgb = rgb.crop(crop_pixels(rgb.size, crop_box))
    if max_side is not None and max(rgb.size) > max_side:
        scale = max_side / max(rgb.size)
        rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                         Image.Resampling.LANCZOS)
    return rgb


def prepare_tail_png(image_bytes: bytes, crop_box=None, max_side=None) -> bytes:
    """Return the exact metadata-free PNG bytes that identify_tail transmits."""
    return png_bytes(tail_crop(oriented_rgb(image_bytes), crop_box, max_side))


def query_tail_png(png: bytes, client=None, endpoint=None) -> dict:
    """Query the endpoint with already prepared PNG bytes."""
    if client is None:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
    endpoint = endpoint or os.environ.get("SERVING_ENDPOINT", DEFAULT_ENDPOINT)
    response = client.serving_endpoints.query(
        name=endpoint,
        dataframe_records=[{"image_base64": base64.b64encode(png).decode("ascii")}],
    )
    prediction = response.predictions[0]
    return {"airline": prediction["airline"], "confidence": prediction["confidence"],
            "probabilities": json.loads(prediction["probabilities_json"])}


def identify_tail(image_bytes: bytes, client=None, *, crop_box=None) -> dict:
    """Send a tail crop; optional normalized box refers to the EXIF-oriented image.

    crop_box is (left, top, right, bottom), each coordinate in [0, 1]. If it
    is omitted, image_bytes must already contain a user-selected tail crop.
    """
    return query_tail_png(prepare_tail_png(image_bytes, crop_box), client)
