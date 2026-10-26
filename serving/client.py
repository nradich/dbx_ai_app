"""Server-side example for a Databricks App using its managed SDK credentials."""
import base64
import io
import json
import os

from databricks.sdk import WorkspaceClient
from PIL import Image, ImageOps


def identify_tail(image_bytes: bytes, client: WorkspaceClient | None = None) -> dict:
    """Accept the user-selected tail crop, not an arbitrary full-plane image."""
    if not image_bytes:
        raise ValueError("Provide a tail-focused image")
    with Image.open(io.BytesIO(image_bytes)) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
        clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG")
    client = client or WorkspaceClient()
    endpoint = os.environ.get("SERVING_ENDPOINT", "airplane-tail-identifier")
    response = client.serving_endpoints.query(
        name=endpoint,
        dataframe_records=[{"image_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}],
    )
    prediction = response.predictions[0]
    return {"airline": prediction["airline"], "confidence": prediction["confidence"],
            "probabilities": json.loads(prediction["probabilities_json"])}
