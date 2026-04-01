import base64
import io
import re
import json
from PIL import Image


def image_bytes_to_base64(image_bytes: bytes, format: str = "PNG") -> str:
    """Convert raw image bytes to a base64-encoded string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def validate_pdf_header(file_bytes: bytes) -> bool:
    """Quick check that file starts with PDF magic bytes."""
    return file_bytes[:5] == b"%PDF-"


def resize_image_if_needed(
    image_bytes: bytes, max_dimension: int = 2048
) -> bytes:
    """Resize image to fit within max_dimension while keeping aspect ratio.
    Reduces token cost for vision APIs."""
    img = Image.open(io.BytesIO(image_bytes))
    if max(img.size) <= max_dimension:
        return image_bytes

    ratio = max_dimension / max(img.size)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def safe_parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences.
    LLMs sometimes wrap JSON in ```json ... ``` blocks."""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)
