"""
Reading images with a vision-language model on NVIDIA NIM (photos, screenshots, scans).

Plain HTTP instead of ChatNVIDIA: base64 images must not end up in traces.
"""

import base64
import io
import logging
import threading
import time

import httpx

from config.settings import settings
from src import runtime
from src.guardrails.injection import DATA_RULE
from src.observability import metrics

logger = logging.getLogger(__name__)

MAX_PIXELS = 60_000_000      # refuse decompression bombs
MAX_SIDE = 1600              # enough for small print on an A4 page
PDFIUM_LOCK = threading.Lock()

TRANSCRIBE = (
    "Read this image.\n"
    "1. Transcribe ALL visible text exactly, in its original language with all diacritics, in reading "
    "order. Write tables as markdown tables. Never guess text you cannot read: write [không đọc được].\n"
    "2. Then add a line starting with 'Mô tả:' that describes in Vietnamese what the image shows besides "
    "the text (objects, people, colours, chart values and trends). Skip this line for a plain text page.\n"
    "Output only that."
)
OBSERVE = (
    "Look at the image and write down, in the language of the question, every visible fact that helps "
    "answer it (including exact text and numbers). State clearly when the image does not show the "
    "answer. Do not use outside knowledge.\nQuestion: "
)


class VisionUnavailable(Exception):
    """No vision model could read the image."""


def to_jpeg(data: bytes) -> bytes:
    """Downscaled, upright JPEG without metadata (drops EXIF such as GPS). ValueError if
    the bytes are not an acceptable image."""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data), formats=["PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"])
        if img.width * img.height > MAX_PIXELS:
            raise ValueError("image has too many pixels")
        return pil_to_jpeg(ImageOps.exif_transpose(img))
    except ValueError:
        raise
    except Exception as e:   # Pillow raises many types on broken files
        raise ValueError(f"not a readable image ({type(e).__name__})") from e


def pil_to_jpeg(img) -> bytes:
    from PIL import Image

    if img.mode in ("RGBA", "LA", "P"):      # transparent background => white, not black
        img = img.convert("RGBA")
        canvas = Image.new("RGB", img.size, "white")
        canvas.paste(img, mask=img.getchannel("A"))
        img = canvas
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def ask(jpeg: bytes, prompt: str) -> str:
    """Ask the vision model about one image; falls back to the second model."""
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    for model in filter(None, dict.fromkeys((settings.vision_llm_model, settings.vision_fallback_model))):
        runtime.check_cancelled()
        started = time.time()
        try:
            resp = httpx.post(
                f"{settings.nvidia_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.nvidia_api_key}", "Accept": "application/json"},
                json={
                    "model": model, "temperature": 0, "max_tokens": settings.vision_max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [
                        {"role": "system", "content": DATA_RULE},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": url}},
                        ]},
                    ],
                },
                timeout=settings.vision_timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            text = (body["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty answer")
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or len(text) // 4)
            ctx = runtime.current()
            if ctx is not None:
                ctx.add_usage(model, prompt_tokens, completion, not usage)
            metrics.record_llm_call(model, "Vision Reader", prompt_tokens, completion, time.time() - started)
            return text
        except Exception as e:
            logger.warning(f"Vision model {model} failed: {type(e).__name__}: {str(e)[:200]}")
    raise VisionUnavailable("no vision model answered")


def transcribe(jpeg: bytes) -> str:
    return ask(jpeg, TRANSCRIBE)


def observe(jpeg: bytes, question: str) -> str:
    return ask(jpeg, OBSERVE + question)
