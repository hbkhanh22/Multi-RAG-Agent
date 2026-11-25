"""
Text-to-image on NVIDIA NIM (FLUX). Prompts must already be safety-checked: the hosted
models do not reliably filter content themselves (see src/agents/image_agent.py).
"""

import base64
import io
import logging
import time
from dataclasses import dataclass, field

import httpx

from config.settings import settings
from src import runtime
from src.observability import metrics

logger = logging.getLogger(__name__)

SIZES = {"square": (1024, 1024), "landscape": (1344, 768), "portrait": (768, 1344)}


class ImageUnavailable(Exception):
    """No image model answered."""


class ContentFiltered(Exception):
    """The provider's own filter refused the prompt."""


@dataclass
class GeneratedImage:
    data: bytes = field(repr=False)
    media_type: str = "image/jpeg"
    width: int = 0
    height: int = 0
    model: str = ""


def _decode(b64: str, model: str) -> GeneratedImage:
    from PIL import Image

    data = base64.b64decode(b64 or "")
    img = Image.open(io.BytesIO(data), formats=["JPEG", "PNG", "WEBP"])
    img.load()                                  # truncated data raises here, not in the browser
    return GeneratedImage(data, Image.MIME[img.format], img.width, img.height, model)


def generate(prompt: str, orientation: str = "square") -> GeneratedImage:
    """One image; tries IMAGE_MODEL, then IMAGE_FALLBACK_MODEL."""
    width, height = SIZES.get(orientation, SIZES["square"])
    for model in filter(None, dict.fromkeys((settings.image_model, settings.image_fallback_model))):
        runtime.check_cancelled()
        started = time.time()
        try:
            resp = httpx.post(
                f"{settings.image_base_url.rstrip('/')}/{model}",
                headers={"Authorization": f"Bearer {settings.nvidia_api_key}", "Accept": "application/json"},
                json={"prompt": prompt, "width": width, "height": height},
                timeout=settings.image_timeout,
            )
            resp.raise_for_status()
            artifact = resp.json()["artifacts"][0]
            if artifact.get("finishReason") == "CONTENT_FILTERED":
                raise ContentFiltered(model)
            image = _decode(artifact.get("base64"), model)
            ctx = runtime.current()
            if ctx is not None:
                ctx.add_usage(model, 0, 0)
            metrics.record_llm_call(model, "Image Generator", 0, 0, time.time() - started)
            return image
        except ContentFiltered:
            raise
        except Exception as e:
            logger.warning(f"Image model {model} failed: {type(e).__name__}: {str(e)[:200]}")
    raise ImageUnavailable("no image model answered")
