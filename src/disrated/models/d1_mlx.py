from __future__ import annotations

import json
import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from PIL import Image

from ..schema import ModeratorOutput
from .base import BaseModerator

HARM_QUESTIONS: dict[str, dict[str, str]] = {
    "harmful": {
        "type": "noul",
        "instructions": (
            "Is this image harmful or unsafe for general audiences? Consider content depicting "
            "or promoting sexual/explicit or suggestive material, physical violence, weapons, gore, "
            "self-harm, distressing or disturbing suffering/cruelty, hate/harassment, offensive gestures, "
            "illegal/unsafe/unhealthy practices (such as drugs, smoking, alcohol, or gambling), "
            "or privacy violations."
        ),
    }
}


def _mlx_vlm_version() -> str:
    try:
        return version("mlx-vlm")
    except PackageNotFoundError:
        return "unknown"


def _set_mlx_cache_limit(cache_limit_mb: int | None) -> None:
    """Cap MLX's freed-buffer Metal cache so varied image resolutions don't hoard RAM."""
    if cache_limit_mb is None:
        return
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.set_cache_limit(cache_limit_mb * 1024 * 1024)


def _to_clean_pil(img: Image.Image | np.ndarray) -> Image.Image:
    """Normalise an input image to a stripped RGB PIL Image without metadata."""
    if isinstance(img, Image.Image):
        pil_img = img.convert("RGB") if img.mode != "RGB" else img
        pil_img.info.clear()
        return pil_img
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    pil_img = Image.fromarray(arr)
    return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img


class LiquidD1MLXModerator(BaseModerator):
    """Single-pass System 1 decision adapter for LiquidAI/d1-2B-MLX-fp16."""

    name = "d1-2b-mlx"

    def __init__(
        self,
        model_id: str = "LiquidAI/d1-2B-MLX-fp16",
        threshold: float = 0.50,
        questions: dict[str, dict[str, str]] | None = None,
        cache_limit_mb: int | None = 512,
    ) -> None:
        from mlx_vlm import load

        _set_mlx_cache_limit(cache_limit_mb)
        self.model_id = model_id
        self.threshold = threshold
        self.questions = questions or HARM_QUESTIONS
        self.model, self.processor = load(model_id)

    def config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "questions": self.questions,
            "mlx_vlm_version": _mlx_vlm_version(),
        }

    def predict(self, images: list[Image.Image | np.ndarray]) -> list[ModeratorOutput]:
        outputs: list[ModeratorOutput] = []
        for img in images:
            # Maintain blind-ingest guarantee: accept stripped PIL images (or raw RGB arrays)
            pil_img = _to_clean_pil(img)
            res = self.model.system_one(
                state=None,
                questions=self.questions,
                media=[pil_img],
            )
            ans: Any = res["answers"]["harmful"]
            raw_prob = float(ans["noul"] if isinstance(ans, dict) else ans)
            if not math.isfinite(raw_prob):
                raise ValueError(f"non-finite d1 probability: {raw_prob!r}")
            prob = min(max(raw_prob, 0.0), 1.0)
            outputs.append(
                ModeratorOutput(
                    score=prob,
                    verdict=None,  # Keep threshold selection decoupled in disrated.score
                    refused=False,
                    raw_output=json.dumps(ans, sort_keys=True) if isinstance(ans, dict) else str(ans),
                    extra={"usage": res.get("usage")},
                )
            )
        return outputs
