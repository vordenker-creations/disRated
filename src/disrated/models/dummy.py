from __future__ import annotations

import hashlib

from PIL import Image

from ..schema import ModeratorOutput
from .base import BaseModerator


class Constant(BaseModerator):
    """Constant(0.0) = always safe, Constant(1.0) = always harmful."""

    def __init__(self, score: float):
        self.score = score
        self.name = f"constant-{score:g}"

    def config(self):
        return {"score": self.score}

    def predict(self, images: list[Image.Image]):
        return [ModeratorOutput(score=self.score, verdict=self.score >= 0.5) for _ in images]


class Random(BaseModerator):
    """Deterministic per image (hash of pixels + seed), so order and resumes don't matter."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def config(self):
        return {"seed": self.seed}

    def predict(self, images: list[Image.Image]):
        out = []
        for im in images:
            h = hashlib.sha256(str(self.seed).encode() + im.tobytes()).digest()
            s = int.from_bytes(h[:8], "big") / 2**64
            out.append(ModeratorOutput(score=s, verdict=s >= 0.5))
        return out