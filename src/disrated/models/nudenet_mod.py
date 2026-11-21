from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

import numpy as np
from PIL import Image

from ..schema import ModeratorOutput
from .base import BaseModerator

# Which detector classes count towards the score. A choice, not a fact.
EXPLICIT = (
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
)


class NudeNet(BaseModerator):
    name = "nudenet"

    def __init__(self, classes: tuple[str, ...] = EXPLICIT):
        from nudenet import NudeDetector  # lazy: slow import, only when actually used

        self.classes = tuple(classes)
        self.detector = NudeDetector()

    def config(self) -> dict[str, Any]:
        return {"classes": list(self.classes), "nudenet_version": version("nudenet")}

    def predict(self, images: Sequence[Image.Image]) -> list[ModeratorOutput]:
        outs = []
        for im in images:
            arr = np.asarray(im)[:, :, ::-1].copy()  # PIL is RGB, cv2 (which NudeNet uses) is BGR
            dets = self.detector.detect(arr)
            keep = [{"class": d["class"], "score": round(float(d["score"]), 4)} for d in dets]
            score = max((d["score"] for d in keep if d["class"] in self.classes), default=0.0)
            outs.append(ModeratorOutput(score=score, extra={"detections": keep}))
        return outs
