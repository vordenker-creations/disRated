from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from ..schema import ModeratorOutput


class BaseModerator(ABC):
    name: str

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """JSON-serialisable settings that change outputs (model id, prompt, ...).
        Stops a resumed run silently mixing configs."""

    @abstractmethod
    def predict(self, images: list[Image.Image]) -> list[ModeratorOutput]:
        """One output per image, same order."""