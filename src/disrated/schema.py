from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModeratorOutput(BaseModel):
    """What a moderator returns for one image. No ids, no paths."""

    score: float | None = Field(default=None, ge=0, le=1)  # P(harmful); None if unusable
    verdict: bool | None = None  # optional hard call; scorer derives it from score if absent
    refused: bool = False
    raw_output: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Prediction(ModeratorOutput):
    sample_id: str
    latency_ms: float  # model time only, batch time / batch size; image decoding excluded
    error: str | None = None
