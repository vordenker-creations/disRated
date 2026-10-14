from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

csv.field_size_limit(2**31 - 1)  # debate columns are long

ANN_FILE = "harmful_image_10000_ann.csv"
IMG_DIR = "all_10000_evenHarmfulUnharmful"

_SDXL = re.compile(r"^sdxl_extend_by_llama3_(harmful|unharmful)_(\d+)_seed(\d+)\.png$")
_CONCEPT = re.compile(r"^(.+)_(\d+)_(\d+)\.\w+$")


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str            # the CSV imagePath; unique
    path: Path
    harmful: bool             # ground truth: the CSV `decision`
    harm_type: str            # free-text VLM reason; error analysis only
    source: Literal["sdxl", "concept"]
    concept: str | None = None    # one of ~45 tags, concept-named half only
    intent: bool | None = None    # what the filename claims (sdxl only). NEVER show to a model.
    group: str                # split unit, so near-duplicates stay together


def parse_name(name: str) -> dict:
    if m := _SDXL.match(name):
        intent = m.group(1) == "harmful"
        # same prompt index across seeds -> same group
        return dict(source="sdxl", intent=intent, group=f"sdxl:{m.group(1)}:{m.group(2)}")
    if m := _CONCEPT.match(name):
        # no known duplicate structure here, so each image is its own group
        return dict(source="concept", concept=m.group(1), group=name)
    raise ValueError(f"unrecognised filename: {name}")


def load_vhd11k(root: Path) -> tuple[list[Sample], list[tuple[str, str]]]:
    """Returns (samples, problems). Bad rows are logged and skipped, never fatal."""
    root = Path(root)
    samples: list[Sample] = []
    problems: list[tuple[str, str]] = []
    with (root / ANN_FILE).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["imagePath"]
            path = root / IMG_DIR / name
            try:
                if row["decision"] not in ("yes", "no"):
                    raise ValueError(f"bad decision {row['decision']!r}")
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError("missing or empty file")
                samples.append(
                    Sample(
                        sample_id=name,
                        path=path,
                        harmful=row["decision"] == "yes",
                        harm_type=row["harmfulType"],
                        **parse_name(name),
                    )
                )
            except ValueError as e:
                problems.append((name, str(e)))
    return samples, problems