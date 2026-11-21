from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

csv.field_size_limit(2**31 - 1)  # debate columns are long

ANN_FILE = "harmful_image_10000_ann.csv"
IMG_DIR = "all_10000_evenHarmfulUnharmful"

_SDXL = re.compile(r"^sdxl_extend_by_llama3_(harmful|unharmful)_(\d+)_seed(\d+)\.png$")
_CONCEPT = re.compile(r"^(.+)_(\d+)_(\d+)\.\w+$")

# Pre-registered partition of VHD11K's 45 concept tags (defined by tag semantics prior to Phase 4).
CONCRETE_HARM_CONCEPTS: frozenset[str] = frozenset(
    {
        "animal_abuse",
        "animal_genitalia_and_human",
        "animal_genitalia_only",
        "animated_animal_genitalia",
        "animated_blood",
        "animated_corpse",
        "animated_gun",
        "bloody",
        "fetish_extremes",
        "gun",
        "hanging",
        "human_corpse",
        "knife_in_hand",
        "nazi",
        "noose",
        "nudity",
        "sex_toy",
        "sexual_activity",
        "shocking",
        "terrorist",
        "theft",
        "violence",
    }
)

CONTEXTUAL_NORMATIVE_CONCEPTS: frozenset[str] = frozenset(
    {
        "alcohol",
        "animated_alcohol",
        "bulge",
        "cheating",
        "cleavage",
        "contamination",
        "cruelty_in_the_animal_kingdom",
        "culinary_knife_in_hand",
        "emaciated_body",
        "gambling",
        "harrassment",
        "inequality",
        "injectables",
        "lack_of_self_control",
        "marijuana",
        "obesity",
        "obscene_gesture",
        "pills",
        "religious_icon",
        "rudeness",
        "sexual_intent",
        "smoking",
        "toy",
    }
)


def concept_taxonomy(concept: str | None) -> str | None:
    """Map a VHD11K concept tag to its pre-registered concrete vs. contextual_normative bucket."""
    if concept is None:
        return None
    if concept in CONCRETE_HARM_CONCEPTS:
        return "concrete_harm"
    if concept in CONTEXTUAL_NORMATIVE_CONCEPTS:
        return "contextual_normative"
    return "other"


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str  # the CSV imagePath; unique
    path: Path
    harmful: bool  # ground truth: the CSV `decision`
    harm_type: str  # free-text VLM reason; error analysis only
    source: Literal["sdxl", "concept"]
    concept: str | None = None  # one of ~45 tags, concept-named half only
    intent: bool | None = None  # what the filename claims (sdxl only). NEVER show to a model.
    group: str  # split unit, so near-duplicates stay together


def parse_name(name: str) -> dict[str, Any]:
    if m := _SDXL.match(name):
        intent = m.group(1) == "harmful"
        # same prompt index across seeds -> same group
        return {"source": "sdxl", "intent": intent, "group": f"sdxl:{m.group(1)}:{m.group(2)}"}
    if m := _CONCEPT.match(name):
        # no known duplicate structure here, so each image is its own group
        return {"source": "concept", "concept": m.group(1), "group": name}
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
