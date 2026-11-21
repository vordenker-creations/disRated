from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .vhd11k import Sample


def make_splits(samples: list[Sample], seed: int = 0, path: Path | None = None) -> dict[str, list[str]]:
    """80/10/10 train/val/test, stratified on source x label, grouped so duplicates never straddle splits.
    If `path` exists it is loaded as-is. Splits are written once and never regenerated."""
    if path is not None and path.exists():
        return json.loads(path.read_text())

    y = [f"{s.source}|{int(s.harmful)}" for s in samples]
    groups = [s.group for s in samples]
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [idx for _, idx in cv.split(np.zeros(len(samples)), y, groups)]

    def ids(idxs) -> list[str]:
        return sorted(samples[i].sample_id for i in idxs)

    out = {
        "test": ids(folds[0]),
        "val": ids(folds[1]),
        "train": ids(i for f in folds[2:] for i in f),
    }
    if path is not None:
        path.write_text(json.dumps(out, indent=1))
    return out
