from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.vhd11k import load_vhd11k
from .runner import run


def build(name: str):
    if name == "random":
        from .models.dummy import Random
        return Random(0)
    if name.startswith("constant-"):
        from .models.dummy import Constant
        return Constant(float(name.split("-")[1]))
    if name == "nudenet":
        from .models.nudenet_mod import NudeNet
        return NudeNet()
    raise SystemExit(f"unknown model: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    if a.split != "all":
        ids = set(json.loads(a.splits.read_text())[a.split])
        samples = [s for s in samples if s.sample_id in ids]
    out = a.output or Path(f"results/{a.model}-{a.split}.jsonl")
    print(run(build(a.model), samples, out, a.batch_size, a.limit))
    print("wrote", out)


if __name__ == "__main__":
    main()