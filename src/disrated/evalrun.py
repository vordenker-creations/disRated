from __future__ import annotations

import argparse
import hashlib
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
    if name == "d1-2b-mlx":
        from .models.d1_mlx import LiquidD1MLXModerator

        return LiquidD1MLXModerator()
    if name == "qwen3-vl-8b-fp8":
        from .models.qwen3_vl import Qwen3VLModerator

        return Qwen3VLModerator()
    raise SystemExit(f"unknown model: {name}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated evalrun")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"), help="path to the active splits JSON file")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args(argv)

    samples, _ = load_vhd11k(a.dataset.expanduser())
    splits_sha256 = None
    if a.splits is not None:
        splits_path = a.splits.expanduser()
        if splits_path.is_file():
            splits_raw = splits_path.read_text(encoding="utf-8")
            splits_sha256 = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
            splits_data = json.loads(splits_raw)
            if a.split != "all":
                if a.split not in splits_data:
                    raise KeyError(f"split '{a.split}' not in {splits_path}")
                ids = set(splits_data[a.split])
                samples = [s for s in samples if s.sample_id in ids]
            else:
                ids = set().union(*splits_data.values())
            missing_ids = sorted(ids - {s.sample_id for s in samples})
            if missing_ids:
                raise SystemExit(
                    f"ERROR: {len(missing_ids)} sample(s) in split '{a.split}' could not be loaded "
                    f"from {a.dataset} (first: {missing_ids[0]})."
                )
        elif a.split != "all":
            raise FileNotFoundError(f"splits file '{splits_path}' required for split '{a.split}' not found")
    elif a.split != "all":
        raise ValueError("--splits required when --split is not 'all'")

    extra = {"splits_sha256": splits_sha256} if splits_sha256 else None
    run_meta = {"split": a.split, "splits_sha256": splits_sha256}
    out = a.output or Path(f"results/{a.model}-{a.split}.jsonl")
    print(run(build(a.model), samples, out, a.batch_size, a.limit, extra=extra, run_meta=run_meta))
    print("wrote", out)


if __name__ == "__main__":
    main()
