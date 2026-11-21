from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from .data.vhd11k import load_vhd11k
from .runner import load_clean

PINNED_REVISIONS: dict[str, str] = {
    "openai/clip-vit-base-patch16": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "google/siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
}


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated embed")
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--revision", default=None, help="HuggingFace commit SHA (defaults to pinned SHA for known models)")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument(
        "--splits", type=Path, default=None, help="optional splits JSON to restrict extraction (e.g. train,val)"
    )
    ap.add_argument(
        "--split-list", default=None, help="comma-separated splits to embed when --splits is given (e.g. train,val)"
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args(argv)

    samples, _ = load_vhd11k(a.dataset.expanduser())
    if a.splits is not None and a.split_list:
        sp = json.loads(a.splits.expanduser().read_text(encoding="utf-8"))
        wanted = set()
        for part_name in [p.strip() for p in a.split_list.split(",") if p.strip()]:
            wanted.update(sp[part_name])
        samples = [s for s in samples if s.sample_id in wanted]
        missing = sorted(wanted - {s.sample_id for s in samples})
        if missing:
            raise SystemExit(f"ERROR: {len(missing)} requested sample(s) missing from dataset (first: {missing[0]}).")

    if a.limit:
        samples = samples[: a.limit]

    revision = a.revision if a.revision is not None else PINNED_REVISIONS.get(a.model)
    load_kwargs = {"revision": revision} if revision else {}
    dev = device()
    print(f"device: {dev} | model: {a.model} | revision: {revision or 'unpinned'} | images: {len(samples)}")
    model = AutoModel.from_pretrained(a.model, **load_kwargs).to(dev).eval()
    proc = AutoProcessor.from_pretrained(a.model, **load_kwargs)

    feats, ms = [], []
    with torch.inference_mode():
        for i in range(0, len(samples), a.batch_size):
            batch = samples[i : i + a.batch_size]
            imgs = [load_clean(s.path) for s in batch]  # pixels only, as always
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
            sync(dev)
            t0 = time.perf_counter()
            out = model.get_image_features(pixel_values=px)
            f = out if torch.is_tensor(out) else out.pooler_output
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            sync(dev)
            dt = (time.perf_counter() - t0) * 1000 / len(batch)  # model forward pass only
            feats.append(f.cpu().numpy())
            ms += [dt] * len(batch)
            if (i // a.batch_size) % 20 == 0:
                print(f"{i}/{len(samples)}", flush=True)

    suffix = f"-n{a.limit}" if a.limit else (f"-{a.split_list.replace(',', '_')}" if a.split_list else "")
    name = a.model.replace("/", "_") + suffix
    out_path = a.output or Path("embeddings") / f"{name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        ids=np.array([s.sample_id for s in samples]),
        feats=np.concatenate(feats),
        ms=np.array(ms, dtype=np.float32),
        model_id=np.array(a.model),
        revision=np.array(revision or ""),
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
