from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from .data.vhd11k import load_vhd11k
from .runner import load_clean


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


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.embed")
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    if a.limit:
        samples = samples[: a.limit]
    dev = device()
    print("device:", dev, "| images:", len(samples))
    model = CLIPModel.from_pretrained(a.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.model)

    feats, ms = [], []
    with torch.no_grad():
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
            dt = (time.perf_counter() - t0) * 1000 / len(batch)  # model time only
            feats.append(f.cpu().numpy())
            ms += [dt] * len(batch)
            if (i // a.batch_size) % 20 == 0:
                print(f"{i}/{len(samples)}", flush=True)

    name = a.model.replace("/", "_") + (f"-n{a.limit}" if a.limit else "")
    out_path = a.output or Path("embeddings") / f"{name}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, ids=np.array([s.sample_id for s in samples]),
             feats=np.concatenate(feats), ms=np.array(ms, dtype=np.float32))
    print("wrote", out_path)


if __name__ == "__main__":
    main()