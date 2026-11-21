from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .data.vhd11k import load_vhd11k


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.audit_clip")
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings/openai_clip-vit-base-patch16.npz"))
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.json"))
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}

    z = np.load(a.embeddings, allow_pickle=False)
    id_to_idx = {sid: idx for idx, sid in enumerate(z["ids"].tolist())}
    feats = z["feats"]  # shape (10000, 512), already L2 normalized

    splits = json.loads(a.splits.read_text(encoding="utf-8"))
    tr_idx = np.array([id_to_idx[sid] for sid in splits["train"]])
    te_idx = np.array([id_to_idx[sid] for sid in splits["test"]])

    # Test-to-Train max cosine similarities
    X_tr = feats[tr_idx]
    X_te = feats[te_idx]
    sims = np.matmul(X_te, X_tr.T)
    max_sims = np.max(sims, axis=1)

    print("\n=== TEST -> TRAIN MAX COSINE SIMILARITY ===")
    qs = [0, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(max_sims, qs)
    for q, v in zip(qs, vals, strict=True):
        print(f"p{q:<3}: {v:.4f}")

    print("\nThreshold exceedances (Test N=1000):")
    for t in [0.90, 0.95, 0.98]:
        cnt = int(np.sum(max_sims >= t))
        print(f"Cosine >= {t:.2f}: {cnt:>4} ({cnt / len(max_sims) * 100:.1f}%)")

    # Same-prompt SDXL vs random SDXL pairs
    sdxl_samples = [s for s in samples if s.source == "sdxl"]
    groups: dict[str, list[str]] = {}
    for s in sdxl_samples:
        groups.setdefault(s.group, []).append(s.sample_id)

    multi = [v for v in groups.values() if len(v) >= 2]
    same_prompt_sims = []
    for g in multi:
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                same_prompt_sims.append(float(np.dot(feats[id_to_idx[g[i]]], feats[id_to_idx[g[j]]])))

    rng = random.Random(0)
    all_sdxl = [s.sample_id for s in sdxl_samples]
    random_sdxl_sims = []
    for _ in range(len(same_prompt_sims)):
        i1, i2 = rng.sample(all_sdxl, 2)
        if by_id[i1].group != by_id[i2].group:
            random_sdxl_sims.append(float(np.dot(feats[id_to_idx[i1]], feats[id_to_idx[i2]])))

    sp = np.array(same_prompt_sims)
    rd = np.array(random_sdxl_sims)
    print(f"\n=== SDXL PAIRWISE COSINE SIMILARITY (N={len(sp)}) ===")
    print(
        f"Same prompt index:   mean={sp.mean():.4f} | median={np.median(sp):.4f} | IQR={np.percentile(sp, 75) - np.percentile(sp, 25):.4f}"
    )
    print(
        f"Random prompt index: mean={rd.mean():.4f} | median={np.median(rd):.4f} | IQR={np.percentile(rd, 75) - np.percentile(rd, 25):.4f}"
    )


if __name__ == "__main__":
    main()
