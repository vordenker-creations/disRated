from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .data.vhd11k import load_vhd11k


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.audit_duplicate_labels")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings/openai_clip-vit-base-patch16.npz"))
    ap.add_argument("--hashes", type=Path, default=Path("embeddings/hashes.npz"))
    ap.add_argument("--cosine-threshold", type=float, default=0.95)
    ap.add_argument("--phash-threshold", type=int, default=8)
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}
    all_ids = [s.sample_id for s in samples]

    z_emb = np.load(a.embeddings, allow_pickle=False)
    feats = z_emb["feats"]

    z_hash = np.load(a.hashes, allow_pickle=False)
    phash = z_hash["phash"]
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    pairs = set()
    n = len(samples)
    chunk = 1000

    for i in range(0, n, chunk):
        end_i = min(i + chunk, n)
        # Cosine pairs
        cos = np.matmul(feats[i:end_i], feats.T)
        for r_off in range(end_i - i):
            g_r = i + r_off
            cos[r_off, :g_r + 1] = 0.0
        r_c, c_c = np.where(cos >= a.cosine_threshold)
        for r, c in zip(r_c, c_c):
            pairs.add((i + r, c))

        # pHash pairs
        for r_off in range(end_i - i):
            g_r = i + r_off
            xor = np.bitwise_xor(phash[g_r:g_r + 1], phash[g_r + 1:])
            dists = lut[xor].sum(axis=-1)
            for col in np.where(dists <= a.phash_threshold)[0]:
                pairs.add((g_r, g_r + 1 + col))

    agreed = 0
    disagreed = 0
    by_type = {"concept_concept": [0, 0], "sdxl_sdxl": [0, 0], "cross": [0, 0]}

    for i, j in pairs:
        s1 = by_id[all_ids[i]]
        s2 = by_id[all_ids[j]]
        match = (s1.harmful == s2.harmful)

        if match:
            agreed += 1
        else:
            disagreed += 1

        if s1.source == "concept" and s2.source == "concept":
            key = "concept_concept"
        elif s1.source == "sdxl" and s2.source == "sdxl":
            key = "sdxl_sdxl"
        else:
            key = "cross"

        by_type[key][0 if match else 1] += 1

    total = len(pairs)
    print("\n=== LABEL AGREEMENT AMONG NEAR-DUPLICATES ===")
    print(f"Total unique duplicate pairs: {total}")
    print(f"Same label:      {agreed:>4} ({agreed / total * 100:.1f}%)")
    print(f"Opposing label:  {disagreed:>4} ({disagreed / total * 100:.1f}%)")
    print("\nBreakdown by cohort (Agreed / Disagreed):")
    for k, (agr, dgr) in by_type.items():
        sub_tot = agr + dgr
        pct = (agr / sub_tot * 100) if sub_tot else 0
        print(f"  {k:<16}: {agr:>4} agreed, {dgr:>4} disagreed ({pct:.1f}% agreement)")


if __name__ == "__main__":
    main()
