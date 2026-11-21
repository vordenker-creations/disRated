from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data.vhd11k import load_vhd11k


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.audit_dhash_residuals")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings/openai_clip-vit-base-patch16.npz"))
    ap.add_argument("--hashes", type=Path, default=Path("embeddings/hashes.npz"))
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}

    splits = json.loads(a.splits.read_text(encoding="utf-8"))
    z_emb = np.load(a.embeddings, allow_pickle=False)
    id_to_idx = {sid: idx for idx, sid in enumerate(z_emb["ids"].tolist())}

    z_hash = np.load(a.hashes, allow_pickle=False)
    dhash = z_hash["dhash"]
    phash = z_hash["phash"]
    feats = z_emb["feats"]
    lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)

    tr_ids = splits["train"]
    te_ids = splits["test"]
    tr_idx = np.array([id_to_idx[sid] for sid in tr_ids])
    te_idx = np.array([id_to_idx[sid] for sid in te_ids])

    print("\n=== DHASH RESIDUAL PAIRS (train vs test, dist <= 4) ===")
    print(f"{'Train ID':<35} {'Test ID':<35} {'dHash':<6} {'pHash':<6} {'Cosine':<7} {'Labels (Tr/Te)'}")
    print("-" * 105)

    for i, te_id in enumerate(te_ids):
        t_idx = te_idx[i]
        xor_d = np.bitwise_xor(dhash[t_idx : t_idx + 1], dhash[tr_idx])
        d_dists = lut[xor_d].sum(axis=-1)

        match_positions = np.where(d_dists <= 4)[0]
        for tr_pos in match_positions:
            tr_id = tr_ids[tr_pos]
            d_val = int(d_dists[tr_pos])

            # pHash distance
            xor_p = np.bitwise_xor(phash[t_idx], phash[tr_idx[tr_pos]])
            p_val = int(lut[xor_p].sum())

            # Cosine similarity
            cos_val = float(np.dot(feats[t_idx], feats[tr_idx[tr_pos]]))

            l_tr = "Harm" if by_id[tr_id].harmful else "Safe"
            l_te = "Harm" if by_id[te_id].harmful else "Safe"

            print(f"{tr_id:<35} {te_id:<35} {d_val:<6} {p_val:<6} {cos_val:<7.4f} {l_tr}/{l_te}")


if __name__ == "__main__":
    main()
