from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .data.vhd11k import load_vhd11k
from .metrics.scoring import read_predictions


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.audit_effect")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, default=Path("results/clip-b16-test.jsonl"))
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings/openai_clip-vit-base-patch16.npz"))
    ap.add_argument("--hashes", type=Path, default=Path("embeddings/hashes.npz"))
    ap.add_argument("--splits", type=Path, default=Path("splits.json"))
    ap.add_argument("--cosine-threshold", type=float, required=True)
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}
    splits = json.loads(a.splits.read_text(encoding="utf-8"))
    preds = read_predictions(a.predictions)

    z_emb = np.load(a.embeddings, allow_pickle=False)
    id_to_idx = {sid: idx for idx, sid in enumerate(z_emb["ids"].tolist())}
    tr_idx = np.array([id_to_idx[sid] for sid in splits["train"]])
    te_ids = splits["test"]
    te_idx = np.array([id_to_idx[sid] for sid in te_ids])

    # Cosine check
    X_tr = z_emb["feats"][tr_idx]
    X_te = z_emb["feats"][te_idx]
    max_cos = np.max(np.matmul(X_te, X_tr.T), axis=1)
    cos_leaked = set(np.where(max_cos >= a.cosine_threshold)[0])

    # pHash check
    z_hash = np.load(a.hashes, allow_pickle=False)
    lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    ph_tr = z_hash["phash"][tr_idx]
    ph_te = z_hash["phash"][te_idx]

    ph_leaked = set()
    for i in range(len(te_idx)):
        xor = np.bitwise_xor(ph_te[i : i + 1], ph_tr)
        if np.any(lut[xor].sum(axis=-1) <= 8):
            ph_leaked.add(i)

    leaked_indices = cos_leaked | ph_leaked
    mask_leaked = np.array([i in leaked_indices for i in range(len(te_ids))])

    y_true = np.array([by_id[sid].harmful for sid in te_ids])
    y_score = np.array([preds[sid].score for sid in te_ids])

    print("\n=== EFFECT SIZE ON TEST SPLIT ===")
    for label, mask in [("With Train Duplicate", mask_leaked), ("Without Train Duplicate", ~mask_leaked)]:
        n_group = int(np.sum(mask))
        if n_group == 0 or len(np.unique(y_true[mask])) < 2:
            print(f"{label:<28}: n={n_group:>4} | undefined (insufficient class variation)")
            continue
        auc_val = roc_auc_score(y_true[mask], y_score[mask])
        ci_lo, ci_hi = bootstrap_ci(y_true[mask], y_score[mask])
        print(f"{label:<28}: n={n_group:>4} | AUC={auc_val:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")


if __name__ == "__main__":
    main()
