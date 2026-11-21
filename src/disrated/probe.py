from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .data.vhd11k import load_vhd11k
from .schema import Prediction

CS = (0.01, 0.1, 0.3, 1.0, 10.0, 100.0)


def fit(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def write(
    path: Path,
    samples,
    scores,
    ms,
    extra: dict[str, Any] | None = None,
    meta_sidecar: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s, sc, t in zip(samples, scores, ms, strict=True):
            p = Prediction(
                sample_id=s.sample_id,
                score=min(max(float(sc), 0.0), 1.0),
                latency_ms=float(t),
                extra=dict(extra) if extra else {},
            )
            f.write(p.model_dump_json() + "\n")
    if meta_sidecar is not None:
        mp = path.with_suffix(path.suffix + ".meta.json")
        mp.write_text(json.dumps(meta_sidecar, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def tune_c(Xtr, ytr, Xva, yva, grid: tuple[float, ...]) -> tuple[float, dict[float, float]]:
    val_auc = {C: auc(yva, fit(Xtr, ytr, C).predict_proba(Xva)[:, 1]) for C in grid}
    best = max(val_auc, key=lambda c: val_auc[c])
    print("val AUC by C:", {C: round(v, 4) for C, v in val_auc.items()}, "-> best C =", best)

    if best == grid[0]:
        c_low = best / 10.0
        print(f"Warning: Best C at lower boundary ({best}). Evaluating {c_low}...")
        a_low = auc(yva, fit(Xtr, ytr, c_low).predict_proba(Xva)[:, 1])
        val_auc[c_low] = a_low
        if a_low > val_auc[best]:
            return c_low, val_auc
    elif best == grid[-1]:
        c_hi = best * 10.0
        print(f"Warning: Best C at upper boundary ({best}). Evaluating {c_hi}...")
        a_hi = auc(yva, fit(Xtr, ytr, c_hi).predict_proba(Xva)[:, 1])
        val_auc[c_hi] = a_hi
        if a_hi > val_auc[best]:
            return c_hi, val_auc

    return best, val_auc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated probe")
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    ap.add_argument("--name", default="clip-b16-v2")
    ap.add_argument(
        "--val-only",
        action="store_true",
        help="Fit on train and write val predictions only (keeps test strictly sealed).",
    )
    a = ap.parse_args(argv)

    samples, _ = load_vhd11k(a.dataset.expanduser())
    z = np.load(a.embeddings, allow_pickle=False)
    row = {i: k for k, i in enumerate(z["ids"].tolist())}

    by_id = {s.sample_id: s for s in samples}
    splits_raw = a.splits.expanduser().read_text(encoding="utf-8")
    splits_sha256 = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    splits = json.loads(splits_raw)

    needed_splits = ("train", "val") if a.val_only else ("train", "val", "test")
    for sname in needed_splits:
        missing_ds = [sid for sid in splits[sname] if sid not in by_id]
        if missing_ds:
            raise SystemExit(
                f"Dataset missing {len(missing_ds)} sample(s) from split '{sname}' (first: {missing_ds[0]})."
            )
        missing_emb = [sid for sid in splits[sname] if sid not in row]
        if missing_emb:
            raise SystemExit(
                f"Embeddings missing {len(missing_emb)} sample(s) from split '{sname}' (first: {missing_emb[0]})."
            )

    def part(sname):
        sel = [by_id[i] for i in splits[sname]]
        idx = [row[s.sample_id] for s in sel]
        return sel, z["feats"][idx], np.array([s.harmful for s in sel]), z["ms"][idx]

    _, Xtr, ytr, _ = part("train")
    va, Xva, yva, mva = part("val")

    best_c, _ = tune_c(Xtr, ytr, Xva, yva, CS)
    m = fit(Xtr, ytr, best_c)

    emb_sha256 = hashlib.sha256(a.embeddings.expanduser().read_bytes()).hexdigest()
    meta = {"splits_sha256": splits_sha256, "c": best_c}
    probe_cfg = {
        "name": a.name,
        "embeddings": str(a.embeddings),
        "embeddings_sha256": emb_sha256,
        "c": best_c,
    }
    write(
        Path(f"results/{a.name}-val.jsonl"),
        va,
        m.predict_proba(Xva)[:, 1],
        mva,
        extra=meta,
        meta_sidecar={
            "moderator": a.name,
            "config": probe_cfg,
            "run": {"split": "val", "splits_sha256": splits_sha256},
        },
    )
    if a.val_only:
        print(f"Wrote results/{a.name}-val.jsonl (--val-only: test untouched)")
        return

    te, Xte, yte, mte = part("test")
    write(
        Path(f"results/{a.name}-test.jsonl"),
        te,
        m.predict_proba(Xte)[:, 1],
        mte,
        extra=meta,
        meta_sidecar={
            "moderator": a.name,
            "config": probe_cfg,
            "run": {"split": "test", "splits_sha256": splits_sha256},
        },
    )

    src = np.array([s.source for s in te])
    p_te = m.predict_proba(Xte)[:, 1]
    print(
        f"TEST AUC  pooled={auc(yte, p_te):.4f} | "
        f"concept={auc(yte[src == 'concept'], p_te[src == 'concept']):.4f} | "
        f"sdxl={auc(yte[src == 'sdxl'], p_te[src == 'sdxl']):.4f}"
    )

    # 4. 5-fold OOF using connected components if available
    # 4. 5-fold OOF using connected components
    comp_file = Path("embeddings/components.json")
    if not comp_file.exists():
        raise SystemExit(f"ERROR: {comp_file} not found. Run rebuild_splits first.")

    comp_raw = comp_file.read_text(encoding="utf-8")
    comp_sha256 = hashlib.sha256(comp_raw.encode("utf-8")).hexdigest()
    comp_map = json.loads(comp_raw)

    idx_all = [row[s.sample_id] for s in samples]
    X_all, ms_all = z["feats"][idx_all], z["ms"][idx_all]
    y_all = np.array([s.harmful for s in samples])
    strat = [f"{s.source}|{int(s.harmful)}" for s in samples]
    groups = [comp_map[s.sample_id] for s in samples]

    oof = np.zeros(len(samples))
    fold_ids = np.zeros(len(samples), dtype=int)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    fold_splits = list(cv.split(np.zeros(len(samples)), strat, groups))

    # Assert every component sits in exactly one fold
    comp_to_folds = defaultdict(set)
    for fold_idx, (_, tst) in enumerate(fold_splits):
        for idx in tst:
            comp_to_folds[groups[idx]].add(fold_idx)
            fold_ids[idx] = fold_idx

    leaked_comps = {c: f for c, f in comp_to_folds.items() if len(f) > 1}
    assert not leaked_comps, f"Component leakage: {len(leaked_comps)} components straddle multiple folds!"

    print("\n=== 5-FOLD OUT-OF-FOLD (OOF) EVALUATION ===")
    print(f"Component map hash: {comp_sha256}")
    fold_aucs = []
    for fold_idx, (trn, tst) in enumerate(fold_splits):
        oof[tst] = fit(X_all[trn], y_all[trn], best_c).predict_proba(X_all[tst])[:, 1]
        f_auc = auc(y_all[tst], oof[tst])
        fold_aucs.append(f_auc)
        print(f"Fold {fold_idx}: n={len(tst):>4} | AUC={f_auc:.4f}")

    print(
        f"Per-fold AUC summary: mean={np.mean(fold_aucs):.4f} | min={np.min(fold_aucs):.4f} | max={np.max(fold_aucs):.4f} | std={np.std(fold_aucs):.4f}"
    )
    print(f"OOF AUC pooled:       {auc(y_all, oof):.4f}")

    oof_path = Path(f"results/{a.name}-oof.jsonl")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    with oof_path.open("w", encoding="utf-8") as f:
        for s, sc, t, f_id in zip(samples, oof, ms_all, fold_ids, strict=True):
            p = Prediction(
                sample_id=s.sample_id,
                score=min(max(float(sc), 0.0), 1.0),
                latency_ms=float(t),
                extra={
                    "splits_sha256": splits_sha256,
                    "components_sha256": comp_sha256,
                    "c": best_c,
                    "fold": int(f_id),
                },
            )
            f.write(p.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
