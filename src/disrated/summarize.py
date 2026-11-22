from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .data.vhd11k import load_vhd11k
from .metrics.scoring import read_predictions


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m disrated.summarize",
        description="Test AUC / accuracy with 95%% bootstrap CIs. Each run is NAME=file1,file2,... "
        "(one file per seed); seeds are averaged inside every bootstrap resample.",
    )
    ap.add_argument("runs", nargs="+", help="NAME=pred1.jsonl,pred2.jsonl,...")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--ref", help="NAME of a run to compare every other run against (paired)")
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "all"],
        help="'all' = all 10,000 images; use it with the *-oof.jsonl files",
    )
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}
    if a.split == "all":
        ids = sorted(by_id)
    else:
        ids = sorted(json.loads(a.splits.read_text(encoding="utf-8"))["test"])
    y = np.array([by_id[i].harmful for i in ids], dtype=int)

    runs: dict[str, np.ndarray] = {}
    for spec in a.runs:
        name, _, files = spec.partition("=")
        rows = []
        for f in files.split(","):
            preds = read_predictions(Path(f))
            miss = [i for i in ids if i not in preds or preds[i].score is None or preds[i].error]
            if miss:
                raise SystemExit(f"{f}: {len(miss)} test samples without a usable score")
            rows.append([preds[i].score for i in ids])
        runs[name] = np.array(rows)  # (n_seeds, n_test)
    if a.ref and a.ref not in runs:
        raise SystemExit(f"--ref {a.ref!r} is not one of the runs: {list(runs)}")

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(a.boot, len(y)))
    idx = np.array([r for r in idx if len(set(y[r].tolist())) == 2])

    def stats(S: np.ndarray, rows: np.ndarray | None = None):
        sel = slice(None) if rows is None else rows
        yy = y if rows is None else y[sel]
        auc = np.mean([roc_auc_score(yy, s[sel]) for s in S])
        acc = np.mean([((s[sel] >= a.threshold) == (yy == 1)).mean() for s in S])
        return auc, acc

    boot = {n: np.array([stats(S, r) for r in idx]) for n, S in runs.items()}  # (B, 2)

    print(f"{a.split} n={len(y)} | bootstrap B={len(idx)} | threshold={a.threshold}\n")
    print(f"{'run':<16}{'seeds':>6}  {'AUC':>7} {'95% CI':>17}  {'seed-sd':>8}   {'acc':>6} {'95% CI':>17}")
    for n, S in runs.items():
        auc, acc = stats(S)
        lo_a, hi_a = np.percentile(boot[n][:, 0], [2.5, 97.5])
        lo_c, hi_c = np.percentile(boot[n][:, 1], [2.5, 97.5])
        sd = np.std([roc_auc_score(y, s) for s in S], ddof=1) if len(S) > 1 else float("nan")
        print(
            f"{n:<16}{len(S):>6}  {auc:>7.3f} [{lo_a:.3f}, {hi_a:.3f}]  {sd:>8.3f}   "
            f"{acc:>6.3f} [{lo_c:.3f}, {hi_c:.3f}]"
        )

    if a.ref:

        def flag(lo: float, hi: float) -> str:
            return "clear" if lo > 0 or hi < 0 else "unclear"

        print(f"\npaired difference vs {a.ref} (run - ref); CI excluding 0 = clear difference")
        for n in runs:
            if n == a.ref:
                continue
            d = boot[n] - boot[a.ref]
            (la, ha), (lc, hc) = np.percentile(d[:, 0], [2.5, 97.5]), np.percentile(d[:, 1], [2.5, 97.5])
            ma, mc = d[:, 0].mean(), d[:, 1].mean()
            print(
                f"{n:<16} dAUC {ma:+.3f} [{la:+.3f}, {ha:+.3f}] {flag(la, ha):<8}"
                f" dAcc {mc:+.3f} [{lc:+.3f}, {hc:+.3f}] {flag(lc, hc)}"
            )


if __name__ == "__main__":
    main()
