from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .data.vhd11k import load_vhd11k
from .metrics.scoring import read_predictions


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def compute_metrics(z: np.ndarray, y: np.ndarray, c: float) -> dict[str, float]:
    band = np.abs(z) < c
    res = ~band
    pred = z > 0.0
    safe, harm = res & ~pred, res & pred
    nan = float("nan")
    return {
        "esc": float(band.mean()),
        "n_band": int(band.sum()),
        "acc_res": float((pred[res] == y[res]).mean()) if res.any() else nan,
        "acc_band": float((pred[band] == y[band]).mean()) if band.any() else nan,
        "npv": float((~y[safe]).mean()) if safe.any() else nan,
        "ppv": float(y[harm].mean()) if harm.any() else nan,
    }


def boot_ci(z: np.ndarray, y: np.ndarray, c: float, n_boot: int = 2000, seed: int = 0) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    res_accs, band_accs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        zb, yb = z[idx], y[idx]
        band = np.abs(zb) < c
        res = ~band
        pred = zb > 0.0
        if res.any():
            res_accs.append((pred[res] == yb[res]).mean())
        if band.any():
            band_accs.append((pred[band] == yb[band]).mean())
    return {
        "acc_res": (float(np.percentile(res_accs, 2.5)), float(np.percentile(res_accs, 97.5))),
        "acc_band": (float(np.percentile(band_accs, 2.5)), float(np.percentile(band_accs, 97.5))),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated risk-coverage")
    ap.add_argument("--dataset", type=Path, required=True, help="path to VHD11K dataset root")
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"), help="path to active splits JSON")
    ap.add_argument("--val-preds", type=Path, default=Path("results/clip-b16-v2-val.jsonl"))
    ap.add_argument("--test-preds", type=Path, default=Path("results/clip-b16-v2-test.jsonl"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}
    splits_raw = a.splits.expanduser().read_text(encoding="utf-8")
    expected_hash = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    sp = json.loads(splits_raw)

    def load(split: str, pred_path: Path):
        preds = read_predictions(pred_path.expanduser())
        logged_hashes = {p.extra["splits_sha256"] for p in preds.values() if "splits_sha256" in p.extra}
        if logged_hashes and logged_hashes != {expected_hash}:
            raise SystemExit(
                f"ERROR: Splits hash mismatch in {pred_path}: {logged_hashes} != {expected_hash}"
            )
        s = np.array([preds[i].score for i in sp[split]])
        y = np.array([by_id[i].harmful for i in sp[split]])
        return s, y

    sv, yv = load("val", a.val_preds)
    zv = logit(sv)
    eval_splits: list[tuple[str, np.ndarray, np.ndarray]] = [("val", zv, yv)]
    if a.test_preds.expanduser().is_file():
        st, yt = load("test", a.test_preds)
        eval_splits.append(("test", logit(st), yt))

    print(f"=== ONE-KNOB RISK-COVERAGE TABLE ({a.splits.name}, B={a.n_boot} bootstrap) ===")
    header = (
        f"{'target':>6} {'split':>5} | {'esc':>6} {'n_band':>6} | "
        f"{'acc(resolved) [95% CI]':>24} | {'acc(band) [95% CI]':>24} | "
        f"{'safe NPV':>8} {'harm PPV':>8}"
    )
    print(header)
    print("-" * len(header))

    for k in (0.10, 0.20, 0.30, 0.40, 0.50):
        c = float(np.quantile(np.abs(zv), k))
        for name, z, y in eval_splits:
            m = compute_metrics(z, y, c)
            ci = boot_ci(z, y, c, n_boot=a.n_boot, seed=a.seed)
            res_str = f"{m['acc_res']:.3f} [{ci['acc_res'][0]:.3f}, {ci['acc_res'][1]:.3f}]"
            band_str = f"{m['acc_band']:.3f} [{ci['acc_band'][0]:.3f}, {ci['acc_band'][1]:.3f}]"
            print(
                f"{k:6.2f} {name:>5} | {m['esc']:6.3f} {m['n_band']:6d} | "
                f"{res_str:>24} | {band_str:>24} | "
                f"{m['npv']:8.3f} {m['ppv']:8.3f}"
            )

    print("\n=== TWO-KNOB FEASIBILITY (val only, >=20 images per side) ===")
    for target in (0.90, 0.85, 0.80):
        best = None
        for tl in np.arange(0.02, 0.50, 0.01):
            for th in np.arange(0.51, 0.99, 0.01):
                safe, harm = sv <= tl, sv >= th
                if safe.sum() < 20 or harm.sum() < 20:
                    continue
                if (~yv[safe]).mean() >= target and yv[harm].mean() >= target:
                    esc = 1.0 - safe.mean() - harm.mean()
                    if best is None or esc < best[0]:
                        best = (esc, tl, th)
        print(
            f"target {target:.2f}:",
            "none feasible" if best is None else f"escalation {best[0]:.3f} at tau_low={best[1]:.2f}, tau_high={best[2]:.2f} (<=0.30 feasible: {best[0] <= 0.30})",
        )


if __name__ == "__main__":
    main()