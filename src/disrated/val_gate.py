from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .data.vhd11k import load_vhd11k
from .metrics.scoring import binary_metrics, read_predictions
from .risk_coverage import logit

CANDIDATE_KS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _macro_f1_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fn = int(np.sum(y_true & (~y_pred)))
    d_pos = 2 * tp + fp + fn
    d_neg = 2 * tn + fn + fp
    f1_pos = (2 * tp / d_pos) if d_pos else 0.0
    f1_neg = (2 * tn / d_neg) if d_neg else 0.0
    return 0.5 * (f1_pos + f1_neg)


def paired_bootstrap_acc_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired non-parametric bootstrap for accuracy(A) - accuracy(B)."""
    n = len(y_true)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "acc_a": nan,
            "acc_b": nan,
            "diff": nan,
            "acc_a_ci95": [nan, nan],
            "acc_b_ci95": [nan, nan],
            "diff_ci95": [nan, nan],
            "p_one_sided_le_0": nan,
        }
    rng = np.random.default_rng(seed)
    ca = (pred_a == y_true).astype(np.float64)
    cb = (pred_b == y_true).astype(np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    acc_a_boot = ca[idx].mean(axis=1)
    acc_b_boot = cb[idx].mean(axis=1)
    diff_boot = acc_a_boot - acc_b_boot
    return {
        "n": int(n),
        "acc_a": float(ca.mean()),
        "acc_b": float(cb.mean()),
        "diff": float(ca.mean() - cb.mean()),
        "acc_a_ci95": [float(np.percentile(acc_a_boot, 2.5)), float(np.percentile(acc_a_boot, 97.5))],
        "acc_b_ci95": [float(np.percentile(acc_b_boot, 2.5)), float(np.percentile(acc_b_boot, 97.5))],
        "diff_ci95": [float(np.percentile(diff_boot, 2.5)), float(np.percentile(diff_boot, 97.5))],
        "p_one_sided_le_0": float(np.mean(diff_boot <= 0.0)),
    }


def paired_bootstrap_macro_f1_diffs(
    y_true: np.ndarray,
    pred_probe: np.ndarray,
    pred_vlm: np.ndarray,
    pred_cascade: np.ndarray,
    pred_random: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap CIs for cascade Macro-F1 and paired differences vs arms."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    arms = {
        "probe": pred_probe,
        "vlm": pred_vlm,
        "cascade": pred_cascade,
        "random": pred_random,
    }
    point = {k: _macro_f1_np(y_true, p) for k, p in arms.items()}

    boot = {k: np.empty(n_boot, dtype=np.float64) for k in arms}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        for k, p in arms.items():
            boot[k][b] = _macro_f1_np(yb, p[idx])

    def ci95(arr: np.ndarray) -> list[float]:
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    diffs = {
        "cascade_minus_probe": (point["cascade"] - point["probe"], boot["cascade"] - boot["probe"]),
        "cascade_minus_vlm": (point["cascade"] - point["vlm"], boot["cascade"] - boot["vlm"]),
        "cascade_minus_random": (point["cascade"] - point["random"], boot["cascade"] - boot["random"]),
    }
    return {
        "macro_f1": {k: {"point": float(point[k]), "ci95": ci95(boot[k])} for k in arms},
        "paired_diffs": {
            k: {"diff": float(val), "ci95": ci95(arr)} for k, (val, arr) in diffs.items()
        },
    }


def fit_validation_calibration(
    y_true: np.ndarray, vlm_scores: np.ndarray
) -> dict[str, Any]:
    """Fit Platt scaling (logit -> calibrated prob) and check optimal val threshold."""
    z = logit(vlm_scores)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(z.reshape(-1, 1), y_true.astype(int))
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    # sigmoid(a * logit(tau) + b) = 0.5  =>  logit(tau) = -b / a
    tau_platt = float(1.0 / (1.0 + np.exp(b / a))) if a > 0 else 0.50

    p_uncal = np.clip(vlm_scores, 1e-6, 1.0 - 1e-6)
    p_platt = np.clip(lr.predict_proba(z.reshape(-1, 1))[:, 1], 1e-6, 1.0 - 1e-6)

    # Pure 1-parameter temperature scaling (no intercept shift)
    temps = np.linspace(0.25, 4.0, 376)
    nlls = [
        float(log_loss(y_true, 1.0 / (1.0 + np.exp(-z / t)), labels=[0, 1]))
        for t in temps
    ]
    best_t_idx = int(np.argmin(nlls))
    temp_only = float(temps[best_t_idx])

    # Direct threshold sweep on val for reference
    taus = np.linspace(0.10, 0.90, 81)
    best_tau, best_f1 = 0.50, -1.0
    for tau in taus:
        f1 = _macro_f1_np(y_true, vlm_scores >= tau)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return {
        "platt_slope": a,
        "platt_intercept": b,
        "platt_threshold": tau_platt,
        "temp_only": temp_only,
        "nll_uncalibrated": float(log_loss(y_true, p_uncal, labels=[0, 1])),
        "nll_temp_only": float(nlls[best_t_idx]),
        "nll_platt": float(log_loss(y_true, p_platt, labels=[0, 1])),
        "val_f1_opt_threshold": round(best_tau, 4),
        "val_f1_opt_macro_f1": float(best_f1),
        "vlm_roc_auc": float(roc_auc_score(y_true, vlm_scores)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def evaluate_val_gate(
    y_true: np.ndarray,
    probe_scores: np.ndarray,
    vlm_scores: np.ndarray,
    tau_vlm: float = 0.50,
    ks: tuple[float, ...] = CANDIDATE_KS,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run Hypothesis Section 4 Validation Gate + Operating Point selection on val."""
    n = len(y_true)
    z_probe = logit(probe_scores)
    abs_z = np.abs(z_probe)
    pred_probe = probe_scores >= 0.50
    pred_vlm = vlm_scores >= tau_vlm

    probe_m = binary_metrics(y_true.tolist(), pred_probe.tolist())
    vlm_m = binary_metrics(y_true.tolist(), pred_vlm.tolist())

    candidates: list[dict[str, Any]] = []
    for k in ks:
        c_k = float(np.quantile(abs_z, k))
        band = abs_z < c_k
        n_esc = int(band.sum())
        pred_cascade = np.where(band, pred_vlm, pred_probe)

        rand_band = np.zeros(n, dtype=bool)
        if n_esc > 0:
            rand_band[np.random.default_rng(seed).choice(n, size=n_esc, replace=False)] = True
        pred_random = np.where(rand_band, pred_vlm, pred_probe)

        in_band = paired_bootstrap_acc_diff(
            y_true[band], pred_vlm[band], pred_probe[band], n_boot=n_boot, seed=seed
        )
        full_boot = paired_bootstrap_macro_f1_diffs(
            y_true, pred_probe, pred_vlm, pred_cascade, pred_random, n_boot=n_boot, seed=seed
        )
        res = ~band
        resolved_acc = float((pred_probe[res] == y_true[res]).mean()) if res.any() else float("nan")

        gate_ok = in_band["n"] > 0 and (in_band["acc_a"] > in_band["acc_b"])
        cas_f1 = full_boot["macro_f1"]["cascade"]["point"]
        prb_f1 = full_boot["macro_f1"]["probe"]["point"]
        vlm_f1 = full_boot["macro_f1"]["vlm"]["point"]

        candidates.append(
            {
                "k": float(k),
                "c_k": c_k,
                "n_esc": n_esc,
                "esc_rate": float(n_esc / n),
                "resolved_acc": resolved_acc,
                "gate_passed": bool(gate_ok),
                "within_non_inferiority_and_beats_probe": bool(
                    gate_ok and (cas_f1 >= vlm_f1 - 0.030) and (cas_f1 > prb_f1)
                ),
                "in_band": in_band,
                "cascade_acc": float((pred_cascade == y_true).mean()),
                "random_acc": float((pred_random == y_true).mean()),
                "arms": full_boot,
            }
        )

    passing = [c for c in candidates if c["gate_passed"]]
    selected: dict[str, Any] | None = None
    selection_rule = "falsified_at_gate"
    if passing:
        primary = [c for c in passing if c["within_non_inferiority_and_beats_probe"]]
        if primary:
            selected = min(primary, key=lambda c: c["k"])
            selection_rule = "smallest_k_within_0.030_of_vlm_and_beats_probe"
        else:
            selected = max(passing, key=lambda c: c["arms"]["macro_f1"]["cascade"]["point"])
            selection_rule = "fallback_max_val_cascade_macro_f1"

    return {
        "tau_vlm": float(tau_vlm),
        "n_val": int(n),
        "probe_val": {"accuracy": float(probe_m["accuracy"]), "macro_f1": float(probe_m["macro_f1"])},
        "vlm_val": {"accuracy": float(vlm_m["accuracy"]), "macro_f1": float(vlm_m["macro_f1"])},
        "gate_passed_any": bool(passing),
        "selection_rule": selection_rule,
        "selected": selected,
        "candidates": candidates,
    }


def _load_val_aligned(
    dataset: Path, splits_path: Path, probe_path: Path, vlm_path: Path
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, str]:
    splits_raw = splits_path.read_text(encoding="utf-8")
    splits_sha256 = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    val_ids: list[str] = json.loads(splits_raw)["val"]

    samples, _ = load_vhd11k(dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}
    probe_preds = read_predictions(probe_path)
    vlm_preds = read_predictions(vlm_path)

    for label, preds in (("probe", probe_preds), ("vlm", vlm_preds)):
        logged_hashes = {p.extra["splits_sha256"] for p in preds.values() if "splits_sha256" in p.extra}
        if logged_hashes and logged_hashes != {splits_sha256}:
            raise SystemExit(f"Splits sha256 mismatch in {label} predictions: {logged_hashes} != {splits_sha256}")
        missing = [sid for sid in val_ids if sid not in preds or preds[sid].error is not None or preds[sid].refused or preds[sid].score is None]
        if missing:
            raise SystemExit(f"{label} predictions missing/unusable on {len(missing)} val samples (first: {missing[0]})")

    y_true = np.array([by_id[sid].harmful for sid in val_ids], dtype=bool)
    p_probe = np.array([float(probe_preds[sid].score) for sid in val_ids], dtype=np.float64)
    p_vlm = np.array([float(vlm_preds[sid].score) for sid in val_ids], dtype=np.float64)
    return val_ids, y_true, p_probe, p_vlm, splits_sha256


def _print_summary(label: str, res: dict[str, Any]) -> None:
    print(f"\n=== VALIDATION GATE SUMMARY ({label}, tau_vlm={res['tau_vlm']:.4f}) ===")
    print(f"probe-only: acc={res['probe_val']['accuracy']:.3f} | macroF1={res['probe_val']['macro_f1']:.3f}")
    print(f"vlm-only:   acc={res['vlm_val']['accuracy']:.3f} | macroF1={res['vlm_val']['macro_f1']:.3f}")
    print(
        f"{'k':>5} {'c_k':>7} {'n':>4} | {'band VLM/Prb (diff [95% CI])':>32} | "
        f"{'casF1':>6} {'rndF1':>6} {'dPrb [95% CI]':>19} {'dVlm [95% CI]':>19} | {'gate':>4}"
    )
    print("-" * 116)
    for c in res["candidates"]:
        ib = c["in_band"]
        arms = c["arms"]
        cas_f1 = arms["macro_f1"]["cascade"]["point"]
        rnd_f1 = arms["macro_f1"]["random"]["point"]
        dp = arms["paired_diffs"]["cascade_minus_probe"]
        dv = arms["paired_diffs"]["cascade_minus_vlm"]
        ib_str = f"{ib['acc_a']:.3f}/{ib['acc_b']:.3f} ({ib['diff']:+.3f} [{ib['diff_ci95'][0]:+.3f},{ib['diff_ci95'][1]:+.3f}])"
        dp_str = f"{dp['diff']:+.3f} [{dp['ci95'][0]:+.3f},{dp['ci95'][1]:+.3f}]"
        dv_str = f"{dv['diff']:+.3f} [{dv['ci95'][0]:+.3f},{dv['ci95'][1]:+.3f}]"
        print(
            f"{c['k']:5.2f} {c['c_k']:7.4f} {c['n_esc']:4d} | {ib_str:>32} | "
            f"{cas_f1:6.3f} {rnd_f1:6.3f} {dp_str:>19} {dv_str:>19} | {'PASS' if c['gate_passed'] else 'FAIL':>4}"
        )
    sel = res["selected"]
    if sel is None:
        print(f"Selection ({res['selection_rule']}): NO candidate passed validation gate.")
    else:
        print(f"Selection ({res['selection_rule']}): k*={sel['k']:.2f}, c*={sel['c_k']:.6f}, cascade_macroF1={sel['arms']['macro_f1']['cascade']['point']:.4f}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated val-gate")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    ap.add_argument("--probe", type=Path, default=Path("results/clip-b16-v2-val.jsonl"))
    ap.add_argument("--vlm", type=Path, default=Path("results/d1-2b-mlx-v2-val.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/phase3_val_gate.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tau-mode",
        choices=("auto", "uncalibrated", "platt", "val_f1_opt"),
        default="auto",
        help="How to lock tau_vlm* ('auto' picks uncalibrated 0.50 unless Platt improves val VLM macro_f1).",
    )
    a = ap.parse_args(argv)

    _, y_true, p_probe, p_vlm, splits_sha256 = _load_val_aligned(
        a.dataset, a.splits, a.probe, a.vlm
    )
    cal = fit_validation_calibration(y_true, p_vlm)
    print(
        f"VLM calibration on val (N={len(y_true)}): ROC-AUC={cal['vlm_roc_auc']:.4f} | "
        f"NLL uncal={cal['nll_uncalibrated']:.4f}, temp={cal['nll_temp_only']:.4f} (T={cal['temp_only']:.2f}), "
        f"platt={cal['nll_platt']:.4f} (tau_platt={cal['platt_threshold']:.4f}) | "
        f"val_f1_opt tau={cal['val_f1_opt_threshold']:.4f} (F1={cal['val_f1_opt_macro_f1']:.4f})"
    )

    uncal = evaluate_val_gate(y_true, p_probe, p_vlm, tau_vlm=0.50, n_boot=a.n_boot, seed=a.seed)
    platt = evaluate_val_gate(y_true, p_probe, p_vlm, tau_vlm=cal["platt_threshold"], n_boot=a.n_boot, seed=a.seed)
    f1_opt = evaluate_val_gate(y_true, p_probe, p_vlm, tau_vlm=cal["val_f1_opt_threshold"], n_boot=a.n_boot, seed=a.seed)

    _print_summary("uncalibrated", uncal)
    _print_summary("platt-calibrated", platt)
    _print_summary("val-f1-opt", f1_opt)

    if a.tau_mode == "uncalibrated":
        active_key, active = "uncalibrated", uncal
    elif a.tau_mode == "platt":
        active_key, active = "platt", platt
    elif a.tau_mode == "val_f1_opt":
        active_key, active = "val_f1_opt", f1_opt
    else:
        if platt["vlm_val"]["macro_f1"] > uncal["vlm_val"]["macro_f1"]:
            active_key, active = "platt", platt
        else:
            active_key, active = "uncalibrated", uncal

    sel = active["selected"]
    payload = {
        "splits": str(a.splits),
        "splits_sha256": splits_sha256,
        "probe_val_path": str(a.probe),
        "vlm_val_path": str(a.vlm),
        "n_boot": a.n_boot,
        "seed": a.seed,
        "calibration": cal,
        "locked_operating_point": {
            "tau_mode": active_key,
            "tau_vlm_star": active["tau_vlm"],
            "gate_passed": sel is not None,
            "selection_rule": active["selection_rule"],
            "k_star": sel["k"] if sel else None,
            "c_star": sel["c_k"] if sel else None,
            "val_cascade_macro_f1": sel["arms"]["macro_f1"]["cascade"]["point"] if sel else None,
            "val_in_band_vlm_acc": sel["in_band"]["acc_a"] if sel else None,
            "val_in_band_probe_acc": sel["in_band"]["acc_b"] if sel else None,
        },
        "evaluations": {
            "uncalibrated": uncal,
            "platt": platt,
            "val_f1_opt": f1_opt,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nLocked Phase 3 Validation Gate -> {a.out}")


if __name__ == "__main__":
    main()
