from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .data.vhd11k import load_vhd11k
from .metrics.scoring import read_predictions


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.sweep_thresholds")
    ap.add_argument("--val-preds", type=Path, default=Path("results/clip-b16-v2-val.jsonl"))
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--max-escalation", type=float, default=0.30)
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}

    splits = json.loads(a.splits.read_text(encoding="utf-8"))
    val_ids = splits["val"]
    preds = read_predictions(a.val_preds)

    y_true = np.array([by_id[sid].harmful for sid in val_ids])
    y_scores = np.array([preds[sid].score for sid in val_ids])

    # Grid search over tau_low in [0.05, 0.45] and tau_high in [0.55, 0.95]
    taus_low = np.linspace(0.05, 0.45, 41)
    taus_high = np.linspace(0.55, 0.95, 41)

    candidates = []

    for t_low in taus_low:
        for t_high in taus_high:
            if t_low >= t_high:
                continue

            auto_safe = y_scores <= t_low
            auto_harm = y_scores >= t_high
            escalated = (y_scores > t_low) & (y_scores < t_high)

            n_safe = int(np.sum(auto_safe))
            n_harm = int(np.sum(auto_harm))
            n_esc = int(np.sum(escalated))
            esc_rate = n_esc / len(val_ids)

            # False Negative Rate on Auto-Safe (leaked violations)
            fnr_safe = np.sum(y_true[auto_safe]) / n_safe if n_safe > 0 else 0.0
            # False Positive Rate on Auto-Harm (false alarms)
            fpr_harm = np.sum(~y_true[auto_harm]) / n_harm if n_harm > 0 else 0.0

            # Combined error on auto-resolved traffic
            auto_errors = np.sum(y_true[auto_safe]) + np.sum(~y_true[auto_harm])
            auto_resolved = n_safe + n_harm
            auto_error_rate = auto_errors / auto_resolved if auto_resolved > 0 else 1.0

            if esc_rate <= a.max_escalation:
                candidates.append({
                    "tau_low": float(t_low),
                    "tau_high": float(t_high),
                    "esc_rate": float(esc_rate),
                    "n_esc": n_esc,
                    "n_safe": n_safe,
                    "n_harm": n_harm,
                    "fnr_safe": float(fnr_safe),
                    "fpr_harm": float(fpr_harm),
                    "auto_error_rate": float(auto_error_rate),
                })

    if not candidates:
        raise SystemExit(f"No operating points found with escalation <= {a.max_escalation * 100:.1f}%.")

    # Sort candidates: primary by lowest auto_error_rate, secondary by closest to max escalation budget
    candidates.sort(key=lambda x: (x["auto_error_rate"], -x["esc_rate"]))

    print(f"\n=== VALIDATION RISK-COVERAGE SWEEP (N={len(val_ids)}, Max Esc={a.max_escalation * 100:.1f}%) ===")
    print(f"{'tau_low':>7} {'tau_high':>8} | {'Esc Rate':>8} {'Esc N':>6} | {'Auto Safe':>9} {'Safe FNR':>9} | {'Auto Harm':>9} {'Harm FPR':>9} | {'Auto Err':>9}")
    print("-" * 92)

    # Display top 10 operating points
    for c in candidates[:10]:
        print(
            f"{c['tau_low']:7.3f} {c['tau_high']:8.3f} | "
            f"{c['esc_rate'] * 100:7.1f}% {c['n_esc']:6d} | "
            f"{c['n_safe']:9d} {c['fnr_safe'] * 100:8.2f}% | "
            f"{c['n_harm']:9d} {c['fpr_harm'] * 100:8.2f}% | "
            f"{c['auto_error_rate'] * 100:8.2f}%"
        )

    best = candidates[0]
    print(f"\nRecommended operating point:")
    print(f"  tau_low  = {best['tau_low']:.3f} (resolves {best['n_safe']} safe, {best['fnr_safe'] * 100:.2f}% false negative rate)")
    print(f"  tau_high = {best['tau_high']:.3f} (resolves {best['n_harm']} harmful, {best['fpr_harm'] * 100:.2f}% false positive rate)")
    print(f"  escalates {best['n_esc']} images ({best['esc_rate'] * 100:.1f}% traffic) to Tier 2")


if __name__ == "__main__":
    main()