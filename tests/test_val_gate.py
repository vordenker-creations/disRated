from __future__ import annotations

import numpy as np

from disrated.val_gate import evaluate_val_gate, fit_validation_calibration


def test_val_gate_selects_max_f1_meeting_ci_gates():
    n = 200
    y_true = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    comp_ids = [f"c_{i // 2}" for i in range(n)]

    # Probe is confident & accurate on outer 75%, wrong on inner 25%
    probe_scores = np.where(y_true, 0.90, 0.10)
    uncertain = np.arange(0, 50)  # 25% of 200
    probe_scores[uncertain] = np.where(y_true[uncertain], 0.48, 0.52)

    # VLM is accurate everywhere except 4 confident probe items
    vlm_scores = np.where(y_true, 0.85, 0.15)
    vlm_scores[50:54] = np.where(y_true[50:54], 0.30, 0.70)

    res = evaluate_val_gate(
        y_true,
        probe_scores,
        vlm_scores,
        tau_vlm=0.50,
        comp_ids=comp_ids,
        n_boot=400,
        n_rand=50,
        seed=0,
    )
    assert res["gate_passed_any"] is True
    assert res["bootstrap_unit"] == "component_cluster"
    assert res["n_clusters"] == 100
    assert res["selection_rule"] == "max_val_cascade_macro_f1_meeting_ci_gates"
    assert res["selected"]["k"] >= 0.25
    assert res["selected"]["ci_aligned_all_three"] is True
    assert res["selected"]["in_band"]["acc_a"] > res["selected"]["in_band"]["acc_b"]


def test_val_gate_marks_falsified_when_vlm_never_beats_probe_in_band():
    n = 100
    y_true = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    probe_scores = np.where(y_true, 0.85, 0.15)
    probe_scores[:30] = np.where(y_true[:30], 0.52, 0.48)
    vlm_scores = np.where(y_true, 0.20, 0.80)  # inverted VLM

    res = evaluate_val_gate(y_true, probe_scores, vlm_scores, tau_vlm=0.50, n_boot=200, n_rand=30, seed=0)
    assert res["gate_passed_any"] is False
    assert res["selection_rule"] == "falsified_at_gate"
    assert res["selected"] is None


def test_fit_validation_calibration_recovers_raw_unclipped_logit_shift():
    n = 200
    y_true = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    # Conservative VLM like Qwen3-VL: negatives around -14, positives around -2 -> z_cut around -8
    raw_logits = np.where(y_true, -2.0, -14.0)
    vlm_scores = 1.0 / (1.0 + np.exp(-raw_logits))

    cal = fit_validation_calibration(
        y_true,
        vlm_scores,
        vlm_logits=raw_logits,
        logit_source="extra.logit_diff",
    )
    assert cal["logit_source"] == "extra.logit_diff"
    assert -10.0 < cal["platt_z_cut"] < -6.0
    assert cal["nll_platt"] < cal["nll_uncalibrated"]
    assert cal["val_f1_opt_macro_f1"] == 1.0
