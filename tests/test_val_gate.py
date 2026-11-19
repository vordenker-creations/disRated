from __future__ import annotations

import numpy as np

from disrated.val_gate import evaluate_val_gate, fit_validation_calibration


def test_val_gate_selects_smallest_k_meeting_non_inferiority():
    rng = np.random.default_rng(42)
    n = 200
    y_true = np.array([i % 2 == 0 for i in range(n)], dtype=bool)

    # Probe is confident & accurate on the outer 80%, wrong on the inner 20%
    probe_scores = np.where(y_true, 0.90, 0.10)
    uncertain = np.arange(0, 40)  # 20% of 200
    probe_scores[uncertain] = np.where(y_true[uncertain], 0.48, 0.52)

    # VLM is accurate everywhere
    vlm_scores = np.where(y_true, 0.85, 0.15)
    # Add slight noise on confident probe region so VLM != 1.000
    vlm_scores[40:44] = np.where(y_true[40:44], 0.3, 0.7)

    res = evaluate_val_gate(y_true, probe_scores, vlm_scores, tau_vlm=0.50, n_boot=200, seed=0)
    assert res["gate_passed_any"] is True
    assert res["selection_rule"] == "smallest_k_within_0.030_of_vlm_and_beats_probe"
    assert res["selected"]["k"] == 0.20
    assert res["selected"]["in_band"]["acc_a"] > res["selected"]["in_band"]["acc_b"]


def test_val_gate_marks_falsified_when_vlm_never_beats_probe_in_band():
    n = 100
    y_true = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    probe_scores = np.where(y_true, 0.85, 0.15)
    # Put 30 images near boundary where probe is 70% right, VLM is 30% right
    probe_scores[:30] = np.where(y_true[:30], 0.52, 0.48)
    vlm_scores = np.where(y_true, 0.20, 0.80)  # inverted VLM

    res = evaluate_val_gate(y_true, probe_scores, vlm_scores, tau_vlm=0.50, n_boot=200, seed=0)
    assert res["gate_passed_any"] is False
    assert res["selection_rule"] == "falsified_at_gate"
    assert res["selected"] is None


def test_fit_validation_calibration_recovers_threshold_shift():
    n = 200
    y_true = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    # Shifted VLM scores: positives ~ 0.45, negatives ~ 0.15 -> optimal threshold between 0.15 and 0.45
    vlm_scores = np.where(y_true, 0.45, 0.15)
    cal = fit_validation_calibration(y_true, vlm_scores)
    assert 0.20 < cal["platt_threshold"] < 0.40
    assert cal["nll_platt"] < cal["nll_uncalibrated"]
