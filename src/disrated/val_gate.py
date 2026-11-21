from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .data.vhd11k import Sample, concept_taxonomy, load_vhd11k
from .metrics.scoring import binary_metrics, read_predictions
from .risk_coverage import logit
from .schema import Prediction

CANDIDATE_KS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DEFAULT_N_RAND: int = 200


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _confusion_indicators(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return float32 (N, 4) matrix with columns [tp, fp, tn, fn]."""
    out = np.empty((len(y_true), 4), dtype=np.float32)
    out[:, 0] = y_true & y_pred
    out[:, 1] = (~y_true) & y_pred
    out[:, 2] = (~y_true) & (~y_pred)
    out[:, 3] = y_true & (~y_pred)
    return out


def _macro_f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """Vectorized Macro-F1 from (..., 4) [tp, fp, tn, fn] count arrays."""
    tp = counts[..., 0]
    fp = counts[..., 1]
    tn = counts[..., 2]
    fn = counts[..., 3]
    d_pos = 2.0 * tp + fp + fn
    d_neg = 2.0 * tn + fn + fp
    f1_pos = np.where(d_pos > 0.0, (2.0 * tp) / np.where(d_pos > 0.0, d_pos, 1.0), 0.0)
    f1_neg = np.where(d_neg > 0.0, (2.0 * tn) / np.where(d_neg > 0.0, d_neg, 1.0), 0.0)
    return 0.5 * (f1_pos + f1_neg)


def build_bootstrap_weights(
    n: int,
    comp_ids: Sequence[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Build float32 (n_boot, n) sample multiplicity weights via cluster (or i.i.d.) bootstrap."""
    rng = np.random.default_rng(seed)
    if comp_ids is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        weights = np.zeros((n_boot, n), dtype=np.float32)
        rows = np.repeat(np.arange(n_boot), n)
        np.add.at(weights, (rows, idx.ravel()), 1.0)
        return weights

    if len(comp_ids) != n:
        raise ValueError(f"comp_ids length ({len(comp_ids)}) != n ({n})")

    uniq_comps = sorted(set(comp_ids))
    comp_to_col = {c: i for i, c in enumerate(uniq_comps)}
    m = len(uniq_comps)
    # Incidence matrix G: (m, n), G[c, i] = 1 iff sample i is in component c
    G = np.zeros((m, n), dtype=np.float32)
    for sample_i, cid in enumerate(comp_ids):
        G[comp_to_col[cid], sample_i] = 1.0

    chosen = rng.integers(0, m, size=(n_boot, m))
    H = np.zeros((n_boot, m), dtype=np.float32)
    rows = np.repeat(np.arange(n_boot), m)
    np.add.at(H, (rows, chosen.ravel()), 1.0)
    return H @ G


def _ci95(arr: np.ndarray) -> list[float]:
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def paired_bootstrap_acc_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    comp_ids: Sequence[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired non-parametric (cluster or i.i.d.) bootstrap for accuracy(A) - accuracy(B)."""
    n = len(y_true)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "n_clusters": 0,
            "acc_a": nan,
            "acc_b": nan,
            "diff": nan,
            "acc_a_ci95": [nan, nan],
            "acc_b_ci95": [nan, nan],
            "diff_ci95": [nan, nan],
            "p_one_sided_le_0": nan,
        }

    ca = (pred_a == y_true).astype(np.float32)
    cb = (pred_b == y_true).astype(np.float32)
    W = build_bootstrap_weights(n, comp_ids=comp_ids, n_boot=n_boot, seed=seed)
    denom = W.sum(axis=1)
    acc_a_boot = (W @ ca) / denom
    acc_b_boot = (W @ cb) / denom
    diff_boot = acc_a_boot - acc_b_boot
    n_clusters = len(set(comp_ids)) if comp_ids is not None else n
    return {
        "n": int(n),
        "n_clusters": int(n_clusters),
        "acc_a": float(ca.mean()),
        "acc_b": float(cb.mean()),
        "diff": float(ca.mean() - cb.mean()),
        "acc_a_ci95": _ci95(acc_a_boot),
        "acc_b_ci95": _ci95(acc_b_boot),
        "diff_ci95": _ci95(diff_boot),
        "p_one_sided_le_0": float(np.mean(diff_boot <= 0.0)),
    }


def paired_bootstrap_macro_f1_diffs(
    y_true: np.ndarray,
    pred_probe: np.ndarray,
    pred_vlm: np.ndarray,
    pred_cascade: np.ndarray,
    n_esc: int,
    comp_ids: Sequence[str] | None = None,
    n_boot: int = 10_000,
    n_rand: int = DEFAULT_N_RAND,
    seed: int = 0,
    weights: np.ndarray | None = None,
    pred_vlm_for_random: np.ndarray | None = None,
) -> dict[str, Any]:
    """Paired cluster/i.i.d. bootstrap CIs for all 4 arms using Monte-Carlo random routing.

    `pred_vlm` is used for the standalone `vlm-only` arm (where refusals must count as errors);
    `pred_vlm_for_random` (defaults to `pred_vlm`) is used when simulating random escalation with
    cascade-consistent refusal fallback.
    """
    n = len(y_true)
    W = weights if weights is not None else build_bootstrap_weights(n, comp_ids=comp_ids, n_boot=n_boot, seed=seed)
    vlm_rand_src = pred_vlm if pred_vlm_for_random is None else pred_vlm_for_random

    fixed_arms = {
        "probe": pred_probe,
        "vlm": pred_vlm,
        "cascade": pred_cascade,
    }
    point: dict[str, float] = {k: _macro_f1_np(y_true, p) for k, p in fixed_arms.items()}
    boot: dict[str, np.ndarray] = {}
    for k, p in fixed_arms.items():
        counts = W @ _confusion_indicators(y_true, p)
        boot[k] = _macro_f1_from_counts(counts)

    # Monte-Carlo random-control across n_rand deterministic masks of exact size n_esc.
    if n_esc <= 0:
        point["random"] = point["probe"]
        boot["random"] = boot["probe"].copy()
        rand_std = 0.0
        rand_acc = float((pred_probe == y_true).mean())
    elif n_esc >= n:
        point["random"] = _macro_f1_np(y_true, vlm_rand_src)
        boot["random"] = _macro_f1_from_counts(W @ _confusion_indicators(y_true, vlm_rand_src))
        rand_std = 0.0
        rand_acc = float((vlm_rand_src == y_true).mean())
    else:
        rng_r = np.random.default_rng(seed)
        c_rands = np.empty((n, n_rand * 4), dtype=np.float32)
        point_f1s = np.empty(n_rand, dtype=np.float64)
        point_accs = np.empty(n_rand, dtype=np.float64)
        for r in range(n_rand):
            rmask = np.zeros(n, dtype=bool)
            rmask[rng_r.choice(n, size=n_esc, replace=False)] = True
            pr = np.where(rmask, vlm_rand_src, pred_probe)
            point_f1s[r] = _macro_f1_np(y_true, pr)
            point_accs[r] = float((pr == y_true).mean())
            c_rands[:, r * 4 : (r + 1) * 4] = _confusion_indicators(y_true, pr)
        point["random"] = float(point_f1s.mean())
        rand_std = float(point_f1s.std())
        rand_acc = float(point_accs.mean())

        # Exact expected Macro-F1 of the random-routing policy on every bootstrap replicate.
        counts_r = (W @ c_rands).reshape(len(W), n_rand, 4)
        boot["random"] = _macro_f1_from_counts(counts_r).mean(axis=1)

    diffs = {
        "cascade_minus_probe": (point["cascade"] - point["probe"], boot["cascade"] - boot["probe"]),
        "cascade_minus_vlm": (point["cascade"] - point["vlm"], boot["cascade"] - boot["vlm"]),
        "cascade_minus_random": (point["cascade"] - point["random"], boot["cascade"] - boot["random"]),
    }
    return {
        "bootstrap_unit": "component_cluster" if comp_ids is not None else "iid_image",
        "n_clusters": int(len(set(comp_ids)) if comp_ids is not None else n),
        "n_rand_masks": int(n_rand),
        "random_mask_f1_std": rand_std,
        "random_acc": rand_acc,
        "macro_f1": {
            k: {"point": float(point[k]), "ci95": _ci95(boot[k])} for k in ("probe", "vlm", "cascade", "random")
        },
        "paired_diffs": {k: {"diff": float(val), "ci95": _ci95(arr)} for k, (val, arr) in diffs.items()},
    }


def _req_score(p: Prediction) -> float:
    if p.score is None:
        raise ValueError(f"Missing score for sample {p.sample_id!r}")
    return float(p.score)


def extract_vlm_logits(preds: dict[str, Prediction], sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (scores, raw_logits, logit_source). Uses unclipped extra['logit_diff'] when present on all rows."""
    scores = np.array([_req_score(preds[sid]) for sid in sample_ids], dtype=np.float64)
    has_raw = all("logit_diff" in preds[sid].extra and preds[sid].extra["logit_diff"] is not None for sid in sample_ids)
    if has_raw:
        raw_logits = np.array([float(preds[sid].extra["logit_diff"]) for sid in sample_ids], dtype=np.float64)
        return scores, raw_logits, "extra.logit_diff"
    return scores, logit(scores), "clipped_score_logit"


def fit_validation_calibration(
    y_true: np.ndarray,
    vlm_scores: np.ndarray,
    vlm_logits: np.ndarray | None = None,
    logit_source: str = "clipped_score_logit",
) -> dict[str, Any]:
    """Fit 2-parameter Platt scaling on VLM logits (unclipped when available) and log reference sweeps."""
    z = logit(vlm_scores) if vlm_logits is None else np.asarray(vlm_logits, dtype=np.float64)
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(z.reshape(-1, 1), y_true.astype(int))
    a = float(np.asarray(lr.coef_).ravel()[0])
    b = float(np.asarray(lr.intercept_).ravel()[0])
    # sigmoid(a * z_cut + b) = 0.5  =>  z_cut = -b / a
    z_platt_cut = float(-b / a) if a > 0 else 0.0
    tau_platt = float(1.0 / (1.0 + np.exp(-np.clip(z_platt_cut, -60.0, 60.0))))

    p_uncal = np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))), 1e-12, 1.0 - 1e-12)
    p_platt = np.clip(lr.predict_proba(z.reshape(-1, 1))[:, 1], 1e-12, 1.0 - 1e-12)

    # Pure 1-parameter temperature scaling (no intercept shift)
    temps = np.concatenate([np.linspace(0.25, 4.0, 151), np.linspace(4.25, 16.0, 48)])
    nlls = [float(log_loss(y_true, 1.0 / (1.0 + np.exp(-np.clip(z / t, -60.0, 60.0))), labels=[0, 1])) for t in temps]
    best_t_idx = int(np.argmin(nlls))
    temp_only = float(temps[best_t_idx])

    # Direct logit-space threshold sweep on val for reference (covers conservative VLMs too)
    lo_q, hi_q = float(np.quantile(z, 0.02)), float(np.quantile(z, 0.98))
    z_grid = np.unique(np.concatenate([np.linspace(lo_q, hi_q, 161), np.array([0.0, z_platt_cut])]))
    best_z_cut, best_f1 = 0.0, -1.0
    for zc in z_grid:
        f1 = _macro_f1_np(y_true, z >= zc)
        if f1 > best_f1:
            best_f1 = f1
            best_z_cut = float(zc)
    best_tau = float(1.0 / (1.0 + np.exp(-np.clip(best_z_cut, -60.0, 60.0))))

    return {
        "logit_source": logit_source,
        "platt_slope": a,
        "platt_intercept": b,
        "platt_z_cut": z_platt_cut,
        "platt_threshold": tau_platt,
        "temp_only": temp_only,
        "nll_uncalibrated": float(log_loss(y_true, p_uncal, labels=[0, 1])),
        "nll_temp_only": float(nlls[best_t_idx]),
        "nll_platt": float(log_loss(y_true, p_platt, labels=[0, 1])),
        "val_f1_opt_z_cut": best_z_cut,
        "val_f1_opt_threshold": best_tau,
        "val_f1_opt_macro_f1": float(best_f1),
        "vlm_roc_auc": float(roc_auc_score(y_true, z)) if len(np.unique(y_true)) == 2 else float("nan"),
    }


def _slice_in_band_breakdown(
    y_true: np.ndarray,
    pred_probe: np.ndarray,
    pred_vlm: np.ndarray,
    band: np.ndarray,
    labels: Sequence[str | None],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    keys = sorted({k for k in labels if k is not None})
    lbl_arr = np.array([k if k is not None else "" for k in labels])
    for key in keys:
        m_in = band & (lbl_arr == key)
        m_out = (~band) & (lbl_arr == key)
        out[key] = {
            "in_band_n": int(m_in.sum()),
            "in_band_probe_acc": float((pred_probe[m_in] == y_true[m_in]).mean()) if m_in.any() else None,
            "in_band_vlm_acc": float((pred_vlm[m_in] == y_true[m_in]).mean()) if m_in.any() else None,
            "resolved_n": int(m_out.sum()),
            "resolved_probe_acc": float((pred_probe[m_out] == y_true[m_out]).mean()) if m_out.any() else None,
            "resolved_vlm_acc": float((pred_vlm[m_out] == y_true[m_out]).mean()) if m_out.any() else None,
        }
    return out


def evaluate_val_gate(
    y_true: np.ndarray,
    probe_scores: np.ndarray,
    vlm_scores: np.ndarray,
    tau_vlm: float = 0.50,
    vlm_logits: np.ndarray | None = None,
    z_vlm_cut: float | None = None,
    comp_ids: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    taxonomies: Sequence[str | None] | None = None,
    ks: tuple[float, ...] = CANDIDATE_KS,
    n_boot: int = 10_000,
    n_rand: int = DEFAULT_N_RAND,
    seed: int = 0,
) -> dict[str, Any]:
    """Run Hypothesis Section 4 Validation Gate + CI-Aligned Operating Point selection on val."""
    n = len(y_true)
    z_probe = logit(probe_scores)
    abs_z = np.abs(z_probe)
    pred_probe = probe_scores >= 0.50

    if vlm_logits is not None and z_vlm_cut is not None:
        pred_vlm = np.asarray(vlm_logits, dtype=np.float64) >= float(z_vlm_cut)
        eff_z_cut = float(z_vlm_cut)
        eff_tau = float(1.0 / (1.0 + np.exp(-np.clip(eff_z_cut, -60.0, 60.0))))
    else:
        pred_vlm = vlm_scores >= float(tau_vlm)
        eff_tau = float(tau_vlm)
        eff_z_cut = float(logit(np.array([eff_tau]))[0])

    probe_m = binary_metrics(y_true.tolist(), pred_probe.tolist())
    vlm_m = binary_metrics(y_true.tolist(), pred_vlm.tolist())

    # Reuse one deterministic bootstrap weight matrix across candidates for exact paired consistency.
    W = build_bootstrap_weights(n, comp_ids=comp_ids, n_boot=n_boot, seed=seed)

    candidates: list[dict[str, Any]] = []
    for k in ks:
        c_k = float(np.quantile(abs_z, k))
        band = abs_z < c_k
        n_esc = int(band.sum())
        pred_cascade = np.where(band, pred_vlm, pred_probe)

        band_comps = [comp_ids[int(i)] for i in np.where(band)[0]] if comp_ids is not None else None
        in_band = paired_bootstrap_acc_diff(
            y_true[band],
            pred_vlm[band],
            pred_probe[band],
            comp_ids=band_comps,
            n_boot=n_boot,
            seed=seed,
        )
        full_boot = paired_bootstrap_macro_f1_diffs(
            y_true,
            pred_probe,
            pred_vlm,
            pred_cascade,
            n_esc=n_esc,
            comp_ids=comp_ids,
            n_boot=n_boot,
            n_rand=n_rand,
            seed=seed,
            weights=W,
        )
        res = ~band
        resolved_acc = float((pred_probe[res] == y_true[res]).mean()) if res.any() else float("nan")
        vlm_resolved_acc = float((pred_vlm[res] == y_true[res]).mean()) if res.any() else float("nan")
        union_all = float(((pred_probe == y_true) | (pred_vlm == y_true)).mean())
        union_band = (
            float(((pred_probe[band] == y_true[band]) | (pred_vlm[band] == y_true[band])).mean())
            if band.any()
            else float("nan")
        )

        gate_ok = in_band["n"] > 0 and (in_band["acc_a"] > in_band["acc_b"])
        dp = full_boot["paired_diffs"]["cascade_minus_probe"]
        dv = full_boot["paired_diffs"]["cascade_minus_vlm"]
        dr = full_boot["paired_diffs"]["cascade_minus_random"]
        ci_aligned = bool(gate_ok and (dp["ci95"][0] > 0.0) and (dv["ci95"][0] >= -0.030) and (dr["ci95"][0] > 0.0))

        cand: dict[str, Any] = {
            "k": float(k),
            "c_k": c_k,
            "n_esc": n_esc,
            "esc_rate": float(n_esc / n),
            "resolved_acc": resolved_acc,
            "vlm_resolved_acc": vlm_resolved_acc,
            "union_oracle_all": union_all,
            "union_oracle_in_band": union_band,
            "gate_passed": bool(gate_ok),
            "ci_aligned_all_three": ci_aligned,
            "in_band": in_band,
            "cascade_acc": float((pred_cascade == y_true).mean()),
            "random_acc": float(full_boot["random_acc"]),
            "arms": full_boot,
        }
        if sources is not None:
            cand["by_source"] = _slice_in_band_breakdown(y_true, pred_probe, pred_vlm, band, sources)
        if taxonomies is not None:
            cand["by_taxonomy"] = _slice_in_band_breakdown(y_true, pred_probe, pred_vlm, band, taxonomies)
        candidates.append(cand)

    passing = [c for c in candidates if c["gate_passed"]]
    selected: dict[str, Any] | None = None
    selection_rule = "falsified_at_gate"
    if passing:
        ci_ok = [c for c in passing if c["ci_aligned_all_three"]]
        if ci_ok:
            # Maximize val cascade macro_f1 among CI-passing candidates (tie-break: smaller k).
            selected = max(
                ci_ok,
                key=lambda c: (round(c["arms"]["macro_f1"]["cascade"]["point"], 9), -c["k"]),
            )
            selection_rule = "max_val_cascade_macro_f1_meeting_ci_gates"
        else:
            selected = max(
                passing,
                key=lambda c: (round(c["arms"]["macro_f1"]["cascade"]["point"], 9), -c["k"]),
            )
            selection_rule = "fallback_max_val_cascade_macro_f1"

    return {
        "tau_vlm": eff_tau,
        "z_vlm_cut": eff_z_cut,
        "n_val": int(n),
        "bootstrap_unit": "component_cluster" if comp_ids is not None else "iid_image",
        "n_clusters": int(len(set(comp_ids)) if comp_ids is not None else n),
        "n_rand_masks": int(n_rand),
        "probe_val": {"accuracy": float(probe_m["accuracy"]), "macro_f1": float(probe_m["macro_f1"])},
        "vlm_val": {"accuracy": float(vlm_m["accuracy"]), "macro_f1": float(vlm_m["macro_f1"])},
        "gate_passed_any": bool(passing),
        "selection_rule": selection_rule,
        "selected": selected,
        "candidates": candidates,
    }


def _read_sidecar_meta(pred_path: Path) -> dict[str, Any] | None:
    mp = pred_path.with_suffix(pred_path.suffix + ".meta.json")
    if not mp.is_file():
        return None
    return json.loads(mp.read_text(encoding="utf-8"))


def _extract_model_identity(
    pred_path: Path,
    preds: dict[str, Prediction],
    split_ids: Sequence[str],
    split_name: str,
    role: str,
    require_vlm_meta: bool = True,
) -> dict[str, Any]:
    """Extract and validate model identity metadata to prevent cross-model or OOF mix-ups."""
    split_set = set(split_ids)
    extra_ids = sorted(set(preds.keys()) - split_set)
    sidecar = _read_sidecar_meta(pred_path)

    if sidecar is not None:
        run_info = sidecar.get("run")
        if isinstance(run_info, dict) and "split" in run_info and run_info["split"] not in (split_name, "all"):
            raise SystemExit(
                f"ERROR: {role} sidecar metadata indicates split='{run_info['split']}', expected '{split_name}' ({pred_path})."
            )

    if role == "probe":
        has_fold = any("fold" in p.extra or "components_sha256" in p.extra for p in preds.values())
        if has_fold:
            raise SystemExit(
                f"ERROR: {role} predictions appear to be out-of-fold (contain 'fold'/'components_sha256' metadata): {pred_path}. "
                f"Pass fixed train-fit '{split_name}' predictions instead."
            )
        if extra_ids:
            raise SystemExit(
                f"ERROR: {role} predictions contain {len(extra_ids)} unexpected sample(s) outside split '{split_name}' "
                f"(first: {extra_ids[0]}): {pred_path}."
            )
        cs = sorted({float(preds[sid].extra["c"]) for sid in split_ids if "c" in preds[sid].extra})
        if len(cs) > 1:
            raise SystemExit(f"ERROR: {role} predictions mix multiple probe C values: {cs} ({pred_path}).")
        sidecar_norm = (
            {"moderator": sidecar.get("moderator"), "config": sidecar.get("config")} if sidecar is not None else None
        )
        return {"c": cs[0] if cs else None, "sidecar": sidecar_norm}

    # role == "vlm"
    if sidecar is None:
        if require_vlm_meta:
            raise SystemExit(
                f"ERROR: VLM predictions are missing required .meta.json sidecar for model provenance: {pred_path}.meta.json"
            )
        return {"moderator": None, "config": None}
    if "moderator" not in sidecar or "config" not in sidecar:
        raise SystemExit(f"ERROR: VLM sidecar {pred_path}.meta.json is missing 'moderator' or 'config'.")
    return {"moderator": sidecar["moderator"], "config": sidecar["config"]}


def verify_model_identity_match(
    role: str,
    val_meta: dict[str, Any] | None,
    test_meta: dict[str, Any] | None,
) -> None:
    """Verify that held-out test predictions were produced by the exact same model/probe locked on val."""
    if val_meta is None or test_meta is None:
        raise SystemExit(
            f"ERROR: Missing {role}_meta provenance for cross-split verification "
            f"(val={val_meta is not None}, test={test_meta is not None}). Re-lock Phase 3 validation gate."
        )

    if role == "vlm":
        if val_meta.get("moderator") != test_meta.get("moderator"):
            raise SystemExit(
                f"ERROR: VLM moderator mismatch vs locked validation gate.\n"
                f"  Gate VLM: {val_meta.get('moderator')!r}\n"
                f"  Test VLM: {test_meta.get('moderator')!r}"
            )
        v_cfg = dict(val_meta.get("config") or {})
        t_cfg = dict(test_meta.get("config") or {})
        # Allow optional HF commit revision to be added on newer runs if absent on earlier val run;
        # if both specify revision, they must match identically.
        if ("revision" in v_cfg) != ("revision" in t_cfg):
            v_cfg.pop("revision", None)
            t_cfg.pop("revision", None)
        if v_cfg != t_cfg:
            raise SystemExit(
                f"ERROR: VLM config mismatch vs locked validation gate.\n  Gate config: {v_cfg}\n  Test config: {t_cfg}"
            )
        return

    # role == "probe"
    if val_meta.get("c") != test_meta.get("c"):
        raise SystemExit(
            f"ERROR: Probe regularization C mismatch vs locked validation gate "
            f"(gate C={val_meta.get('c')!r} != test C={test_meta.get('c')!r})."
        )
    v_sc = val_meta.get("sidecar")
    t_sc = test_meta.get("sidecar")
    if v_sc is not None and t_sc is not None and v_sc != t_sc:
        raise SystemExit(
            f"ERROR: Probe sidecar config mismatch vs locked validation gate.\n"
            f"  Gate probe: {v_sc}\n"
            f"  Test probe: {t_sc}"
        )
    if (v_sc is None) != (t_sc is None):
        raise SystemExit(
            f"ERROR: Probe sidecar presence mismatch vs locked validation gate "
            f"(gate has sidecar={v_sc is not None}, test has sidecar={t_sc is not None})."
        )


def load_split_aligned(
    dataset: Path,
    splits_path: Path,
    split_name: str,
    probe_path: Path,
    vlm_path: Path,
    components_path: Path | None = None,
    allow_refusals: bool = False,
    require_vlm_meta: bool = True,
) -> dict[str, Any]:
    """Load split ground truth, component IDs, and strictly verified probe/VLM predictions."""
    splits_path = splits_path.expanduser()
    splits_raw = splits_path.read_text(encoding="utf-8")
    splits_sha256 = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    splits_data = json.loads(splits_raw)
    if split_name not in splits_data:
        raise SystemExit(f"ERROR: split '{split_name}' not found in {splits_path}")
    split_ids: list[str] = splits_data[split_name]

    samples, _ = load_vhd11k(dataset.expanduser())
    by_id: dict[str, Sample] = {s.sample_id: s for s in samples}
    missing_ds = [sid for sid in split_ids if sid not in by_id]
    if missing_ds:
        raise SystemExit(
            f"ERROR: {len(missing_ds)} sample(s) from split '{split_name}' missing in dataset "
            f"{dataset} (first: {missing_ds[0]})"
        )

    probe_path = probe_path.expanduser()
    vlm_path = vlm_path.expanduser()
    probe_preds = read_predictions(probe_path)
    vlm_preds = read_predictions(vlm_path)
    probe_sha256 = sha256_file(probe_path)
    vlm_sha256 = sha256_file(vlm_path)

    probe_meta = _extract_model_identity(probe_path, probe_preds, split_ids, split_name, role="probe")
    vlm_meta = _extract_model_identity(
        vlm_path, vlm_preds, split_ids, split_name, role="vlm", require_vlm_meta=require_vlm_meta
    )

    for label, preds in (("probe", probe_preds), ("vlm", vlm_preds)):
        all_hashes = {p.extra["splits_sha256"] for p in preds.values() if "splits_sha256" in p.extra}
        if all_hashes and all_hashes != {splits_sha256}:
            raise SystemExit(
                f"ERROR: Splits sha256 mismatch in {label} predictions: {sorted(all_hashes)} != {splits_sha256}"
            )
        missing_pred = [sid for sid in split_ids if sid not in preds]
        if missing_pred:
            raise SystemExit(
                f"ERROR: {label} predictions missing {len(missing_pred)} '{split_name}' sample(s) (first: {missing_pred[0]})"
            )
        missing_hash = [sid for sid in split_ids if "splits_sha256" not in preds[sid].extra]
        if missing_hash:
            raise SystemExit(
                f"ERROR: {label} predictions missing splits_sha256 metadata on {len(missing_hash)} "
                f"'{split_name}' sample(s) (first: {missing_hash[0]})"
            )
        bad_err = [sid for sid in split_ids if preds[sid].error is not None]
        if bad_err:
            raise SystemExit(
                f"ERROR: {label} predictions have error!=None on {len(bad_err)} '{split_name}' sample(s) (first: {bad_err[0]})"
            )
        if not allow_refusals or label == "probe":
            unusable = [sid for sid in split_ids if preds[sid].refused or preds[sid].score is None]
            if unusable:
                raise SystemExit(
                    f"ERROR: {label} predictions refused/missing score on {len(unusable)} "
                    f"'{split_name}' sample(s) (first: {unusable[0]})"
                )

    comp_ids: list[str] | None = None
    comp_sha256: str | None = None
    if components_path is not None:
        cp = components_path.expanduser()
        if not cp.is_file():
            raise SystemExit(f"ERROR: components file not found: {cp}")
        comp_raw = cp.read_text(encoding="utf-8")
        comp_sha256 = hashlib.sha256(comp_raw.encode("utf-8")).hexdigest()
        comp_map = json.loads(comp_raw)
        missing_c = [sid for sid in split_ids if sid not in comp_map]
        if missing_c:
            raise SystemExit(
                f"ERROR: components file {cp} missing {len(missing_c)} '{split_name}' sample(s) (first: {missing_c[0]})"
            )
        comp_ids = [str(comp_map[sid]) for sid in split_ids]

    y_true = np.array([by_id[sid].harmful for sid in split_ids], dtype=bool)
    p_probe = np.array([_req_score(probe_preds[sid]) for sid in split_ids], dtype=np.float64)
    sources = [by_id[sid].source for sid in split_ids]
    taxonomies = [concept_taxonomy(by_id[sid].concept) for sid in split_ids]
    concepts = [by_id[sid].concept for sid in split_ids]
    vlm_refused = np.array([bool(vlm_preds[sid].refused) for sid in split_ids], dtype=bool)

    if allow_refusals and vlm_refused.any():
        p_vlm = np.array(
            [
                np.nan if (vlm_preds[sid].refused or vlm_preds[sid].score is None) else _req_score(vlm_preds[sid])
                for sid in split_ids
            ],
            dtype=np.float64,
        )
        has_raw = all(
            vlm_preds[sid].refused
            or ("logit_diff" in vlm_preds[sid].extra and vlm_preds[sid].extra["logit_diff"] is not None)
            for sid in split_ids
        )
        if has_raw:
            z_vlm = np.array(
                [np.nan if vlm_preds[sid].refused else float(vlm_preds[sid].extra["logit_diff"]) for sid in split_ids],
                dtype=np.float64,
            )
            logit_source = "extra.logit_diff"
        else:
            z_vlm = np.where(np.isfinite(p_vlm), logit(np.nan_to_num(p_vlm, nan=0.5)), np.nan)
            logit_source = "clipped_score_logit"
    else:
        p_vlm, z_vlm, logit_source = extract_vlm_logits(vlm_preds, split_ids)

    return {
        "split_ids": split_ids,
        "y_true": y_true,
        "p_probe": p_probe,
        "p_vlm": p_vlm,
        "z_vlm": z_vlm,
        "vlm_refused": vlm_refused,
        "logit_source": logit_source,
        "sources": sources,
        "concepts": concepts,
        "taxonomies": taxonomies,
        "comp_ids": comp_ids,
        "splits_sha256": splits_sha256,
        "components_sha256": comp_sha256,
        "probe_sha256": probe_sha256,
        "vlm_sha256": vlm_sha256,
        "probe_meta": probe_meta,
        "vlm_meta": vlm_meta,
    }


def _print_summary(label: str, res: dict[str, Any]) -> None:
    print(
        f"\n=== VALIDATION GATE SUMMARY ({label}, tau_vlm={res['tau_vlm']:.5f}, "
        f"z_cut={res['z_vlm_cut']:+.4f}, boot={res['bootstrap_unit']} [{res['n_clusters']} clusters]) ==="
    )
    print(f"probe-only: acc={res['probe_val']['accuracy']:.3f} | macroF1={res['probe_val']['macro_f1']:.3f}")
    print(f"vlm-only:   acc={res['vlm_val']['accuracy']:.3f} | macroF1={res['vlm_val']['macro_f1']:.3f}")
    print(
        f"{'k':>5} {'c_k':>7} {'n':>4} | {'band VLM/Prb (diff [95% CI])':>32} | "
        f"{'casF1':>6} {'rndF1':>6} {'dPrb [95% CI]':>19} {'dVlm [95% CI]':>19} {'dRnd [95% CI]':>19} | {'ok':>4}"
    )
    print("-" * 138)
    for c in res["candidates"]:
        ib = c["in_band"]
        arms = c["arms"]
        cas_f1 = arms["macro_f1"]["cascade"]["point"]
        rnd_f1 = arms["macro_f1"]["random"]["point"]
        dp = arms["paired_diffs"]["cascade_minus_probe"]
        dv = arms["paired_diffs"]["cascade_minus_vlm"]
        dr = arms["paired_diffs"]["cascade_minus_random"]
        ib_str = f"{ib['acc_a']:.3f}/{ib['acc_b']:.3f} ({ib['diff']:+.3f} [{ib['diff_ci95'][0]:+.3f},{ib['diff_ci95'][1]:+.3f}])"
        dp_str = f"{dp['diff']:+.3f} [{dp['ci95'][0]:+.3f},{dp['ci95'][1]:+.3f}]"
        dv_str = f"{dv['diff']:+.3f} [{dv['ci95'][0]:+.3f},{dv['ci95'][1]:+.3f}]"
        dr_str = f"{dr['diff']:+.3f} [{dr['ci95'][0]:+.3f},{dr['ci95'][1]:+.3f}]"
        flag = "CI3" if c["ci_aligned_all_three"] else ("GATE" if c["gate_passed"] else "FAIL")
        print(
            f"{c['k']:5.2f} {c['c_k']:7.4f} {c['n_esc']:4d} | {ib_str:>32} | "
            f"{cas_f1:6.3f} {rnd_f1:6.3f} {dp_str:>19} {dv_str:>19} {dr_str:>19} | {flag:>4}"
        )
    sel = res["selected"]
    if sel is None:
        print(f"Selection ({res['selection_rule']}): NO candidate passed validation gate.")
    else:
        print(
            f"Selection ({res['selection_rule']}): k*={sel['k']:.2f}, c*={sel['c_k']:.6f}, "
            f"cascade_macroF1={sel['arms']['macro_f1']['cascade']['point']:.4f} "
            f"(in-band VLM={sel['in_band']['acc_a']:.3f} vs out-of-band VLM={sel['vlm_resolved_acc']:.3f})"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated val-gate")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    ap.add_argument("--components", type=Path, default=Path("embeddings/components.json"))
    ap.add_argument(
        "--no-cluster-boot", action="store_true", help="use i.i.d. image bootstrap instead of component clusters"
    )
    ap.add_argument("--probe", type=Path, default=Path("results/clip-b16-v2-val.jsonl"))
    ap.add_argument("--vlm", type=Path, default=Path("results/d1-2b-mlx-v2-val.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/phase3_val_gate.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-rand", type=int, default=DEFAULT_N_RAND)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tau-mode",
        choices=("auto", "uncalibrated", "platt", "val_f1_opt"),
        default="auto",
        help="How to lock tau_vlm* ('auto' picks uncalibrated 0.50 unless Platt improves val VLM macro_f1).",
    )
    a = ap.parse_args(argv)

    comp_path = None if a.no_cluster_boot else a.components
    aligned = load_split_aligned(
        dataset=a.dataset,
        splits_path=a.splits,
        split_name="val",
        probe_path=a.probe,
        vlm_path=a.vlm,
        components_path=comp_path,
    )
    y_true = aligned["y_true"]
    p_probe = aligned["p_probe"]
    p_vlm = aligned["p_vlm"]
    z_vlm = aligned["z_vlm"]
    comp_ids = aligned["comp_ids"]
    sources = aligned["sources"]
    taxonomies = aligned["taxonomies"]

    cal = fit_validation_calibration(y_true, p_vlm, vlm_logits=z_vlm, logit_source=aligned["logit_source"])
    print(
        f"VLM calibration on val (N={len(y_true)}, logit_source={cal['logit_source']}): "
        f"ROC-AUC={cal['vlm_roc_auc']:.4f} | NLL uncal={cal['nll_uncalibrated']:.4f}, "
        f"temp={cal['nll_temp_only']:.4f} (T={cal['temp_only']:.2f}), "
        f"platt={cal['nll_platt']:.4f} (z_cut={cal['platt_z_cut']:+.4f}, tau_platt={cal['platt_threshold']:.5f}) | "
        f"val_f1_opt z_cut={cal['val_f1_opt_z_cut']:+.4f} (F1={cal['val_f1_opt_macro_f1']:.4f})"
    )

    common_kw: dict[str, Any] = {
        "vlm_logits": z_vlm,
        "comp_ids": comp_ids,
        "sources": sources,
        "taxonomies": taxonomies,
        "n_boot": a.n_boot,
        "n_rand": a.n_rand,
        "seed": a.seed,
    }
    uncal = evaluate_val_gate(y_true, p_probe, p_vlm, tau_vlm=0.50, z_vlm_cut=0.0, **common_kw)
    platt = evaluate_val_gate(
        y_true, p_probe, p_vlm, tau_vlm=cal["platt_threshold"], z_vlm_cut=cal["platt_z_cut"], **common_kw
    )
    f1_opt = evaluate_val_gate(
        y_true, p_probe, p_vlm, tau_vlm=cal["val_f1_opt_threshold"], z_vlm_cut=cal["val_f1_opt_z_cut"], **common_kw
    )

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
        "splits_sha256": aligned["splits_sha256"],
        "components": str(comp_path) if comp_path is not None else None,
        "components_sha256": aligned["components_sha256"],
        "probe_val_path": str(a.probe),
        "probe_val_sha256": aligned["probe_sha256"],
        "probe_meta": aligned["probe_meta"],
        "vlm_val_path": str(a.vlm),
        "vlm_val_sha256": aligned["vlm_sha256"],
        "vlm_meta": aligned["vlm_meta"],
        "logit_source": aligned["logit_source"],
        "bootstrap_unit": active["bootstrap_unit"],
        "n_boot": a.n_boot,
        "n_rand": a.n_rand,
        "seed": a.seed,
        "calibration": cal,
        "locked_operating_point": {
            "tau_mode": active_key,
            "tau_vlm_star": active["tau_vlm"],
            "z_vlm_cut_star": active["z_vlm_cut"],
            "gate_passed": sel is not None,
            "selection_rule": active["selection_rule"],
            "k_star": sel["k"] if sel else None,
            "c_star": sel["c_k"] if sel else None,
            "val_cascade_macro_f1": sel["arms"]["macro_f1"]["cascade"]["point"] if sel else None,
            "val_in_band_vlm_acc": sel["in_band"]["acc_a"] if sel else None,
            "val_in_band_probe_acc": sel["in_band"]["acc_b"] if sel else None,
            "val_out_of_band_vlm_acc": sel["vlm_resolved_acc"] if sel else None,
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
