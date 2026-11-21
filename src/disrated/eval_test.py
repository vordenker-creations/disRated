from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics.scoring import binary_metrics
from .risk_coverage import logit
from .val_gate import (
    DEFAULT_N_RAND,
    _slice_in_band_breakdown,
    load_split_aligned,
    paired_bootstrap_acc_diff,
    paired_bootstrap_macro_f1_diffs,
    sha256_file,
    verify_model_identity_match,
)


def _per_slice_arm_f1(
    y_true: np.ndarray,
    pred_probe: np.ndarray,
    pred_vlm: np.ndarray,
    pred_cascade: np.ndarray,
    labels: Sequence[str | None],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    keys = sorted({k for k in labels if k is not None})
    lbl_arr = np.array([k if k is not None else "" for k in labels])
    for key in keys:
        m = lbl_arr == key
        if not m.any():
            continue
        out[key] = {
            "n": int(m.sum()),
            "probe_macro_f1": float(binary_metrics(y_true[m].tolist(), pred_probe[m].tolist())["macro_f1"]),
            "vlm_macro_f1": float(binary_metrics(y_true[m].tolist(), pred_vlm[m].tolist())["macro_f1"]),
            "cascade_macro_f1": float(binary_metrics(y_true[m].tolist(), pred_cascade[m].tolist())["macro_f1"]),
            "probe_acc": float((pred_probe[m] == y_true[m]).mean()),
            "vlm_acc": float((pred_vlm[m] == y_true[m]).mean()),
            "cascade_acc": float((pred_cascade[m] == y_true[m]).mean()),
        }
    return out


def evaluate_locked_test(
    y_true: np.ndarray,
    probe_scores: np.ndarray,
    vlm_scores: np.ndarray,
    c_star: float,
    z_vlm_cut: float,
    vlm_logits: np.ndarray | None = None,
    vlm_refused: np.ndarray | None = None,
    comp_ids: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    taxonomies: Sequence[str | None] | None = None,
    budget_cap: float = 0.300,
    non_inferiority_margin: float = 0.030,
    n_boot: int = 10_000,
    n_rand: int = DEFAULT_N_RAND,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a locked (c_star, z_vlm_cut) operating point unchanged on held-out test data."""
    n = len(y_true)
    z_probe = logit(probe_scores)
    abs_z = np.abs(z_probe)
    pred_probe = probe_scores >= 0.50

    refused = np.zeros(n, dtype=bool) if vlm_refused is None else np.asarray(vlm_refused, dtype=bool)
    z_v = logit(np.nan_to_num(vlm_scores, nan=0.5)) if vlm_logits is None else np.asarray(vlm_logits, dtype=np.float64)
    # Standalone vlm-only and in-band VLM accuracy count refusals as errors (~y_true),
    # never giving the standalone VLM credit for Tier 0's answer.
    # Cascade and random-control fall back to Tier 0 on escalated refusals.
    pred_vlm_raw = np.where(refused, False, z_v >= float(z_vlm_cut))
    pred_vlm_standalone = np.where(refused, ~y_true, pred_vlm_raw)
    pred_vlm_for_cascade = np.where(refused, pred_probe, pred_vlm_raw)

    band = abs_z < float(c_star)
    res = ~band
    n_esc = int(band.sum())
    esc_rate = float(n_esc / n) if n else float("nan")
    pred_cascade = np.where(band, pred_vlm_for_cascade, pred_probe)

    band_comps = [comp_ids[int(i)] for i in np.where(band)[0]] if comp_ids is not None else None
    in_band_cluster = paired_bootstrap_acc_diff(
        y_true[band], pred_vlm_standalone[band], pred_probe[band], comp_ids=band_comps, n_boot=n_boot, seed=seed
    )
    arms_cluster = paired_bootstrap_macro_f1_diffs(
        y_true,
        pred_probe,
        pred_vlm_standalone,
        pred_cascade,
        n_esc=n_esc,
        comp_ids=comp_ids,
        n_boot=n_boot,
        n_rand=n_rand,
        seed=seed,
        pred_vlm_for_random=pred_vlm_for_cascade,
    )

    # Also compute i.i.d. bootstrap sensitivity whenever cluster IDs are supplied.
    if comp_ids is not None:
        in_band_iid = paired_bootstrap_acc_diff(
            y_true[band], pred_vlm_standalone[band], pred_probe[band], comp_ids=None, n_boot=n_boot, seed=seed
        )
        arms_iid = paired_bootstrap_macro_f1_diffs(
            y_true,
            pred_probe,
            pred_vlm_standalone,
            pred_cascade,
            n_esc=n_esc,
            comp_ids=None,
            n_boot=n_boot,
            n_rand=n_rand,
            seed=seed,
            pred_vlm_for_random=pred_vlm_for_cascade,
        )
    else:
        in_band_iid = in_band_cluster
        arms_iid = arms_cluster

    def _criteria_for(arms: dict[str, Any], in_band: dict[str, Any]) -> dict[str, Any]:
        dp = arms["paired_diffs"]["cascade_minus_probe"]
        dv = arms["paired_diffs"]["cascade_minus_vlm"]
        dr = arms["paired_diffs"]["cascade_minus_random"]
        c1 = bool(esc_rate <= budget_cap)
        c2 = bool(dp["diff"] > 0.0 and dp["ci95"][0] > 0.0)
        c3 = bool(dv["ci95"][0] >= -non_inferiority_margin)
        c4 = bool(dr["diff"] > 0.0 and dr["ci95"][0] > 0.0)
        c5 = bool(in_band["n"] > 0 and in_band["acc_a"] > in_band["acc_b"])
        return {
            "1_budget_cap": {"passed": c1, "esc_rate": esc_rate, "limit": budget_cap},
            "2_beats_probe": {"passed": c2, "diff": dp["diff"], "ci95": dp["ci95"]},
            "3_non_inferior_to_vlm": {
                "passed": c3,
                "diff": dv["diff"],
                "ci95": dv["ci95"],
                "margin": -non_inferiority_margin,
            },
            "4_beats_random": {"passed": c4, "diff": dr["diff"], "ci95": dr["ci95"]},
            "5_in_band_gain": {
                "passed": c5,
                "vlm_acc": in_band["acc_a"],
                "probe_acc": in_band["acc_b"],
                "diff": in_band["diff"],
                "ci95": in_band["diff_ci95"],
            },
            "all_passed": bool(c1 and c2 and c3 and c4 and c5),
        }

    probe_m = binary_metrics(y_true.tolist(), pred_probe.tolist())
    vlm_m = binary_metrics(y_true.tolist(), pred_vlm_standalone.tolist())
    cas_m = binary_metrics(y_true.tolist(), pred_cascade.tolist())

    res_out: dict[str, Any] = {
        "n_test": int(n),
        "c_star": float(c_star),
        "z_vlm_cut": float(z_vlm_cut),
        "tau_vlm": float(1.0 / (1.0 + np.exp(-np.clip(z_vlm_cut, -60.0, 60.0)))),
        "n_esc": n_esc,
        "esc_rate": esc_rate,
        "n_refused_all": int(refused.sum()),
        "n_refused_in_band": int((refused & band).sum()),
        "vlm_refusal_rate": float(refused.mean()) if n else 0.0,
        "resolved_probe_acc": float((pred_probe[res] == y_true[res]).mean()) if res.any() else float("nan"),
        "resolved_vlm_acc": float((pred_vlm_standalone[res] == y_true[res]).mean()) if res.any() else float("nan"),
        "union_oracle_all": float(((pred_probe == y_true) | (pred_vlm_standalone == y_true)).mean()),
        "union_oracle_in_band": (
            float(((pred_probe[band] == y_true[band]) | (pred_vlm_standalone[band] == y_true[band])).mean())
            if band.any()
            else float("nan")
        ),
        "metrics": {"probe": probe_m, "vlm": vlm_m, "cascade": cas_m},
        "primary": {
            "bootstrap_unit": arms_cluster["bootstrap_unit"],
            "n_clusters": arms_cluster["n_clusters"],
            "in_band": in_band_cluster,
            "arms": arms_cluster,
            "criteria": _criteria_for(arms_cluster, in_band_cluster),
        },
        "iid_sensitivity": {
            "bootstrap_unit": "iid_image",
            "in_band": in_band_iid,
            "arms": arms_iid,
            "criteria": _criteria_for(arms_iid, in_band_iid),
        },
    }
    if sources is not None:
        res_out["by_source"] = _per_slice_arm_f1(y_true, pred_probe, pred_vlm_standalone, pred_cascade, sources)
        res_out["by_source_band"] = _slice_in_band_breakdown(y_true, pred_probe, pred_vlm_standalone, band, sources)
    if taxonomies is not None:
        res_out["by_taxonomy"] = _per_slice_arm_f1(y_true, pred_probe, pred_vlm_standalone, pred_cascade, taxonomies)
        res_out["by_taxonomy_band"] = _slice_in_band_breakdown(
            y_true, pred_probe, pred_vlm_standalone, band, taxonomies
        )
    return res_out


def _print_test_report(title: str, ev: dict[str, Any], gate_passed: bool) -> None:
    prim = ev["primary"]
    arms = prim["arms"]
    crit = prim["criteria"]
    mf1 = arms["macro_f1"]
    print(
        f"\n=== PHASE 4 HELD-OUT TEST REPORT ({title}, c*={ev['c_star']:.6f}, "
        f"z_cut={ev['z_vlm_cut']:+.4f}, tau={ev['tau_vlm']:.5f}) ==="
    )
    print(
        f"Realised escalation: {ev['n_esc']}/{ev['n_test']} ({ev['esc_rate']:.3f}) | "
        f"VLM refusals: {ev['n_refused_all']} ({ev['vlm_refusal_rate']:.3%}) | "
        f"Bootstrap: {prim['bootstrap_unit']} ({prim['n_clusters']} clusters)"
    )
    print(
        f"Macro-F1 [95% CI]:  "
        f"probe={mf1['probe']['point']:.4f} {mf1['probe']['ci95']} | "
        f"vlm={mf1['vlm']['point']:.4f} {mf1['vlm']['ci95']} | "
        f"cascade={mf1['cascade']['point']:.4f} {mf1['cascade']['ci95']} | "
        f"random={mf1['random']['point']:.4f} {mf1['random']['ci95']}"
    )
    print(
        f"In-band acc (n={prim['in_band']['n']}): VLM={prim['in_band']['acc_a']:.3f} vs Probe={prim['in_band']['acc_b']:.3f} "
        f"(diff={prim['in_band']['diff']:+.3f} [{prim['in_band']['diff_ci95'][0]:+.3f}, {prim['in_band']['diff_ci95'][1]:+.3f}]) | "
        f"Out-of-band VLM={ev['resolved_vlm_acc']:.3f}, Probe={ev['resolved_probe_acc']:.3f}"
    )
    print("Pass / Falsification Criteria (HYPOTHESIS.md §5):")
    for key in (
        "1_budget_cap",
        "2_beats_probe",
        "3_non_inferior_to_vlm",
        "4_beats_random",
        "5_in_band_gain",
    ):
        item = crit[key]
        status = "PASS" if item["passed"] else "FAIL"
        print(f"  [{status}] {key}: {item}")
    verdict = "PASSED" if (gate_passed and crit["all_passed"]) else "FALSIFIED"
    print(f"Overall Protocol Verdict ({prim['bootstrap_unit']}): {verdict} (val_gate_passed={gate_passed})")
    if ev["iid_sensitivity"]["criteria"]["all_passed"] != crit["all_passed"]:
        print(
            f"  [NOTE] i.i.d. image bootstrap sensitivity differs: all_passed={ev['iid_sensitivity']['criteria']['all_passed']}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated eval-test")
    ap.add_argument("--gate", type=Path, required=True, help="locked Phase 3 validation gate JSON")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    ap.add_argument("--components", type=Path, default=Path("embeddings/components.json"))
    ap.add_argument("--probe", type=Path, required=True, help="Tier 0 test predictions JSONL")
    ap.add_argument("--vlm", type=Path, required=True, help="Tier 2 VLM test predictions JSONL")
    ap.add_argument("--out", type=Path, required=True, help="output path for Phase 4 test evaluation JSON")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-rand", type=int, default=DEFAULT_N_RAND)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    gate_path = a.gate.expanduser()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_sha256 = sha256_file(gate_path)
    op = gate["locked_operating_point"]
    if op.get("c_star") is None:
        raise SystemExit(
            f"ERROR: Validation gate {gate_path} did not lock a cutoff c_star (gate_passed={op.get('gate_passed')})."
        )

    comp_path = a.components if gate.get("components_sha256") is not None else None
    aligned = load_split_aligned(
        dataset=a.dataset,
        splits_path=a.splits,
        split_name="test",
        probe_path=a.probe,
        vlm_path=a.vlm,
        components_path=comp_path,
        allow_refusals=True,
    )

    if aligned["splits_sha256"] != gate["splits_sha256"]:
        raise SystemExit(
            f"ERROR: Splits hash mismatch vs locked gate.\n"
            f"  Gate splits_sha256: {gate['splits_sha256']}\n"
            f"  Test splits_sha256: {aligned['splits_sha256']}"
        )
    if gate.get("components_sha256") is not None and aligned["components_sha256"] != gate["components_sha256"]:
        raise SystemExit(
            f"ERROR: Components hash mismatch vs locked gate.\n"
            f"  Gate components_sha256: {gate['components_sha256']}\n"
            f"  Test components_sha256: {aligned['components_sha256']}"
        )

    # Verify model and probe identity match the locked Phase 3 gate (prevents cross-model or OOF mix-ups).
    verify_model_identity_match("vlm", gate.get("vlm_meta"), aligned["vlm_meta"])
    verify_model_identity_match("probe", gate.get("probe_meta"), aligned["probe_meta"])
    if gate.get("logit_source") is not None and aligned["logit_source"] != gate["logit_source"]:
        raise SystemExit(
            f"ERROR: VLM logit_source mismatch vs locked gate "
            f"(gate={gate['logit_source']!r} != test={aligned['logit_source']!r})."
        )

    c_star = float(op["c_star"])
    if "z_vlm_cut_star" in op and op["z_vlm_cut_star"] is not None:
        z_cut_star = float(op["z_vlm_cut_star"])
    else:
        tau_star = float(op["tau_vlm_star"])
        z_cut_star = float(logit(np.array([tau_star]))[0])

    common_kw: dict[str, Any] = {
        "y_true": aligned["y_true"],
        "probe_scores": aligned["p_probe"],
        "vlm_scores": aligned["p_vlm"],
        "vlm_logits": aligned["z_vlm"],
        "vlm_refused": aligned["vlm_refused"],
        "comp_ids": aligned["comp_ids"],
        "sources": aligned["sources"],
        "taxonomies": aligned["taxonomies"],
        "n_boot": a.n_boot,
        "n_rand": a.n_rand,
        "seed": a.seed,
    }

    locked_ev = evaluate_locked_test(c_star=c_star, z_vlm_cut=z_cut_star, **common_kw)
    uncal_ev = evaluate_locked_test(c_star=c_star, z_vlm_cut=0.0, **common_kw)

    gate_passed = bool(op.get("gate_passed", False))
    _print_test_report(f"locked:{op['tau_mode']}", locked_ev, gate_passed=gate_passed)
    if op["tau_mode"] != "uncalibrated":
        _print_test_report("uncalibrated-reference", uncal_ev, gate_passed=gate_passed)

    payload = {
        "gate_path": str(a.gate),
        "gate_sha256": gate_sha256,
        "splits": str(a.splits),
        "splits_sha256": aligned["splits_sha256"],
        "components": str(comp_path) if comp_path is not None else None,
        "components_sha256": aligned["components_sha256"],
        "probe_test_path": str(a.probe),
        "probe_test_sha256": aligned["probe_sha256"],
        "probe_meta": aligned["probe_meta"],
        "vlm_test_path": str(a.vlm),
        "vlm_test_sha256": aligned["vlm_sha256"],
        "vlm_meta": aligned["vlm_meta"],
        "logit_source": aligned["logit_source"],
        "n_boot": a.n_boot,
        "n_rand": a.n_rand,
        "seed": a.seed,
        "locked_from_val": op,
        "hypothesis_passed": bool(gate_passed and locked_ev["primary"]["criteria"]["all_passed"]),
        "locked_evaluation": locked_ev,
        "uncalibrated_evaluation": uncal_ev,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote Phase 4 Held-Out Evaluation -> {a.out}")


if __name__ == "__main__":
    main()
