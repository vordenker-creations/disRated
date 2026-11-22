from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from disrated.eval_test import evaluate_locked_test
from disrated.eval_test import main as eval_test_main
from disrated.schema import Prediction
from disrated.val_gate import main as val_gate_main


def _build_synthetic_workspace(tmp_path: Path, n_val: int = 120, n_test: int = 120):
    ds_dir = tmp_path / "dataset"
    img_dir = ds_dir / "all_10000_evenHarmfulUnharmful"
    img_dir.mkdir(parents=True)

    rows = ["imagePath,decision,harmfulType"]
    splits = {"train": [], "val": [], "test": []}
    comp_map: dict[str, str] = {}

    def _add_split(split_name: str, count: int, offset: int):
        for i in range(count):
            idx = offset + i
            concept = "gun" if i % 2 == 0 else "alcohol"
            sid = f"{concept}_000_{idx:05d}.png"
            Image.new("RGB", (8, 8), (i % 256, 0, 0)).save(img_dir / sid)
            decision = "yes" if i % 2 == 0 else "no"
            rows.append(f"{sid},{decision},{concept}")
            splits[split_name].append(sid)
            comp_map[sid] = f"comp_{split_name}_{i // 2}"

    _add_split("val", n_val, 0)
    _add_split("test", n_test, n_val)
    (ds_dir / "harmful_image_10000_ann.csv").write_text("\n".join(rows), encoding="utf-8")

    splits_file = tmp_path / "splits.v2.json"
    splits_raw = json.dumps(splits, indent=1)
    splits_file.write_text(splits_raw, encoding="utf-8")
    splits_sha = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()

    comp_file = tmp_path / "components.json"
    comp_file.write_text(json.dumps(comp_map, indent=1), encoding="utf-8")

    def _write_preds(
        path: Path,
        sids: list[str],
        scores: np.ndarray,
        logits: np.ndarray | None = None,
        c_val: float | None = None,
        sidecar: dict[str, Any] | None = None,
    ):
        lines = []
        for k, sid in enumerate(sids):
            extra: dict[str, Any] = {"splits_sha256": splits_sha}
            if c_val is not None:
                extra["c"] = c_val
            if logits is not None:
                extra["logit_diff"] = float(logits[k])
            lines.append(
                Prediction(
                    sample_id=sid,
                    score=float(scores[k]),
                    latency_ms=1.0,
                    extra=extra,
                ).model_dump_json()
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if sidecar is not None:
            mp = path.with_suffix(path.suffix + ".meta.json")
            mp.write_text(json.dumps(sidecar, indent=1, sort_keys=True), encoding="utf-8")

    for split_name, n in (("val", n_val), ("test", n_test)):
        y = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
        p_probe = np.where(y, 0.90, 0.10)
        # First 25% near boundary where probe is wrong
        k25 = n // 4
        p_probe[:k25] = np.where(y[:k25], 0.48, 0.52)

        # Conservative VLM (uncalibrated fails at 0.0, Platt recovers near-100%)
        z_vlm = np.where(y, -2.0, -10.0)
        # Make VLM miss 2 outer points so cascade slightly beats VLM too
        z_vlm[k25 : k25 + 2] = np.where(y[k25 : k25 + 2], -10.0, -2.0)
        p_vlm = 1.0 / (1.0 + np.exp(-z_vlm))

        _write_preds(tmp_path / f"probe-{split_name}.jsonl", splits[split_name], p_probe, c_val=1.0)
        vlm_sidecar = {
            "moderator": "qwen3-vl-8b-fp8",
            "config": {"model_id": "Qwen/Qwen3-VL-8B-Instruct-FP8", "prompt": "v2"},
            "run": {"split": split_name, "splits_sha256": splits_sha},
        }
        _write_preds(
            tmp_path / f"vlm-{split_name}.jsonl",
            splits[split_name],
            p_vlm,
            logits=z_vlm,
            sidecar=vlm_sidecar,
        )

    return ds_dir, splits_file, comp_file


def test_val_gate_to_eval_test_end_to_end(tmp_path: Path):
    ds_dir, splits_file, comp_file = _build_synthetic_workspace(tmp_path)
    gate_path = tmp_path / "gate.json"
    test_out = tmp_path / "phase4_test.json"

    val_gate_main(
        [
            "--dataset",
            str(ds_dir),
            "--splits",
            str(splits_file),
            "--components",
            str(comp_file),
            "--probe",
            str(tmp_path / "probe-val.jsonl"),
            "--vlm",
            str(tmp_path / "vlm-val.jsonl"),
            "--out",
            str(gate_path),
            "--n-boot",
            "400",
            "--n-rand",
            "40",
        ]
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["locked_operating_point"]["tau_mode"] == "platt"
    assert gate["locked_operating_point"]["gate_passed"] is True
    assert gate["probe_val_sha256"]
    assert gate["vlm_val_sha256"]
    assert gate["vlm_meta"]["moderator"] == "qwen3-vl-8b-fp8"
    assert gate["probe_meta"]["c"] == 1.0

    eval_test_main(
        [
            "--gate",
            str(gate_path),
            "--dataset",
            str(ds_dir),
            "--splits",
            str(splits_file),
            "--components",
            str(comp_file),
            "--probe",
            str(tmp_path / "probe-test.jsonl"),
            "--vlm",
            str(tmp_path / "vlm-test.jsonl"),
            "--out",
            str(test_out),
            "--n-boot",
            "400",
            "--n-rand",
            "40",
        ]
    )
    res = json.loads(test_out.read_text(encoding="utf-8"))
    assert res["hypothesis_passed"] is True
    assert res["locked_evaluation"]["primary"]["criteria"]["all_passed"] is True
    assert res["locked_evaluation"]["by_taxonomy"]["concrete_harm"]["n"] == 60
    assert res["locked_evaluation"]["by_taxonomy"]["contextual_normative"]["n"] == 60
    # Uncalibrated and original-rule (k=0.05) evaluations + oracle-gap metric are logged transparently
    assert res["uncalibrated_evaluation"]["z_vlm_cut"] == 0.0
    assert res["original_rule_evaluation"] is not None
    assert res["original_rule_evaluation"]["z_vlm_cut"] == 0.0
    assert 0.0 < res["locked_evaluation"]["oracle_gap_captured_acc"] <= 1.0

    # Verify tampering with gate splits_sha256 is rejected
    bad_gate = tmp_path / "bad_gate.json"
    bad_gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    bad_gate_payload["splits_sha256"] = "0" * 64
    bad_gate.write_text(json.dumps(bad_gate_payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="Splits hash mismatch"):
        eval_test_main(
            [
                "--gate",
                str(bad_gate),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--components",
                str(comp_file),
                "--probe",
                str(tmp_path / "probe-test.jsonl"),
                "--vlm",
                str(tmp_path / "vlm-test.jsonl"),
                "--out",
                str(tmp_path / "fail.json"),
                "--n-boot",
                "100",
                "--n-rand",
                "20",
            ]
        )

    # Verify wrong VLM model (e.g., random baseline sharing the same splits_sha256) is rejected
    wrong_vlm = tmp_path / "wrong-vlm-test.jsonl"
    wrong_vlm.write_text((tmp_path / "vlm-test.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    wrong_vlm.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"moderator": "random", "config": {"seed": 0}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="VLM moderator mismatch"):
        eval_test_main(
            [
                "--gate",
                str(gate_path),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--components",
                str(comp_file),
                "--probe",
                str(tmp_path / "probe-test.jsonl"),
                "--vlm",
                str(wrong_vlm),
                "--out",
                str(tmp_path / "fail2.json"),
                "--n-boot",
                "100",
                "--n-rand",
                "20",
            ]
        )

    # Verify OOF probe file (containing 'fold' metadata) is rejected
    oof_probe = tmp_path / "oof-probe-test.jsonl"
    oof_lines = []
    for ln in (tmp_path / "probe-test.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        rec["extra"]["fold"] = 0
        oof_lines.append(json.dumps(rec))
    oof_probe.write_text("\n".join(oof_lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="out-of-fold"):
        eval_test_main(
            [
                "--gate",
                str(gate_path),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--components",
                str(comp_file),
                "--probe",
                str(oof_probe),
                "--vlm",
                str(tmp_path / "vlm-test.jsonl"),
                "--out",
                str(tmp_path / "fail3.json"),
                "--n-boot",
                "100",
                "--n-rand",
                "20",
            ]
        )

    # Verify wrong probe hyperparameter C is rejected
    wrong_c_probe = tmp_path / "wrong-c-probe-test.jsonl"
    wrong_c_lines = []
    for ln in (tmp_path / "probe-test.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        rec["extra"]["c"] = 10.0
        wrong_c_lines.append(json.dumps(rec))
    wrong_c_probe.write_text("\n".join(wrong_c_lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Probe regularization C mismatch"):
        eval_test_main(
            [
                "--gate",
                str(gate_path),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--components",
                str(comp_file),
                "--probe",
                str(wrong_c_probe),
                "--vlm",
                str(tmp_path / "vlm-test.jsonl"),
                "--out",
                str(tmp_path / "fail4.json"),
                "--n-boot",
                "100",
                "--n-rand",
                "20",
            ]
        )


def test_evaluate_locked_test_handles_budget_cap_and_refusals():
    n = 100
    y = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    p_probe = np.where(y, 0.55, 0.45)  # all near boundary -> 100% escalated -> violates 0.30 budget cap
    p_vlm = np.where(y, 0.90, 0.10)
    refused = np.zeros(n, dtype=bool)
    refused[:5] = True

    ev = evaluate_locked_test(
        y_true=y,
        probe_scores=p_probe,
        vlm_scores=p_vlm,
        c_star=1.0,
        z_vlm_cut=0.0,
        vlm_refused=refused,
        n_boot=200,
        n_rand=20,
    )
    assert ev["n_refused_all"] == 5
    assert ev["vlm_refusal_rate"] == 0.05
    assert ev["metrics"]["vlm"]["accuracy"] == 0.95  # 5 refused items counted as errors, NOT probe fallback
    assert ev["primary"]["criteria"]["1_budget_cap"]["passed"] is False
    assert ev["primary"]["criteria"]["all_passed"] is False


def test_total_vlm_refusal_scores_zero_not_probe_credit():
    n = 20
    y = np.array([i % 2 == 0 for i in range(n)], dtype=bool)
    p_probe = np.where(y, 0.90, 0.10)  # probe is 100% accurate
    p_probe[:4] = np.where(y[:4], 0.52, 0.48)  # 20% in-band, still 100% right
    p_vlm = np.full(n, np.nan)
    refused = np.ones(n, dtype=bool)  # VLM refuses all 20 images

    ev = evaluate_locked_test(
        y_true=y,
        probe_scores=p_probe,
        vlm_scores=p_vlm,
        c_star=0.20,
        z_vlm_cut=0.0,
        vlm_refused=refused,
        n_boot=200,
        n_rand=20,
    )
    # Standalone VLM and in-band VLM must get 0.0 (never probe's 1.0!)
    assert ev["metrics"]["vlm"]["macro_f1"] == 0.0
    assert ev["primary"]["arms"]["macro_f1"]["vlm"]["point"] == 0.0
    assert ev["primary"]["in_band"]["acc_a"] == 0.0
    assert ev["primary"]["criteria"]["5_in_band_gain"]["passed"] is False
    assert ev["primary"]["criteria"]["2_beats_probe"]["passed"] is False
    assert ev["primary"]["criteria"]["all_passed"] is False
