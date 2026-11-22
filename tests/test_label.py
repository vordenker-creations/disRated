from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from disrated.label import (
    LabelRecord,
    _render_clean_jpeg_bytes,
    build_split_manifest,
    cohens_kappa,
    score_human_audit,
)
from disrated.label import (
    main as label_main,
)
from disrated.schema import Prediction


def test_cohens_kappa_known_values():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    assert cohens_kappa([True, True, False, False], [False, False, True, True]) == pytest.approx(-1.0)


def test_render_clean_jpeg_strips_png_metadata(tmp_path: Path):
    p = tmp_path / "sdxl_extend_by_llama3_harmful_00001_seed2024.png"
    meta = PngInfo()
    meta.add_text("prompt", "secret_label_leak")
    Image.new("RGB", (12, 12), (40, 80, 120)).save(p, pnginfo=meta)

    raw = _render_clean_jpeg_bytes(p)
    assert b"secret_label_leak" not in raw
    with Image.open(io.BytesIO(raw)) as im:
        assert im.format == "JPEG"
        assert im.size == (12, 12)


def test_label_prepare_blocks_sealed_test_and_scores_blind_audit(tmp_path: Path):
    ds_dir = tmp_path / "dataset"
    img_dir = ds_dir / "all_10000_evenHarmfulUnharmful"
    img_dir.mkdir(parents=True)

    rows = ["imagePath,decision,harmfulType"]
    val_ids: list[str] = []
    for i in range(12):
        concept = "gun" if i % 2 == 0 else "alcohol"
        sid = f"{concept}_000_{i:05d}.png"
        Image.new("RGB", (8, 8), (i * 10, 0, 0)).save(img_dir / sid)
        rows.append(f"{sid},{'yes' if i % 2 == 0 else 'no'},{concept}")
        val_ids.append(sid)
    (ds_dir / "harmful_image_10000_ann.csv").write_text("\n".join(rows), encoding="utf-8")

    splits = {"train": [], "val": val_ids, "test": ["dummy_test.png"]}
    splits_file = tmp_path / "splits.v2.json"
    splits_raw = json.dumps(splits)
    splits_file.write_text(splits_raw, encoding="utf-8")
    splits_sha = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()

    # Refuses split=test unless --allow-test is passed
    with pytest.raises(SystemExit, match="sealed"):
        label_main(
            [
                "prepare",
                "--mode",
                "split",
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--split",
                "test",
                "--out",
                str(tmp_path / "bad.json"),
            ]
        )

    # Build a val split manifest with 4 hidden sentinel repeats
    manifest = build_split_manifest(splits_file, "val", n_sentinels=4, seed=0)
    assert manifest["counts"]["unique_images"] == 12
    assert manifest["counts"]["sentinels"] == 4
    assert manifest["counts"]["total_slots"] == 16
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Write synthetic probe predictions (matches judge on all 12)
    probe_path = tmp_path / "probe-val.jsonl"
    probe_lines = [
        Prediction(
            sample_id=sid,
            score=0.9 if i % 2 == 0 else 0.1,
            latency_ms=1.0,
            extra={"splits_sha256": splits_sha},
        ).model_dump_json()
        for i, sid in enumerate(val_ids)
    ]
    probe_path.write_text("\n".join(probe_lines) + "\n", encoding="utf-8")

    # Simulate two raters agreeing on 10/12 items (disagreeing with judge on item 0 and 1 -> consensus 10 items)
    r1_path = tmp_path / "r1.jsonl"
    r2_path = tmp_path / "r2.jsonl"
    r1_lines, r2_lines = [], []
    for it in manifest["items"]:
        sid = it["sample_id"]
        idx = val_ids.index(sid)
        judge = idx % 2 == 0
        h1 = (not judge) if idx in (0, 1) else judge
        h2 = (not judge) if idx in (0, 1) else judge
        if idx == 2 and not it["is_sentinel"]:
            h2 = not h1  # single inter-rater disagreement
        r1_lines.append(
            LabelRecord(
                slot=it["slot"],
                sample_id=sid,
                is_sentinel=it["is_sentinel"],
                harmful=h1,
                borderline=(idx < 4),
                rater="vor",
            ).model_dump_json()
        )
        r2_lines.append(
            LabelRecord(
                slot=it["slot"],
                sample_id=sid,
                is_sentinel=it["is_sentinel"],
                harmful=h2,
                borderline=(idx < 4),
                rater="van",
            ).model_dump_json()
        )
    r1_path.write_text("\n".join(r1_lines) + "\n", encoding="utf-8")
    r2_path.write_text("\n".join(r2_lines) + "\n", encoding="utf-8")

    report = score_human_audit(ds_dir, manifest_path, [r1_path, r2_path], probe_path=probe_path)
    assert report["n_primary_items"] == 12
    assert report["n_consensus_items"] == 11
    assert report["per_rater"]["vor"]["intra_rater_sentinels"]["n"] == 4
    assert report["per_rater"]["vor"]["intra_rater_sentinels"]["kappa"] == pytest.approx(1.0)
    assert report["inter_rater"]["vor_vs_van"]["n_overlap"] == 12
    assert report["inter_rater"]["vor_vs_van"]["agreement"] == pytest.approx(11 / 12)
    assert report["model_vs_human_and_judge"]["vhd11k_judge"]["acc_vs_human"] == pytest.approx(9 / 11)
    assert report["model_vs_human_and_judge"]["probe"]["acc_vs_human"] == pytest.approx(9 / 11)
