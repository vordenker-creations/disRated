import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from disrated.evalrun import main as evalrun_main
from disrated.metrics.scoring import read_predictions
from disrated.score import main as score_main


def _create_mock_dataset(tmp_path: Path, n: int = 10) -> tuple[Path, Path, Path]:
    ds_dir = tmp_path / "dataset"
    img_dir = ds_dir / "all_10000_evenHarmfulUnharmful"
    img_dir.mkdir(parents=True)

    rows = ["imagePath,decision,harmfulType"]
    splits = {"train": [], "val": [], "test": []}
    for i in range(n):
        name = f"gun_000_{i:05d}.png"
        img_path = img_dir / name
        Image.new("RGB", (10, 10), (i * 20 % 256, 0, 0)).save(img_path)
        decision = "yes" if i % 2 == 0 else "no"
        rows.append(f"{name},{decision},violence")
        if i < 6:
            splits["train"].append(name)
        elif i < 8:
            splits["val"].append(name)
        else:
            splits["test"].append(name)

    ann_csv = ds_dir / "harmful_image_10000_ann.csv"
    ann_csv.write_text("\n".join(rows), encoding="utf-8")

    splits_file = tmp_path / "splits.v2.json"
    splits_file.write_text(json.dumps(splits), encoding="utf-8")

    # Also create a mismatched splits file
    mismatched_splits = dict(splits)
    mismatched_splits["test"] = list(reversed(splits["test"]))
    mismatched_file = tmp_path / "mismatched_splits.json"
    mismatched_file.write_text(json.dumps(mismatched_splits) + " ", encoding="utf-8")

    return ds_dir, splits_file, mismatched_file


def test_evalrun_with_splits_flag_and_provenance(tmp_path):
    ds_dir, splits_file, mismatched_file = _create_mock_dataset(tmp_path, n=10)
    out_file = tmp_path / "preds.jsonl"

    # 1. Verify running evalrun with --splits and --split test does not crash
    evalrun_main(
        [
            "--model",
            "constant-0.8",
            "--dataset",
            str(ds_dir),
            "--splits",
            str(splits_file),
            "--split",
            "test",
            "--output",
            str(out_file),
        ]
    )

    # 2. Verify output exists and contains splits_sha256 in extra
    preds = read_predictions(out_file)
    assert len(preds) == 2  # exactly 2 test items in our mock split

    expected_hash = hashlib.sha256(splits_file.read_bytes()).hexdigest()
    for p in preds.values():
        assert "splits_sha256" in p.extra
        assert p.extra["splits_sha256"] == expected_hash

    # 3. Verify score.py succeeds when supplied the matching splits file
    score_main(
        [
            str(out_file),
            "--dataset",
            str(ds_dir),
            "--splits",
            str(splits_file),
            "--split",
            "test",
        ]
    )

    # 4. Verify score.py raises SystemExit with error when supplied a mismatched splits file
    with pytest.raises(SystemExit) as exc_info:
        score_main(
            [
                str(out_file),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(mismatched_file),
                "--split",
                "test",
            ]
        )
    assert "Splits hash mismatch" in str(exc_info.value)

    # 5. Verify score.py also catches a mixed/corrupted hash on a later record (not just record 1)
    lines = out_file.read_text(encoding="utf-8").splitlines()
    rec1 = json.loads(lines[1])
    rec1["extra"]["splits_sha256"] = "0" * 64
    mixed_file = tmp_path / "mixed_preds.jsonl"
    mixed_file.write_text(lines[0] + "\n" + json.dumps(rec1) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        score_main(
            [
                str(mixed_file),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--split",
                "test",
            ]
        )
    assert "Splits hash mismatch" in str(exc_info.value)

    # 6. Verify score.py rejects legacy predictions missing splits_sha256 unless --allow-legacy is set
    legacy_file = tmp_path / "legacy_preds.jsonl"
    legacy_lines = []
    for ln in lines:
        rec = json.loads(ln)
        rec["extra"].pop("splits_sha256", None)
        legacy_lines.append(json.dumps(rec))
    legacy_file.write_text("\n".join(legacy_lines) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        score_main(
            [
                str(legacy_file),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--split",
                "test",
            ]
        )
    assert "missing splits_sha256" in str(exc_info.value)

    score_main(
        [
            str(legacy_file),
            "--dataset",
            str(ds_dir),
            "--splits",
            str(splits_file),
            "--split",
            "test",
            "--allow-legacy",
        ]
    )

    # 7. Verify runner/evalrun refuses reusing an output path on a different split or splits file
    with pytest.raises(RuntimeError):
        evalrun_main(
            [
                "--model",
                "constant-0.8",
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--split",
                "val",
                "--output",
                str(out_file),
            ]
        )

    # 8. Verify missing dataset image cannot silently shrink split denominator in evalrun or score
    missing_img = ds_dir / "all_10000_evenHarmfulUnharmful" / "gun_000_00008.png"
    missing_img.unlink()
    with pytest.raises(SystemExit) as exc_info:
        score_main(
            [
                str(out_file),
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--split",
                "test",
            ]
        )
    assert "failed to load" in str(exc_info.value)

    with pytest.raises(SystemExit) as exc_info:
        evalrun_main(
            [
                "--model",
                "constant-0.8",
                "--dataset",
                str(ds_dir),
                "--splits",
                str(splits_file),
                "--split",
                "test",
                "--output",
                str(tmp_path / "fresh.jsonl"),
            ]
        )
    assert "could not be loaded" in str(exc_info.value)
