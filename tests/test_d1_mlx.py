from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from disrated.evalrun import build
from disrated.evalrun import main as evalrun_main
from disrated.metrics.scoring import read_predictions
from disrated.models.base import BaseModerator
from disrated.models.d1_mlx import HARM_QUESTIONS, LiquidD1MLXModerator
from disrated.score import main as score_main


class _FakeD1Model:
    def __init__(self, probs: list[float]):
        self._probs = list(probs)
        self.calls: list[dict[str, Any]] = []

    def system_one(self, *, state, questions, media):
        assert state is None
        assert len(media) == 1
        img = media[0]
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.info == {}
        assert not getattr(img, "filename", "")
        self.calls.append({"questions": questions, "size": img.size})
        p = self._probs[len(self.calls) - 1]
        return {
            "answers": {"harmful": {"type": "noul", "noul": p}},
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }


def _install_fake_mlx_vlm(monkeypatch, probs: list[float]) -> _FakeD1Model:
    fake_model = _FakeD1Model(probs)
    fake_mod = SimpleNamespace(load=lambda model_id: (fake_model, object()))
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_mod)
    return fake_model


def test_d1_mlx_contract_and_numpy_or_pil_ingest(monkeypatch):
    fake_model = _install_fake_mlx_vlm(monkeypatch, [0.12, 1.0000002])
    mod = LiquidD1MLXModerator()
    assert isinstance(mod, BaseModerator)
    assert mod.name == "d1-2b-mlx"
    cfg = mod.config()
    assert cfg["model_id"] == "LiquidAI/d1-2B-MLX-fp16"
    assert cfg["revision"] == "f4e6e118bf850880e6f45a5b22085057778ac243"
    assert cfg["questions"] == HARM_QUESTIONS

    pil_in = Image.new("RGB", (8, 8), (10, 20, 30))
    pil_in.info["prompt"] = "leak"
    arr_in = np.zeros((8, 8, 3), dtype=np.uint8)

    outs = mod.predict([pil_in, arr_in])
    assert len(outs) == 2
    assert outs[0].score == 0.12
    assert outs[1].score == 1.0
    assert all(o.verdict is None and o.refused is False for o in outs)
    assert outs[0].extra["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert len(fake_model.calls) == 2


def test_evalrun_build_and_score_d1_mlx(monkeypatch, tmp_path: Path):
    _install_fake_mlx_vlm(monkeypatch, [0.85, 0.25])

    ds_dir = tmp_path / "dataset"
    img_dir = ds_dir / "all_10000_evenHarmfulUnharmful"
    img_dir.mkdir(parents=True)
    rows = ["imagePath,decision,harmfulType"]
    val_ids = []
    for i, decision in enumerate(["yes", "no"]):
        name = f"gun_000_{i:05d}.png"
        Image.new("RGB", (10, 10), (i * 50, 0, 0)).save(img_dir / name)
        rows.append(f"{name},{decision},violence")
        val_ids.append(name)
    (ds_dir / "harmful_image_10000_ann.csv").write_text("\n".join(rows), encoding="utf-8")

    splits_file = tmp_path / "splits.v2.json"
    splits_file.write_text(json.dumps({"train": [], "val": val_ids, "test": []}), encoding="utf-8")

    out_file = tmp_path / "d1-val.jsonl"
    evalrun_main(
        [
            "--model",
            "d1-2b-mlx",
            "--dataset",
            str(ds_dir),
            "--splits",
            str(splits_file),
            "--split",
            "val",
            "--batch-size",
            "1",
            "--output",
            str(out_file),
        ]
    )

    preds = read_predictions(out_file)
    assert list(preds.keys()) == val_ids
    assert preds[val_ids[0]].score == 0.85
    assert preds[val_ids[1]].score == 0.25

    report_file = tmp_path / "report.json"
    score_main(
        [
            str(out_file),
            "--dataset",
            str(ds_dir),
            "--splits",
            str(splits_file),
            "--split",
            "val",
            "--json",
            str(report_file),
        ]
    )
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["overall"]["accuracy"] == 1.0
    assert report["overall"]["refused"] == 0


def test_build_returns_d1_mlx(monkeypatch):
    _install_fake_mlx_vlm(monkeypatch, [0.5])
    mod = build("d1-2b-mlx")
    assert isinstance(mod, LiquidD1MLXModerator)
