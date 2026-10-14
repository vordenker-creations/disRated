from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from disrated.data.vhd11k import Sample
from disrated.models.dummy import Constant, Random
from disrated.runner import load_clean, run
from disrated.schema import Prediction


def _samples(tmp_path: Path, n: int = 25) -> list[Sample]:
    out = []
    for i in range(n):
        p = tmp_path / f"gun_000_{i:05d}.png"
        Image.new("RGB", (8, 8), (i * 10 % 256, 0, 0)).save(p)
        out.append(Sample(sample_id=p.name, path=p, harmful=i % 2 == 0, harm_type="",
                          source="concept", concept="gun", group=p.name))
    return out


def _read(path: Path) -> list[Prediction]:
    return [Prediction.model_validate_json(l) for l in path.read_text().splitlines()]


def test_run_and_resume(tmp_path):
    s = _samples(tmp_path)
    out = tmp_path / "r" / "x.jsonl"
    assert run(Random(seed=1), s, out, batch_size=4, limit=10)["ran"] == 10
    r = run(Random(seed=1), s, out, batch_size=4)
    assert r["ran"] == 15 and r["already_done"] == 10
    preds = _read(out)
    assert len(preds) == len({p.sample_id for p in preds}) == 25
    assert all(p.error is None for p in preds)


def test_resume_survives_truncated_line(tmp_path):
    s = _samples(tmp_path, 10)
    out = tmp_path / "x.jsonl"
    run(Constant(0.0), s, out)
    out.write_text(out.read_text() + '{"sample_id": "tru')  # crash mid-write
    assert run(Constant(0.0), s, out)["ran"] == 0
    assert len(_read(out)) == 10


def test_config_mismatch_refuses(tmp_path):
    s = _samples(tmp_path, 4)
    out = tmp_path / "x.jsonl"
    run(Random(seed=1), s, out)
    with pytest.raises(RuntimeError):
        run(Random(seed=2), s, out)


def test_errors_recorded_then_retried(tmp_path):
    s = _samples(tmp_path, 6)
    s[2].path.unlink()
    out = tmp_path / "x.jsonl"
    assert run(Constant(1.0), s, out, batch_size=3)["errors"] == 1
    Image.new("RGB", (8, 8)).save(s[2].path)
    r = run(Constant(1.0), s, out, batch_size=3)
    assert r["ran"] == 1 and r["errors"] == 0
    assert len(_read(out)) == 6


def test_moderator_sees_pixels_only(tmp_path):
    p = tmp_path / "sdxl_extend_by_llama3_harmful_00001_seed2024.png"
    meta = PngInfo()
    meta.add_text("prompt", "secret")
    Image.new("RGB", (8, 8)).save(p, pnginfo=meta)
    img = load_clean(p)
    assert img.info == {}
    assert not getattr(img, "filename", "")