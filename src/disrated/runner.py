from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from PIL import Image
from pydantic import ValidationError

from .data.vhd11k import Sample
from .models.base import BaseModerator
from .schema import Prediction


def load_clean(path: Path) -> Image.Image:
    """Decode to RGB and keep nothing but pixels. PNG text chunks and EXIF can carry
    prompts or labels, and PIL's .filename would leak the SDXL label tag."""
    with Image.open(path) as im:
        img = im.convert("RGB")
    img.info.clear()
    return img


def _meta_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".meta.json")


def _check_meta(out: Path, moderator: BaseModerator) -> None:
    meta = json.loads(json.dumps({"moderator": moderator.name, "config": moderator.config()}, sort_keys=True))
    mp = _meta_path(out)
    if mp.exists():
        old = json.loads(mp.read_text(encoding="utf-8"))
        if old != meta:
            raise RuntimeError(
                f"{out} was produced by a different moderator/config. Use a new output path.\n"
                f"  existing: {old}\n  requested: {meta}"
            )
    else:
        mp.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")


def _prepare_resume(out: Path) -> set[str]:
    """Keep complete, non-errored records; drop truncated lines and errors so they get re-run."""
    keep: list[Prediction] = []
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                p = Prediction.model_validate_json(line)
            except ValidationError:
                continue  # half-written line from a crash
            if p.error is None:
                keep.append(p)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text("".join(p.model_dump_json() + "\n" for p in keep), encoding="utf-8")
        tmp.replace(out)
    return {p.sample_id for p in keep}


def run(
    moderator: BaseModerator,
    samples: Sequence[Sample],
    out: Path,
    batch_size: int = 16,
    limit: int | None = None,
) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _check_meta(out, moderator)
    done = _prepare_resume(out)
    todo = [s for s in samples[:limit] if s.sample_id not in done]

    n_err = 0
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            preds: list[Prediction] = []
            imgs, ok = [], []
            for s in todo[i : i + batch_size]:
                try:
                    imgs.append(load_clean(s.path))  # moderator gets pixels only, never the Sample
                    ok.append(s)
                except Exception as e:
                    preds.append(Prediction(sample_id=s.sample_id, latency_ms=0.0, error=f"load: {e!r}"))

            if imgs:
                t0 = time.perf_counter()
                err = None
                try:
                    outs = moderator.predict(imgs)
                    if len(outs) != len(imgs):
                        raise ValueError(f"got {len(outs)} outputs for {len(imgs)} images")
                except Exception as e:
                    outs, err = [None] * len(imgs), f"predict: {e!r}"
                ms = (time.perf_counter() - t0) * 1000 / len(imgs)
                for s, o in zip(ok, outs):
                    if o is None:
                        preds.append(Prediction(sample_id=s.sample_id, latency_ms=ms, error=err))
                    else:
                        preds.append(Prediction(sample_id=s.sample_id, latency_ms=ms, **o.model_dump()))

            for p in preds:
                f.write(p.model_dump_json() + "\n")
            f.flush()
            n_err += sum(p.error is not None for p in preds)

    return {"requested": len(samples[:limit]), "already_done": len(done), "ran": len(todo), "errors": n_err}