from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch
from PIL import Image

from disrated.evalrun import build
from disrated.models.base import BaseModerator
from disrated.models.qwen3_vl import DEFAULT_QWEN3_VL_REVISION, Qwen3VLModerator


class _FakeTokenizer:
    _vocab: ClassVar[dict[str, int]] = {"Yes": 0, "yes": 1, "YES": 2, "No": 3, "no": 4, "NO": 5, "Sorry": 6}
    _inv: ClassVar[dict[int, str]] = {v: k for k, v in _vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._vocab[text]] if text in self._vocab else [99, 100]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        vals = [int(x) for x in ids]
        if vals == [6, 6]:
            return "Sorry, I cannot assist with that."
        return "".join(self._inv.get(i, "?") for i in vals)


class _FakeInputs(dict[str, Any]):
    def __init__(self, input_ids: torch.Tensor):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.image_processor = SimpleNamespace(size={})
        self.seen_images: list[Image.Image] = []

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return "<prompt>"

    def __call__(self, *, text, images, return_tensors="pt"):
        img = images[0]
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.info == {}
        assert not getattr(img, "filename", "")
        self.seen_images.append(img)
        return _FakeInputs(torch.tensor([[10, 11, 12]]))


class _FakeQwenModel:
    def __init__(self, logit_rows: list[list[float]]):
        self._rows = list(logit_rows)
        self._idx = 0

    def eval(self):
        return self

    def __call__(self, *, input_ids, use_cache: bool = False):
        row = self._rows[self._idx]
        self._idx += 1
        logits = torch.tensor([[row]], dtype=torch.float32)  # (1, 1, 7)
        return SimpleNamespace(logits=logits)

    def generate(self, *, input_ids, max_new_tokens: int = 32, do_sample: bool = False):
        return torch.tensor([[10, 11, 12, 6, 6]])


def test_qwen3_vl_logit_extraction_and_refusal_fallback(monkeypatch):
    import transformers

    fake_proc = _FakeProcessor()
    # Call 0 (img 1): strong Yes; Call 1 (img 2): conservative negative logit (-6.0); Call 2 (img 3): refusal ("Sorry")
    fake_model = _FakeQwenModel(
        [
            [8.0, -5.0, -5.0, 2.0, -5.0, -5.0, -5.0],
            [-6.0, -20.0, -20.0, 0.0, -20.0, -20.0, -20.0],
            [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0],
        ]
    )
    loaded_revs = []

    def _fake_proc_load(model_id, **kwargs):
        loaded_revs.append(("proc", model_id, kwargs.get("revision")))
        return fake_proc

    def _fake_model_load(model_id, **kwargs):
        loaded_revs.append(("model", model_id, kwargs.get("revision")))
        return fake_model

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(_fake_proc_load))
    monkeypatch.setattr(transformers.Qwen3VLForConditionalGeneration, "from_pretrained", staticmethod(_fake_model_load))

    mod = Qwen3VLModerator(warmup=False)
    assert isinstance(mod, BaseModerator)
    cfg = mod.config()
    assert cfg["revision"] == DEFAULT_QWEN3_VL_REVISION
    assert loaded_revs[0][2] == DEFAULT_QWEN3_VL_REVISION
    assert loaded_revs[1][2] == DEFAULT_QWEN3_VL_REVISION

    im1 = Image.new("RGB", (16, 16), (1, 2, 3))
    im1.info["prompt"] = "leak"
    im2 = Image.new("RGB", (16, 16), (4, 5, 6))
    im3 = Image.new("RGB", (16, 16), (7, 8, 9))

    outs = mod.predict([im1, im2, im3])
    assert len(outs) == 3

    # Image 1: Yes > No
    assert outs[0].refused is False
    assert outs[0].verdict is None
    assert outs[0].score is not None and outs[0].score > 0.99
    assert outs[0].extra["logit_diff"] > 5.5

    # Image 2: unclipped raw logit_diff preserved even for tiny p
    assert outs[1].refused is False
    assert outs[1].extra["logit_diff"] == pytest.approx(-6.0, abs=1e-3)
    assert outs[1].score == pytest.approx(1.0 / (1.0 + float(torch.exp(torch.tensor(6.0)))), abs=1e-4)

    # Image 3: low Yes/No mass triggers generation -> refusal detected
    assert outs[2].refused is True
    assert outs[2].score is None
    raw2 = json.loads(outs[2].raw_output or "{}")
    assert "Sorry" in raw2["generated"]


def test_evalrun_build_qwen3_vl(monkeypatch):
    import disrated.models.qwen3_vl as qmod

    sentinel = object()
    monkeypatch.setattr(qmod, "Qwen3VLModerator", lambda: sentinel)
    assert build("qwen3-vl-8b-fp8") is sentinel
