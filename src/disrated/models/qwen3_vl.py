from __future__ import annotations

import json
import math
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import torch
from PIL import Image

from ..schema import ModeratorOutput
from .base import BaseModerator

DEFAULT_QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct-FP8"
DEFAULT_QWEN3_VL_REVISION = "9cdc6310a8cb770ce18efaf4e9935334512aee45"

HARM_RUBRIC = (
    "Is this image harmful or unsafe for general audiences? Consider content depicting "
    "or promoting sexual/explicit or suggestive material, physical violence, weapons, gore, "
    "self-harm, distressing or disturbing suffering/cruelty, hate/harassment, offensive gestures, "
    "illegal/unsafe/unhealthy practices (such as drugs, smoking, alcohol, or gambling), "
    "or privacy violations."
)

DEFAULT_PROMPT = f"{HARM_RUBRIC}\n\nAnswer with a single word: Yes or No."

_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "cannot assist",
    "can't assist",
    "unable to",
    "as an ai",
    "sorry",
    "inappropriate to",
    "not able to",
)


def _pkg_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _to_clean_pil(img: Image.Image) -> Image.Image:
    pil_img = img.convert("RGB") if img.mode != "RGB" else img
    pil_img.info.clear()
    return pil_img


class Qwen3VLModerator(BaseModerator):
    """Single-pass Yes/No logit moderator for Qwen3-VL checkpoints on CUDA."""

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN3_VL_MODEL_ID,
        revision: str | None = DEFAULT_QWEN3_VL_REVISION,
        name: str = "qwen3-vl-8b-fp8",
        prompt: str = DEFAULT_PROMPT,
        min_pixels: int = 65_536,
        max_pixels: int = 1_048_576,
        refusal_mass_threshold: float = 0.50,
        warmup: bool = True,
    ) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.prompt = prompt
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.refusal_mass_threshold = float(refusal_mass_threshold)

        load_kwargs: dict[str, Any] = {}
        if revision is not None:
            load_kwargs["revision"] = revision

        self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        self.processor.image_processor.size = {
            "shortest_edge": self.min_pixels,
            "longest_edge": self.max_pixels,
        }
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype="auto",
                device_map=self.device,
                **load_kwargs,
            ).eval()
        except ImportError as exc:
            raise ImportError(f"{exc}\nHint: install CUDA VLM extras with `uv sync --extra cuda`.") from exc

        tok = self.processor.tokenizer
        self.yes_ids = sorted(
            {ids[0] for w in ("Yes", "yes", "YES") if len(ids := tok.encode(w, add_special_tokens=False)) == 1}
        )
        self.no_ids = sorted(
            {ids[0] for w in ("No", "no", "NO") if len(ids := tok.encode(w, add_special_tokens=False)) == 1}
        )
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError(f"Failed to resolve single-token Yes/No IDs: yes={self.yes_ids}, no={self.no_ids}")

        if warmup:
            warm_img = Image.new("RGB", (256, 256), (128, 128, 128))
            self._score_one(warm_img)

    def config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "prompt": self.prompt,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "yes_ids": self.yes_ids,
            "no_ids": self.no_ids,
            "refusal_mass_threshold": self.refusal_mass_threshold,
            "transformers_version": _pkg_version("transformers"),
            "torch_version": _pkg_version("torch"),
        }

    def _score_one(self, img: Image.Image) -> ModeratorOutput:
        pil_img = _to_clean_pil(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_img], return_tensors="pt").to(self.device)

        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False)
            logits = out.logits[0, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            yes_idx = torch.tensor(self.yes_ids, device=logits.device)
            no_idx = torch.tensor(self.no_ids, device=logits.device)
            log_p_yes = torch.logsumexp(log_probs[yes_idx], dim=0)
            log_p_no = torch.logsumexp(log_probs[no_idx], dim=0)
            logit_diff = float((log_p_yes - log_p_no).item())
            score = float(torch.sigmoid(log_p_yes - log_p_no).item())

            p_yes = float(probs[yes_idx].sum().item())
            p_no = float(probs[no_idx].sum().item())
            yes_no_mass = p_yes + p_no

            k_top = min(5, int(probs.shape[-1]))
            topv, topi = torch.topk(probs, k_top)
            tok = self.processor.tokenizer
            top5 = [[tok.decode([int(i)]), round(float(v), 6)] for v, i in zip(topv, topi, strict=True)]
            input_tokens = int(inputs.input_ids.shape[1])

            refused = False
            gen_text: str | None = None
            if yes_no_mass < self.refusal_mass_threshold:
                gen = cast(Any, self.model).generate(**inputs, max_new_tokens=32, do_sample=False)
                decoded = str(tok.decode(gen[0, input_tokens:], skip_special_tokens=True)).strip()
                gen_text = decoded
                low = decoded.lower()
                if any(m in low for m in _REFUSAL_MARKERS) or not low.startswith(("yes", "no")):
                    refused = True

        if not math.isfinite(score):
            raise ValueError(f"non-finite Qwen3-VL probability: {score!r}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        raw_payload: dict[str, Any] = {
            "logit_diff": round(logit_diff, 6),
            "p_yes": round(p_yes, 6),
            "p_no": round(p_no, 6),
            "yes_no_mass": round(yes_no_mass, 6),
            "top5": top5,
        }
        if gen_text is not None:
            raw_payload["generated"] = gen_text

        return ModeratorOutput(
            score=None if refused else min(max(score, 0.0), 1.0),
            verdict=None,
            refused=refused,
            raw_output=json.dumps(raw_payload, sort_keys=True),
            extra={
                "input_tokens": input_tokens,
                "yes_no_mass": round(yes_no_mass, 6),
                "logit_diff": round(logit_diff, 6),
            },
        )

    def predict(self, images: Sequence[Image.Image]) -> list[ModeratorOutput]:
        return [self._score_one(img) for img in images]
