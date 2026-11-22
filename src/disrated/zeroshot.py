"""Zero-shot CLIP baseline (RQ1): score cached image embeddings against text prompts, no training.

Reads an `embeddings/*.npz` written by `disrated embed` (unit-normalised image features in the
joint image-text space, plus the model id and revision), encodes a few prompt sets with the
matching text encoder, and scores each image as

    p(harmful) = sigmoid(scale * (cos(img, harmful_class) - cos(img, safe_class)))

where each class vector is the normalised mean of its prompt embeddings. The prompt set is chosen
on `val` by ROC-AUC; only the chosen set is then written for the requested output splits, so
`test` never influences the choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .data.vhd11k import load_vhd11k
from .probe import write

# Fixed before looking at any results; selection among them happens on `val` only.
PROMPT_SETS: dict[str, dict[str, list[str]]] = {
    "generic": {
        "harmful": ["a harmful image", "an unsafe image", "an inappropriate image"],
        "safe": ["a harmless image", "a safe image", "an appropriate image"],
    },
    "rubric": {
        "harmful": [
            "an image depicting sexual or explicit content",
            "an image depicting violence, weapons or gore",
            "an image depicting self-harm or suicide",
            "an image depicting disturbing suffering or cruelty",
            "an image depicting hate, harassment or an offensive gesture",
            "an image depicting drugs, smoking, alcohol or gambling",
        ],
        "safe": [
            "an ordinary, harmless everyday image",
            "a safe image suitable for all audiences",
        ],
    },
    "photo": {
        "harmful": [
            "a photo of something harmful",
            "a photo of something offensive",
            "a photo of something disturbing",
            "a photo of something dangerous",
        ],
        "safe": [
            "a photo of something harmless",
            "a photo of something ordinary",
            "a photo of something pleasant",
            "a photo of something safe",
        ],
    },
}


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm feature vector")
    return x / n


def class_vector(text_feats: np.ndarray) -> np.ndarray:
    """Normalised mean of normalised prompt embeddings (standard CLIP prompt ensembling)."""
    return _unit(_unit(np.asarray(text_feats, dtype=np.float64)).mean(axis=0))


def zeroshot_scores(img_feats: np.ndarray, harmful: np.ndarray, safe: np.ndarray, scale: float) -> np.ndarray:
    """P(harmful) for each row of `img_feats` from its cosine margin between the two class vectors."""
    img = _unit(np.asarray(img_feats, dtype=np.float64))
    margin = img @ harmful - img @ safe
    return 1.0 / (1.0 + np.exp(-scale * margin))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) == 2 else float("nan")


def select_prompt_set(val_aucs: dict[str, float]) -> str:
    """Highest val AUC; ties go to the earliest set in PROMPT_SETS order."""
    order = [k for k in PROMPT_SETS if k in val_aucs] + [k for k in val_aucs if k not in PROMPT_SETS]
    usable = [k for k in order if np.isfinite(val_aucs[k])]
    if not usable:
        raise ValueError("no prompt set has a finite val AUC")
    return max(usable, key=lambda k: (val_aucs[k], -usable.index(k)))


def encode_texts(model_id: str, revision: str | None, prompts: Sequence[str]) -> tuple[np.ndarray, float]:
    """Text embeddings for `prompts` from the same checkpoint that produced the image cache."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    kw = {"revision": revision} if revision else {}
    model = AutoModel.from_pretrained(model_id, **kw).eval()
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    padding: Any = "max_length" if "siglip" in model_id.lower() else True
    batch = tok(list(prompts), padding=padding, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        out = model.get_text_features(**batch)
        feats = out if torch.is_tensor(out) else out.pooler_output
        scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 100.0
    return feats.float().cpu().numpy(), scale


def load_embeddings(
    path: Path, model_override: str | None = None, revision_override: str | None = None
) -> tuple[dict[str, int], np.ndarray, np.ndarray, str, str | None, str]:
    """Returns (index, feats, ms, model_id, revision, provenance).

    Caches written by `disrated embed` >= 0.1.2 carry model_id/revision. Older caches do not; for those
    the caller must name the model with `--model`, and provenance is recorded as "declared" rather than
    "cache" so the result file shows the text encoder was matched by hand.
    """
    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["ids"]]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"ERROR: duplicate ids in {path}")
    cached_model = str(z["model_id"]) if "model_id" in z.files else None
    cached_rev = str(z["revision"]) if "revision" in z.files and str(z["revision"]) else None
    if cached_model is not None:
        if model_override is not None and model_override != cached_model:
            raise SystemExit(f"ERROR: --model {model_override!r} does not match the cache's model_id {cached_model!r}")
        return {sid: i for i, sid in enumerate(ids)}, z["feats"], z["ms"], cached_model, cached_rev, "cache"
    if model_override is None:
        raise SystemExit(
            f"ERROR: {path} has no model_id (written by an older `disrated embed`). "
            "Pass --model with the checkpoint that produced it, e.g. --model openai/clip-vit-base-patch16."
        )
    return {sid: i for i, sid in enumerate(ids)}, z["feats"], z["ms"], model_override, revision_override, "declared"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated zeroshot")
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings/openai_clip-vit-base-patch16.npz"))
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    ap.add_argument("--name", default="clip-b16-zeroshot", help="output stem: results/<name>-<split>.jsonl")
    ap.add_argument("--model", default=None, help="checkpoint of an older cache that lacks model_id")
    ap.add_argument(
        "--revision",
        default=None,
        help="HuggingFace commit for --model (defaults to the pinned SHA in `disrated embed` for known models)",
    )
    ap.add_argument(
        "--write-splits",
        default="val,test",
        help="comma-separated splits to write after choosing the prompt set on val (val, test, all)",
    )
    a = ap.parse_args(argv)

    splits_raw = a.splits.expanduser().read_text(encoding="utf-8")
    splits_sha256 = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    splits = json.loads(splits_raw)

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}
    revision_override = a.revision
    if a.model is not None and revision_override is None:
        from .embed import PINNED_REVISIONS

        revision_override = PINNED_REVISIONS.get(a.model)
    index, feats, ms, model_id, revision, provenance = load_embeddings(
        a.embeddings.expanduser(), a.model, revision_override
    )

    def rows(split: str) -> list[str]:
        ids = sorted(by_id) if split == "all" else list(splits[split])
        missing = [i for i in ids if i not in index or i not in by_id]
        if missing:
            raise SystemExit(
                f"ERROR: {len(missing)} {split} sample(s) missing from embeddings/dataset (first: {missing[0]})"
            )
        return ids

    # 1) Score every prompt set on val only and pick one.
    val_ids = rows("val")
    y_val = np.array([by_id[i].harmful for i in val_ids], dtype=int)
    x_val = feats[[index[i] for i in val_ids]]
    all_prompts = [p for ps in PROMPT_SETS.values() for p in ps["harmful"] + ps["safe"]]
    text, scale = encode_texts(model_id, revision, all_prompts)  # load the text encoder once
    class_vecs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    val_aucs: dict[str, float] = {}
    start = 0
    for set_name, ps in PROMPT_SETS.items():
        n_h, n_s = len(ps["harmful"]), len(ps["safe"])
        harm, safe = text[start : start + n_h], text[start + n_h : start + n_h + n_s]
        start += n_h + n_s
        class_vecs[set_name] = (class_vector(harm), class_vector(safe))
        val_aucs[set_name] = auc(y_val, zeroshot_scores(x_val, *class_vecs[set_name], scale))
    chosen = select_prompt_set(val_aucs)
    print(f"model={model_id} revision={revision or 'unpinned'} ({provenance}) scale={scale:.2f}")
    print("val AUC by prompt set:", {k: round(v, 4) for k, v in val_aucs.items()}, "-> chosen:", chosen)

    # 2) Write only the chosen set for the requested splits.
    extra = {"splits_sha256": splits_sha256, "prompt_set": chosen}
    for split in [s.strip() for s in a.write_splits.split(",") if s.strip()]:
        ids = rows(split)
        sel = [index[i] for i in ids]
        scores = zeroshot_scores(feats[sel], *class_vecs[chosen], scale)
        out = Path("results") / f"{a.name}-{split}.jsonl"
        write(
            out,
            [SimpleNamespace(sample_id=i) for i in ids],
            scores,
            ms[sel],  # image-encoder time from the cache; the two dot products add ~nothing
            extra=extra,
            meta_sidecar={
                "moderator": a.name,
                "config": {
                    "model_id": model_id,
                    "model_provenance": provenance,
                    "revision": revision,
                    "embeddings": str(a.embeddings),
                    "prompt_sets": PROMPT_SETS,
                    "chosen_prompt_set": chosen,
                    "val_auc_by_prompt_set": val_aucs,
                    "logit_scale": scale,
                },
                "run": {"split": split, "splits_sha256": splits_sha256},
            },
        )
        print(f"wrote {out} ({len(ids)} rows)")
    print(
        f"score with: uv run disrated score results/{a.name}-val.jsonl --dataset {a.dataset} --splits {a.splits} --split val"
    )


if __name__ == "__main__":
    main()
