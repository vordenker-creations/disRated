from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

from .data.vhd11k import Sample, load_vhd11k
from .runner import load_clean
from .schema import Prediction

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
DEFAULT_LR = {"resnet50": 1e-4, "convnext_tiny": 1e-4, "vit_b_16": 3e-5, "clip_vit_b16": 1e-5}
CLIP_ID = "openai/clip-vit-base-patch16"


class ClipClassifier(nn.Module):
    """CLIP ViT-B/16 image encoder (fine-tuned) + one linear head on the pooled 768-d feature."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import CLIPVisionModel  # imported here so other archs don't need it

        self.vision = CLIPVisionModel.from_pretrained(CLIP_ID)
        self.head = nn.Linear(self.vision.config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vision(pixel_values=x).pooler_output)


def build(arch: str) -> nn.Module:
    """Pretrained backbone with a single-logit head (P(harmful) = sigmoid(logit))."""
    if arch == "clip_vit_b16":
        return ClipClassifier()
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        head = cast(nn.Linear, m.classifier[2])
        m.classifier[2] = nn.Linear(head.in_features, 1)
    elif arch == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        head = cast(nn.Linear, m.heads.head)
        m.heads.head = nn.Linear(head.in_features, 1)
    else:
        raise SystemExit(f"unknown --arch {arch!r}; choose from {sorted(DEFAULT_LR)}")
    return m


class Images(Dataset):
    def __init__(self, samples: list[Sample], tf) -> None:
        self.samples, self.tf = samples, tf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        # pixels only, as everywhere else in this repo (load_clean strips EXIF / PNG text / filename)
        return self.tf(load_clean(s.path)), int(s.harmful)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, dev: str):
    model.eval()
    ys, ps, ms = [], [], []
    for x, y in loader:
        x = x.to(dev, non_blocking=True)
        sync(dev)
        t0 = time.perf_counter()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logit = model(x).squeeze(1)
        p = torch.sigmoid(logit.float())
        sync(dev)
        dt = (time.perf_counter() - t0) * 1000 / len(y)  # model time only
        ys.append(y.numpy())
        ps.append(p.cpu().numpy())
        ms.append(np.full(len(y), dt))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ms)


def write(path: Path, samples: list[Sample], scores, ms, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s, sc, t in zip(samples, scores, ms, strict=True):
            p = Prediction(
                sample_id=s.sample_id,
                score=min(max(float(sc), 0.0), 1.0),
                latency_ms=float(t),
                extra=dict(extra),
            )
            f.write(p.model_dump_json() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.finetune")
    ap.add_argument("--arch", required=True, choices=sorted(DEFAULT_LR))
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.v2.json"))
    ap.add_argument("--name", required=True, help="output stem, e.g. rn50-s0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without val-AUC gain")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--head-lr", type=float, default=1e-3, help="clip_vit_b16 only: lr of the new linear head")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="subsample each split to N images (smoke test only)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lr = a.lr or DEFAULT_LR[a.arch]
    set_seed(a.seed)

    samples, _ = load_vhd11k(a.dataset.expanduser())
    by_id = {s.sample_id: s for s in samples}
    splits_raw = a.splits.read_text(encoding="utf-8")
    splits_sha256 = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    splits = json.loads(splits_raw)

    def part(name: str) -> list[Sample]:
        sel = [by_id[i] for i in splits[name]]
        if a.limit:
            sel = random.Random(0).sample(sel, min(a.limit, len(sel)))
        return sel

    tr, va, te = part("train"), part("val"), part("test")
    print(f"device={dev} arch={a.arch} lr={lr} | train={len(tr)} val={len(va)} test={len(te)}")

    mean, std = (CLIP_MEAN, CLIP_STD) if a.arch == "clip_vit_b16" else (MEAN, STD)
    tf_train = T.Compose(
        [
            T.RandomResizedCrop(224, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])

    g = torch.Generator().manual_seed(a.seed)
    kw: dict[str, Any] = {
        "num_workers": a.workers,
        "pin_memory": dev == "cuda",
        "persistent_workers": a.workers > 0,
    }
    dl_tr = DataLoader(Images(tr, tf_train), batch_size=a.batch_size, shuffle=True, drop_last=True, generator=g, **kw)
    dl_va = DataLoader(Images(va, tf_eval), batch_size=64, shuffle=False, **kw)
    dl_te = DataLoader(Images(te, tf_eval), batch_size=64, shuffle=False, **kw)

    model = build(a.arch).to(dev)
    if isinstance(model, ClipClassifier):
        groups = [
            {"params": model.vision.parameters(), "lr": lr},
            {"params": model.head.parameters(), "lr": a.head_lr},
        ]
    else:
        groups = model.parameters()
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    bce = nn.BCEWithLogitsLoss()

    best_auc, best_epoch, best_state, stale = -1.0, -1, None, 0
    for epoch in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True).float()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logit = model(x).squeeze(1)
            loss = bce(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()

        yv, pv, _ = predict(model, dl_va, dev)
        v_auc = float(roc_auc_score(yv, pv)) if len(set(yv.tolist())) == 2 else float("nan")
        v_acc = float(((pv >= 0.5) == (yv == 1)).mean())
        print(
            f"epoch {epoch + 1:>2}/{a.epochs} | loss {tot / n:.4f} | val AUC {v_auc:.4f} acc {v_acc:.4f} | {time.time() - t0:.0f}s"
        )
        if v_auc > best_auc:
            best_auc, best_epoch, stale = v_auc, epoch + 1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= a.patience:
                print(f"early stop: no val-AUC gain for {a.patience} epochs")
                break

    # Model selection used val only; test is touched once, here.
    if best_state is None:
        raise RuntimeError("Training completed without recording a valid checkpoint.")
    model.load_state_dict(best_state)
    meta = {"splits_sha256": splits_sha256, "arch": a.arch, "epoch": best_epoch, "lr": lr, "seed": a.seed}
    out = Path("results")
    yv, pv, mv = predict(model, dl_va, dev)
    yt, pt, mt = predict(model, dl_te, dev)
    write(out / f"{a.name}-val.jsonl", va, pv, mv, meta)
    write(out / f"{a.name}-test.jsonl", te, pt, mt, meta)
    torch.save(best_state, out / f"{a.name}.pt")

    t_auc = float(roc_auc_score(yt, pt)) if len(set(yt.tolist())) == 2 else float("nan")
    t_acc = float(((pt >= 0.5) == (yt == 1)).mean())
    print(f"best epoch {best_epoch} (val AUC {best_auc:.4f}) | TEST AUC {t_auc:.4f} acc {t_acc:.4f}")
    print(
        f"score with: uv run python -m disrated.score results/{a.name}-test.jsonl "
        f"--dataset {a.dataset} --splits {a.splits} --split test"
    )


if __name__ == "__main__":
    main()
