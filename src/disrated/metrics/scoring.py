from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

from pydantic import ValidationError
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.vhd11k import Sample
from ..schema import Prediction

def read_predictions(path: Path) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            p = Prediction.model_validate_json(line)
        except ValidationError:
            continue  # truncated line from a crash
        preds[p.sample_id] = p  # later lines win
    return preds


def hard_call(p: Prediction, threshold: float = 0.5) -> bool | None:
    """The model's own verdict wins; otherwise threshold the score."""
    if p.verdict is not None:
        return p.verdict
    if p.score is not None:
        return p.score >= threshold
    return None


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict:
    """'Positive' = harmful. precision/recall/specificity are None when undefined;
    F1 follows sklearn's zero_division=0 convention."""
    pairs = list(zip(y_true, y_pred))
    tp = sum(1 for t, p in pairs if t and p)
    fp = sum(1 for t, p in pairs if not t and p)
    tn = sum(1 for t, p in pairs if not t and not p)
    fn = sum(1 for t, p in pairs if t and not p)
    n = len(pairs)

    def div(a, b):
        return a / b if b else None

    def f1(a, b, c):  # a = true positives of the class, b/c = the two error counts
        d = 2 * a + b + c
        return 2 * a / d if d else 0.0

    if n == 0:
        return dict(tp=0, fp=0, tn=0, fn=0, accuracy=None, precision=None, recall=None,
                    specificity=None, f1=None, macro_f1=None)
    f1_pos, f1_neg = f1(tp, fp, fn), f1(tn, fn, fp)
    return dict(
        tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=div(tp + tn, n),
        precision=div(tp, tp + fp),
        recall=div(tp, tp + fn),
        specificity=div(tn, tn + fp),
        f1=f1_pos,
        macro_f1=(f1_pos + f1_neg) / 2,  # mean of F1(harmful) and F1(unharmful)
    )


def evaluate(samples: Sequence[Sample], preds: dict[str, Prediction], threshold: float = 0.5) -> dict:
    """Metrics over `samples`. Anything without a usable call (missing, error, refusal)
    is excluded from the metrics but counted, and shows up in `coverage`."""
    y_true, y_pred, ys, sc = [], [], [], []
    c = dict(missing=0, error=0, refused=0, unusable=0)
    for s in samples:
        p = preds.get(s.sample_id)
        if p is None:
            c["missing"] += 1
        elif p.error is not None:
            c["error"] += 1
        elif p.refused:
            c["refused"] += 1
        elif (call := hard_call(p, threshold)) is None:
            c["unusable"] += 1
        else:
            y_true.append(s.harmful)
            y_pred.append(call)
            if p.score is not None:
                ys.append(s.harmful)
                sc.append(p.score)

    n = len(samples)
    out = {"n": n, "scored": len(y_true), "coverage": len(y_true) / n if n else None, **c}
    out.update(binary_metrics(y_true, y_pred))
    ok = len(set(ys)) == 2
    out["average_precision"] = average_precision_score(ys, sc) if ok else None
    out["roc_auc"] = roc_auc_score(ys, sc) if ok else None
    out["prevalence"] = sum(y_true) / len(y_true) if y_true else None
    return out


def slice_by(samples: Iterable[Sample], preds: dict[str, Prediction],
             key: Callable[[Sample], str | None], threshold: float = 0.5) -> dict[str, dict]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        k = key(s)
        if k is not None:
            groups[k].append(s)
    return {k: evaluate(v, preds, threshold) for k, v in sorted(groups.items())}


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, math.ceil(q / 100 * len(xs)) - 1))]


def latency_stats(samples: Sequence[Sample], preds: dict[str, Prediction]) -> dict:
    xs = [preds[s.sample_id].latency_ms for s in samples
          if s.sample_id in preds and preds[s.sample_id].error is None]
    if not xs:
        return {}
    return dict(mean_ms=sum(xs) / len(xs), p50_ms=_pct(xs, 50), p95_ms=_pct(xs, 95),
                p99_ms=_pct(xs, 99), images_per_s=1000 / (sum(xs) / len(xs)))