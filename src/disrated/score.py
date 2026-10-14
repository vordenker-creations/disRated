from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data.vhd11k import load_vhd11k
from .metrics.scoring import evaluate, latency_stats, read_predictions, slice_by

COLS = ["n", "coverage", "accuracy", "precision", "recall", "specificity", "f1", "macro_f1",
        "average_precision", "roc_auc", "prevalence"]

SHORT = {"coverage": "cov", "accuracy": "acc", "precision": "prec", "specificity": "spec",
         "macro_f1": "macroF1", "average_precision": "AP", "roc_auc": "AUC", "prevalence": "prev"}


def row(label: str, m: dict) -> str:
    return f"{label:<26}" + "".join(f"{fmt(m.get(c)):>9}" for c in COLS)


def fmt(x) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.score")
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True, help="path to the active splits JSON file")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--concepts", action="store_true", help="also print the per-concept table")
    ap.add_argument("--json", type=Path, help="write the full report here")
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    if a.split != "all":
        ids = set(json.loads(a.splits.read_text())[a.split])
        samples = [s for s in samples if s.sample_id in ids]
    preds = read_predictions(a.predictions)

    splits_raw = a.splits.read_text(encoding="utf-8")
    expected_hash = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()

    # Verify provenance against prediction metadata
    logged_hash = None
    for p in preds.values():
        if "splits_sha256" in p.extra:
            logged_hash = p.extra["splits_sha256"]
            break

    if logged_hash is not None and logged_hash != expected_hash:
        raise SystemExit(
            f"ERROR: Splits hash mismatch.\n"
            f"  Prediction metadata: {logged_hash}\n"
            f"  Supplied splits:     {expected_hash}\n"
            f"Pass the matching splits file to score this run."
        )
    overall = evaluate(samples, preds, a.threshold)
    by_source = slice_by(samples, preds, lambda s: s.source, a.threshold)
    by_concept = slice_by(samples, preds, lambda s: s.concept, a.threshold)

    print(f"{a.predictions}  split={a.split}  threshold={a.threshold}")
    print(f"{'':<26}" + "".join(f"{SHORT.get(c, c):>9}" for c in COLS))
    print(row("overall", overall))
    for k, m in by_source.items():
        print(row(k, m))
    print(f"unscored: missing={overall['missing']} error={overall['error']} "
          f"refused={overall['refused']} unusable={overall['unusable']}")
    print("latency:", {k: round(v, 2) for k, v in latency_stats(samples, preds).items()})
    if a.concepts:
        print()
        for k, m in by_concept.items():
            print(row(k, m))
    if a.json:
        a.json.write_text(json.dumps(dict(overall=overall, by_source=by_source,
                                          by_concept=by_concept,
                                          latency=latency_stats(samples, preds)), indent=1))


if __name__ == "__main__":
    main()