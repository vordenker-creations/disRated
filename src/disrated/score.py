from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .data.vhd11k import load_vhd11k
from .metrics.scoring import evaluate, latency_stats, read_predictions, slice_by

COLS = [
    "n",
    "coverage",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "macro_f1",
    "average_precision",
    "roc_auc",
    "prevalence",
]

SHORT = {
    "coverage": "cov",
    "accuracy": "acc",
    "precision": "prec",
    "specificity": "spec",
    "macro_f1": "macroF1",
    "average_precision": "AP",
    "roc_auc": "AUC",
    "prevalence": "prev",
}


def row(label: str, m: dict[str, Any]) -> str:
    return f"{label:<26}" + "".join(f"{fmt(m.get(c)):>9}" for c in COLS)


def fmt(x) -> str:
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="disrated score")
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True, help="path to the active splits JSON file")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--concepts", action="store_true", help="also print the per-concept table")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument(
        "--allow-legacy",
        action="store_true",
        help="allow scoring legacy prediction files that omit splits_sha256 metadata",
    )
    a = ap.parse_args(argv)

    splits_raw = a.splits.expanduser().read_text(encoding="utf-8")
    expected_hash = hashlib.sha256(splits_raw.encode("utf-8")).hexdigest()
    splits_data = json.loads(splits_raw)

    samples, _ = load_vhd11k(a.dataset.expanduser())
    if a.split != "all":
        if a.split not in splits_data:
            raise SystemExit(f"ERROR: split '{a.split}' not in {a.splits}")
        expected_ids = set(splits_data[a.split])
        samples = [s for s in samples if s.sample_id in expected_ids]
    else:
        expected_ids = set().union(*splits_data.values())
        samples = [s for s in samples if s.sample_id in expected_ids]

    missing_from_ds = sorted(expected_ids - {s.sample_id for s in samples})
    if missing_from_ds:
        raise SystemExit(
            f"ERROR: {len(missing_from_ds)} sample(s) in split '{a.split}' failed to load "
            f"from {a.dataset} (first: {missing_from_ds[0]})."
        )

    preds = read_predictions(a.predictions)

    # Verify provenance across all prediction records (reject mismatched or missing hashes)
    all_hashes = {p.extra["splits_sha256"] for p in preds.values() if "splits_sha256" in p.extra}
    if all_hashes and all_hashes != {expected_hash}:
        raise SystemExit(
            f"ERROR: Splits hash mismatch.\n"
            f"  Prediction metadata: {sorted(all_hashes)}\n"
            f"  Supplied splits:     {expected_hash}\n"
            f"Pass the matching splits file to score this run."
        )
    relevant_preds = [preds[s.sample_id] for s in samples if s.sample_id in preds]
    missing_hash = [p.sample_id for p in relevant_preds if "splits_sha256" not in p.extra]
    if missing_hash and not a.allow_legacy:
        raise SystemExit(
            f"ERROR: {len(missing_hash)} prediction record(s) are missing splits_sha256 metadata "
            f"(first: {missing_hash[0]}). Re-run on the active split or pass --allow-legacy."
        )
    overall = evaluate(samples, preds, a.threshold)
    by_source = slice_by(samples, preds, lambda s: s.source, a.threshold)
    by_concept = slice_by(samples, preds, lambda s: s.concept, a.threshold)

    print(f"{a.predictions}  split={a.split}  threshold={a.threshold}")
    print(f"{'':<26}" + "".join(f"{SHORT.get(c, c):>9}" for c in COLS))
    print(row("overall", overall))
    for k, m in by_source.items():
        print(row(k, m))
    print(
        f"unscored: missing={overall['missing']} error={overall['error']} "
        f"refused={overall['refused']} unusable={overall['unusable']}"
    )
    print("latency:", {k: round(v, 2) for k, v in latency_stats(samples, preds).items()})
    if a.concepts:
        print()
        for k, m in by_concept.items():
            print(row(k, m))
    if a.json:
        a.json.write_text(
            json.dumps(
                {
                    "overall": overall,
                    "by_source": by_source,
                    "by_concept": by_concept,
                    "latency": latency_stats(samples, preds),
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
