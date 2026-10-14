import random
from pathlib import Path

import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from disrated.data.vhd11k import Sample
from disrated.metrics.scoring import binary_metrics, evaluate
from disrated.schema import Prediction


def test_binary_metrics_match_sklearn():
    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(50, 300)
        y = [rng.random() < 0.5 for _ in range(n)]
        p = [rng.random() < 0.5 for _ in range(n)]
        m = binary_metrics(y, p)
        assert m["accuracy"] == pytest.approx(accuracy_score(y, p))
        assert m["precision"] == pytest.approx(precision_score(y, p, zero_division=0))
        assert m["recall"] == pytest.approx(recall_score(y, p, zero_division=0))
        assert m["specificity"] == pytest.approx(recall_score(y, p, pos_label=False, zero_division=0))
        assert m["f1"] == pytest.approx(f1_score(y, p, zero_division=0))
        assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro", zero_division=0))


def test_degenerate_classifiers_have_known_answers():
    y = [True, False] * 10
    safe = binary_metrics(y, [False] * 20)
    assert safe["precision"] is None and safe["recall"] == 0 and safe["specificity"] == 1
    assert safe["macro_f1"] == pytest.approx(1 / 3)
    harm = binary_metrics(y, [True] * 20)
    assert harm["recall"] == 1 and harm["precision"] == 0.5
    assert harm["macro_f1"] == pytest.approx(1 / 3)


def _s(i: int, harmful: bool) -> Sample:
    name = f"gun_000_{i:05d}.png"
    return Sample(sample_id=name, path=Path(name), harmful=harmful, harm_type="",
                  source="concept", concept="gun", group=name)


def test_evaluate_accounts_for_unscored_and_keeps_alignment():
    samples = [_s(i, i % 2 == 0) for i in range(10)]
    preds = {s.sample_id: Prediction(sample_id=s.sample_id, latency_ms=1.0,
                                     score=0.9 if s.harmful else 0.1) for s in samples[:6]}
    preds[samples[6].sample_id] = Prediction(sample_id=samples[6].sample_id, latency_ms=1.0, error="boom")
    preds[samples[7].sample_id] = Prediction(sample_id=samples[7].sample_id, latency_ms=1.0, refused=True)
    m = evaluate(samples, preds)  # samples 8 and 9 have no prediction at all
    assert (m["scored"], m["error"], m["refused"], m["missing"]) == (6, 1, 1, 2)
    assert m["coverage"] == pytest.approx(0.6)
    assert m["accuracy"] == 1.0 and m["average_precision"] == pytest.approx(1.0)