from __future__ import annotations

import numpy as np
import pytest

from disrated.zeroshot import PROMPT_SETS, class_vector, select_prompt_set, zeroshot_scores


def test_class_vector_is_unit_mean_of_unit_prompts():
    v = class_vector(np.array([[2.0, 0.0], [0.0, 5.0]]))
    np.testing.assert_allclose(v, np.array([1.0, 1.0]) / np.sqrt(2.0))
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_scores_follow_cosine_margin_and_are_scale_invariant_to_image_norm():
    harmful, safe = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    imgs = np.array([[1.0, 0.0], [0.0, 3.0], [1.0, 1.0]])
    s = zeroshot_scores(imgs, harmful, safe, scale=100.0)
    assert s[0] > 0.99  # aligned with the harmful class
    assert s[1] < 0.01  # aligned with the safe class (norm 3 must not matter)
    assert np.isclose(s[2], 0.5)  # equidistant -> undecided
    assert np.all((s >= 0) & (s <= 1))


def test_select_prompt_set_uses_val_auc_and_breaks_ties_by_declared_order():
    names = list(PROMPT_SETS)
    assert select_prompt_set({names[0]: 0.70, names[1]: 0.81, names[2]: 0.75}) == names[1]
    assert select_prompt_set({names[0]: 0.80, names[1]: 0.80, names[2]: 0.60}) == names[0]
    assert select_prompt_set({names[0]: float("nan"), names[1]: 0.55}) == names[1]
    with pytest.raises(ValueError):
        select_prompt_set({names[0]: float("nan")})


def test_main_end_to_end_with_fake_encoder(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    import disrated.zeroshot as zs

    rng = np.random.default_rng(0)
    ids = [f"img_{i:02d}.png" for i in range(12)]
    harmful = [i % 2 == 0 for i in range(12)]
    # harmful images lean towards axis 0, safe towards axis 1
    feats = np.array([[1.0, 0.2] if h else [0.2, 1.0] for h in harmful]) + rng.normal(0, 0.05, (12, 2))
    np.savez(
        tmp_path / "emb.npz",
        ids=np.array(ids),
        feats=feats,
        ms=np.full(12, 3.0, dtype=np.float32),
        model_id=np.array("fake/clip"),
        revision=np.array("abc123"),
    )
    splits = {"train": ids[:6], "val": ids[6:10], "test": ids[10:]}
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    sha = hashlib.sha256((tmp_path / "splits.json").read_text(encoding="utf-8").encode("utf-8")).hexdigest()

    samples = [SimpleNamespace(sample_id=i, harmful=h) for i, h in zip(ids, harmful, strict=True)]
    monkeypatch.setattr(zs, "load_vhd11k", lambda root: (samples, []))

    def fake_encode(model_id, revision, prompts):
        assert (model_id, revision) == ("fake/clip", "abc123")
        harm_words = {p for ps in zs.PROMPT_SETS.values() for p in ps["harmful"]}
        return np.array([[1.0, 0.0] if p in harm_words else [0.0, 1.0] for p in prompts]), 100.0

    monkeypatch.setattr(zs, "encode_texts", fake_encode)
    monkeypatch.chdir(tmp_path)
    zs.main(["--embeddings", "emb.npz", "--dataset", ".", "--splits", "splits.json", "--name", "zs"])

    for split, n in (("val", 4), ("test", 2)):
        rows = [json.loads(line) for line in (tmp_path / "results" / f"zs-{split}.jsonl").read_text().splitlines()]
        assert [r["sample_id"] for r in rows] == splits[split]
        assert len(rows) == n
        assert all(r["extra"]["splits_sha256"] == sha for r in rows)
        for r in rows:
            assert (r["score"] > 0.5) == dict(zip(ids, harmful, strict=True))[r["sample_id"]]
        meta = json.loads((tmp_path / "results" / f"zs-{split}.jsonl.meta.json").read_text())
        assert meta["config"]["chosen_prompt_set"] in zs.PROMPT_SETS
        assert meta["run"] == {"split": split, "splits_sha256": sha}


def test_prompt_sets_are_well_formed():
    for ps in PROMPT_SETS.values():
        assert ps["harmful"] and ps["safe"]
        assert not set(ps["harmful"]) & set(ps["safe"])
