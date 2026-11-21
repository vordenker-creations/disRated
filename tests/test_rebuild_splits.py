from __future__ import annotations

import numpy as np
import pytest

from disrated.rebuild_splits import align_cache_by_ids, find_duplicate_edges


def test_align_cache_by_ids_makes_duplicate_edges_invariant_to_cache_shuffling():
    all_ids = [f"img_{i:02d}.png" for i in range(6)]
    # Make img_01 and img_04 cosine duplicates, and img_02 and img_05 pHash duplicates
    feats = np.eye(6, dtype=np.float32)
    feats[4] = feats[1]  # cosine = 1.0 between img_01 and img_04

    phash = np.array(
        [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00],
            [0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA],
            [0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0],
            [0x33, 0xCC, 0x33, 0xCC, 0x33, 0xCC, 0x33, 0xCC],
            [0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAB],  # Hamming dist 1 from img_02
        ],
        dtype=np.uint8,
    )

    ref_edges = {tuple(sorted(e)) for e in find_duplicate_edges(all_ids, feats, phash)}
    assert ref_edges == {("img_01.png", "img_04.png"), ("img_02.png", "img_05.png")}

    # Shuffle embedding and pHash caches independently
    perm_e = [3, 5, 0, 4, 1, 2]
    perm_h = [5, 2, 1, 0, 4, 3]
    aligned_feats = align_cache_by_ids(all_ids, [all_ids[i] for i in perm_e], feats[perm_e], "emb")
    aligned_phash = align_cache_by_ids(all_ids, [all_ids[i] for i in perm_h], phash[perm_h], "hash")

    shuffled_edges = {tuple(sorted(e)) for e in find_duplicate_edges(all_ids, aligned_feats, aligned_phash)}
    assert shuffled_edges == ref_edges


def test_align_cache_by_ids_rejects_missing_and_duplicate_ids():
    all_ids = ["a.png", "b.png"]
    arr = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="missing"):
        align_cache_by_ids(all_ids, ["a.png", "c.png"], arr, "emb")

    with pytest.raises(ValueError, match="duplicate"):
        align_cache_by_ids(all_ids, ["a.png", "a.png"], arr, "emb")
