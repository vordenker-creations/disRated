from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data.vhd11k import load_vhd11k


def get_connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, str]:
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    node_to_group = {}

    for node in nodes:
        if node not in visited:
            comp = []
            q = [node]
            visited.add(node)
            for curr in q:
                comp.append(curr)
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            comp_id = f"comp_{min(comp)}"
            for member in comp:
                node_to_group[member] = comp_id

    return node_to_group


def align_cache_by_ids(
    all_ids: list[str],
    cache_ids: list[str],
    array: np.ndarray,
    cache_name: str,
) -> np.ndarray:
    """Align a cached feature/hash array strictly to `all_ids` order, rejecting missing/duplicate IDs."""
    if len(cache_ids) != len(array):
        raise ValueError(f"{cache_name}: ids length ({len(cache_ids)}) != array rows ({len(array)})")
    id_to_idx: dict[str, int] = {}
    for idx, sid in enumerate(cache_ids):
        if sid in id_to_idx:
            raise ValueError(f"{cache_name}: duplicate sample_id {sid!r}")
        id_to_idx[sid] = idx
    missing = [sid for sid in all_ids if sid not in id_to_idx]
    if missing:
        raise ValueError(f"{cache_name}: missing {len(missing)} sample_id(s) (first: {missing[0]!r})")
    return array[[id_to_idx[sid] for sid in all_ids]]


def find_duplicate_edges(
    all_ids: list[str],
    feats: np.ndarray,
    phash: np.ndarray,
    cosine_threshold: float = 0.95,
    phash_threshold: int = 8,
    chunk_size: int = 1000,
) -> list[tuple[str, str]]:
    """Find undirected near-duplicate pairs (cosine >= threshold OR pHash Hamming <= threshold)."""
    lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    edges: list[tuple[str, str]] = []
    n = len(all_ids)
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        cos_chunk = np.matmul(feats[i:end_i], feats.T)
        for row_offset in range(end_i - i):
            global_row = i + row_offset
            cos_chunk[row_offset, : global_row + 1] = -np.inf

        r_cos, c_cos = np.where(cos_chunk >= cosine_threshold)
        for r, c in zip(r_cos, c_cos, strict=True):
            edges.append((all_ids[i + int(r)], all_ids[int(c)]))

        for row_offset in range(end_i - i):
            global_row = i + row_offset
            xor = np.bitwise_xor(phash[global_row : global_row + 1], phash[global_row + 1 :])
            dists = lut[xor].sum(axis=-1)
            match_cols = np.where(dists <= phash_threshold)[0]
            for col in match_cols:
                edges.append((all_ids[global_row], all_ids[global_row + 1 + int(col)]))
    return edges


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.rebuild_splits")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, default=Path("embeddings/openai_clip-vit-base-patch16.npz"))
    ap.add_argument("--hashes", type=Path, default=Path("embeddings/hashes.npz"))
    ap.add_argument("--cosine-threshold", type=float, default=0.95)
    ap.add_argument("--phash-threshold", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("splits.v2.json"))
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    all_ids = [s.sample_id for s in samples]

    z_emb = np.load(a.embeddings, allow_pickle=False)
    feats = align_cache_by_ids(all_ids, z_emb["ids"].tolist(), z_emb["feats"], str(a.embeddings))

    z_hash = np.load(a.hashes, allow_pickle=False)
    if "ids" not in z_hash:
        raise SystemExit(f"ERROR: {a.hashes} does not contain an 'ids' array for safe alignment.")
    phash = align_cache_by_ids(all_ids, z_hash["ids"].tolist(), z_hash["phash"], str(a.hashes))

    edges: list[tuple[str, str]] = []

    # 1. Existing Sample.group equivalences
    group_members = defaultdict(list)
    for s in samples:
        group_members[s.group].append(s.sample_id)
    for members in group_members.values():
        for i in range(len(members) - 1):
            edges.append((members[i], members[i + 1]))

    # 2. Pairwise Cosine and pHash checks (strictly aligned to all_ids)
    print(f"Finding duplicate edges across all {len(samples)} samples...")
    edges.extend(
        find_duplicate_edges(
            all_ids,
            feats,
            phash,
            cosine_threshold=a.cosine_threshold,
            phash_threshold=a.phash_threshold,
        )
    )

    print(f"Total graph edges (group + duplicate): {len(edges)}")
    node_to_comp = get_connected_components(all_ids, edges)
    unique_comps = len(set(node_to_comp.values()))
    print(f"Compressed 10,000 samples into {unique_comps} connected component groups.")

    # 3. StratifiedGroupKFold split
    y = [f"{s.source}|{int(s.harmful)}" for s in samples]
    groups = [node_to_comp[s.sample_id] for s in samples]

    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    folds = [idx for _, idx in cv.split(np.zeros(len(samples)), y, groups)]

    def ids(idxs) -> list[str]:
        return sorted(samples[i].sample_id for i in idxs)

    out_splits = {
        "test": ids(folds[0]),
        "val": ids(folds[1]),
        "train": ids(i for f in folds[2:] for i in f),
    }

    raw_json = json.dumps(out_splits, indent=1)
    a.out.write_text(raw_json, encoding="utf-8")
    sha256 = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
    print(f"Wrote {a.out} | sha256: {sha256}")
    print({k: len(v) for k, v in out_splits.items()})
    # Save the component mapping alongside the splits
    comp_path = Path("embeddings/components.json")
    comp_path.write_text(json.dumps(node_to_comp, indent=1), encoding="utf-8")
    print(f"Wrote component map to {comp_path}")


if __name__ == "__main__":
    main()
