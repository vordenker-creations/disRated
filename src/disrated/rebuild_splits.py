from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

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
    id_to_idx = {sid: idx for idx, sid in enumerate(z_emb["ids"].tolist())}
    feats = z_emb["feats"]

    z_hash = np.load(a.hashes, allow_pickle=False)
    phash = z_hash["phash"]
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    edges: list[tuple[str, str]] = []

    # 1. Existing Sample.group equivalences
    group_members = defaultdict(list)
    for s in samples:
        group_members[s.group].append(s.sample_id)
    for members in group_members.values():
        for i in range(len(members) - 1):
            edges.append((members[i], members[i + 1]))

    # 2. Pairwise Cosine and pHash checks
    print("Finding duplicate edges across all 10,000 samples...")
    chunk_size = 1000
    n = len(samples)
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        # Cosine chunk
        cos_chunk = np.matmul(feats[i:end_i], feats.T)
        # Zero out self and backward checks to avoid duplicate edges
        for row_offset in range(end_i - i):
            global_row = i + row_offset
            cos_chunk[row_offset, :global_row + 1] = 0.0

        r_cos, c_cos = np.where(cos_chunk >= a.cosine_threshold)
        for r, c in zip(r_cos, c_cos):
            edges.append((all_ids[i + r], all_ids[c]))

        # pHash chunk
        for row_offset in range(end_i - i):
            global_row = i + row_offset
            xor = np.bitwise_xor(phash[global_row:global_row + 1], phash[global_row + 1:])
            dists = lut[xor].sum(axis=-1)
            match_cols = np.where(dists <= a.phash_threshold)[0]
            for col in match_cols:
                edges.append((all_ids[global_row], all_ids[global_row + 1 + col]))

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