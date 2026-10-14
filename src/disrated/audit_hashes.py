from __future__ import annotations

import argparse
import json
from pathlib import Path
import imagehash
import numpy as np

from .data.vhd11k import load_vhd11k
from .runner import load_clean


def compute_hashes(samples, cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        return {"ids": z["ids"], "phash": z["phash"], "dhash": z["dhash"]}

    ids, ph_bits, dh_bits = [], [], []
    for i, s in enumerate(samples):
        img = load_clean(s.path)
        ids.append(s.sample_id)
        ph_bits.append(imagehash.phash(img).hash.flatten())
        dh_bits.append(imagehash.dhash(img).hash.flatten())
        if (i + 1) % 1000 == 0:
            print(f"hashed {i + 1}/{len(samples)}", flush=True)

    ids_arr = np.array(ids)
    phash_packed = np.packbits(np.array(ph_bits, dtype=bool), axis=1)
    dhash_packed = np.packbits(np.array(dh_bits, dtype=bool), axis=1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=ids_arr, phash=phash_packed, dhash=dhash_packed)
    return {"ids": ids_arr, "phash": phash_packed, "dhash": dhash_packed}


def hamming_chunked(a: np.ndarray, b: np.ndarray, chunk_size: int = 500) -> np.ndarray:
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    dists = np.zeros((a.shape[0], b.shape[0]), dtype=np.uint8)
    for start in range(0, a.shape[0], chunk_size):
        end = min(start + chunk_size, a.shape[0])
        xor = np.bitwise_xor(a[start:end, None, :], b[None, :, :])
        dists[start:end] = lut[xor].sum(axis=-1)
    return dists


def audit_hamming(samples, hashes: dict[str, np.ndarray], splits_path: Path) -> dict:
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    id_to_idx = {sid: idx for idx, sid in enumerate(hashes["ids"].tolist())}
    by_id = {s.sample_id: s for s in samples}

    split_pairs = [
        ("train", "val", splits["train"], splits["val"]),
        ("train", "test", splits["train"], splits["test"]),
        ("val", "test", splits["val"], splits["test"]),
    ]

    findings = {}
    for hname in ["phash", "dhash"]:
        print(f"\n=== {hname.upper()} CROSS-SPLIT HAMMING AUDIT ===")
        data = hashes[hname]
        for s1_name, s2_name, ids1, ids2 in split_pairs:
            idx1 = np.array([id_to_idx[sid] for sid in ids1])
            idx2 = np.array([id_to_idx[sid] for sid in ids2])
            dists = hamming_chunked(data[idx1], data[idx2])

            pair_key = f"{s1_name}_{s2_name}"
            findings.setdefault(hname, {})[pair_key] = {}

            print(f"\n--- {s1_name} ({len(ids1)}) vs {s2_name} ({len(ids2)}) ---")
            for t in [0, 4, 8]:
                r, c = np.where(dists <= t)
                breakdown = {"concept_concept": 0, "sdxl_sdxl": 0, "cross_source": 0}
                for i_pos, j_pos in zip(r, c):
                    s1 = by_id[ids1[i_pos]]
                    s2 = by_id[ids2[j_pos]]
                    if s1.source == "concept" and s2.source == "concept":
                        breakdown["concept_concept"] += 1
                    elif s1.source == "sdxl" and s2.source == "sdxl":
                        breakdown["sdxl_sdxl"] += 1
                    else:
                        breakdown["cross_source"] += 1

                findings[hname][pair_key][t] = {"total": len(r), **breakdown}
                print(
                    f"Dist <= {t:>2}: total={len(r):>5} | "
                    f"concept-concept={breakdown['concept_concept']:>4} | "
                    f"sdxl-sdxl={breakdown['sdxl_sdxl']:>4} | "
                    f"cross-source={breakdown['cross_source']:>4}"
                )
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m disrated.audit_hashes")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=Path("splits.json"))
    ap.add_argument("--cache", type=Path, default=Path("embeddings/hashes.npz"))
    a = ap.parse_args()

    samples, _ = load_vhd11k(a.dataset.expanduser())
    h = compute_hashes(samples, a.cache)
    audit_hamming(samples, h, a.splits)


if __name__ == "__main__":
    main()