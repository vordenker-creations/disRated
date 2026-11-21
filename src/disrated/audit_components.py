from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    comp_map = json.loads(Path("embeddings/components.json").read_text(encoding="utf-8"))
    counts = Counter(comp_map.values())
    size_dist = Counter(counts.values())

    print("\n=== COMPONENT SIZE DISTRIBUTION ===")
    print(f"{'Component Size':<16} | {'Count':<8} | {'Total Images':<12}")
    print("-" * 42)
    for size in sorted(size_dist.keys()):
        num_comps = size_dist[size]
        print(f"{size:<16} | {num_comps:<8} | {num_comps * size:<12}")

    largest_comp_id, max_size = counts.most_common(1)[0]
    largest_members = [sid for sid, cid in comp_map.items() if cid == largest_comp_id]
    print(f"\nLargest component: {largest_comp_id} (size = {max_size})")
    print("Members (first 10):", largest_members[:10])


if __name__ == "__main__":
    main()
