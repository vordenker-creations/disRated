from pathlib import Path

from disrated.data.splits import make_splits
from disrated.data.vhd11k import (
    CONCRETE_HARM_CONCEPTS,
    CONTEXTUAL_NORMATIVE_CONCEPTS,
    Sample,
    concept_taxonomy,
    parse_name,
)


def test_parse_sdxl():
    r = parse_name("sdxl_extend_by_llama3_harmful_00418_seed2024.png")
    assert r["source"] == "sdxl" and r["intent"] is True and r["group"] == "sdxl:harmful:00418"


def test_parse_concept():
    r = parse_name("knife_in_hand_000_00007.jpeg")
    assert r["source"] == "concept" and r["concept"] == "knife_in_hand"


def _fake(n=2000):
    out = []
    for i in range(n):
        if i % 2:
            name = f"sdxl_extend_by_llama3_harmful_{i // 4:05d}_seed{2024 + i % 6}.png"
        else:
            name = f"gun_000_{i:05d}.jpeg"
        out.append(Sample(sample_id=name, path=Path(name), harmful=(i % 3 == 0), harm_type="", **parse_name(name)))
    return out


def test_splits_disjoint_complete_and_grouped():
    s = _fake()
    sp = make_splits(s, seed=0)
    ids = [x for v in sp.values() for x in v]
    assert len(ids) == len(set(ids)) == len({x.sample_id for x in s})
    which = {i: k for k, v in sp.items() for i in v}
    seen = {}
    for x in s:
        assert seen.setdefault(x.group, which[x.sample_id]) == which[x.sample_id]


def test_concept_taxonomy_partition_is_disjoint_and_45_tags():
    assert CONCRETE_HARM_CONCEPTS.isdisjoint(CONTEXTUAL_NORMATIVE_CONCEPTS)
    assert len(CONCRETE_HARM_CONCEPTS) + len(CONTEXTUAL_NORMATIVE_CONCEPTS) == 45
    assert concept_taxonomy("gun") == "concrete_harm"
    assert concept_taxonomy("alcohol") == "contextual_normative"
    assert concept_taxonomy(None) is None
