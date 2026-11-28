# Human label audit (val, 120 images + 12 sentinels): analysis plan

> Status: **draft, to be agreed and committed before anyone looks at `disrated label score` output or the label files.**
> Purpose: answer question 3 — among the hard-band val images that both Tier 0 (CLIP B/16 + LR) and Qwen3-VL-8B get "wrong" relative to the VHD11K judge, how many are model misses and how many are questionable labels.

## 1. Data (fixed)

- Manifest: `results/val_audit_120_manifest.json` (sha256: `0f628af7354d9e8fd3ebc33aa1a6401fdce83f53a02c57fd3ef534b0621b8f26`), 132 slots = 120 primary images + 12 hidden sentinel repeats.
  - `in_band_both_wrong`: 60 (hard band |logit| < c* = 0.659, both models disagree with the judge)
  - `in_band_control`: 30 (hard band, both models agree with the judge)
  - `outer_resolved_control`: 30 (outside the hard band, both models agree with the judge)
- Raters (9, all 132/132), reported only as anonymous IDs R1–R9:
  - authors (2); external without the criteria panel (2); external with the criteria panel (5).
- Excluded before analysis (already decided): raw `*_blind.jsonl` files and incomplete raters.

## 2. Rater exclusion rule (applied before computing any judge-agreement number)

A rater is excluded from the primary analysis if **either**:
1. intra-rater consistency on the 12 sentinel pairs is below **9/12**, or
2. more than **10%** of the 120 primary items are skipped.

Excluded raters are listed with the reason; all results are also reported with every rater included (sensitivity).

## 3. Human reference label

- **Primary:** simple majority of retained raters' non-skipped votes (harmful vs safe). Borderline flags do not change the vote. A tie is **"undecided"**; undecided items are reported as a count and excluded from rate denominators.
- **Sensitivity:** strong majority (≥ 7/9 of retained raters, scaled if raters are excluded); unanimous consensus (as computed by `disrated label score`).

## 4. Primary outcomes

For each of the three strata:
- **Human–judge disagreement rate:** share of items where the human majority differs from the VHD11K label, with a Wilson 95% CI.
- **Contrast:** disagreement in `in_band_both_wrong` vs each control stratum (difference with 95% CI; Fisher exact test as a check).

For `in_band_both_wrong` specifically (the question-3 answer), each item is classified as:
- **questionable label:** human majority agrees with the models (both models predicted the same class, opposite to the judge);
- **model miss:** human majority agrees with the judge;
- **undecided:** tie.

## 5. Reliability

- Fleiss' kappa over retained raters (primary items only).
- Pairwise Cohen's kappa (from `disrated label score`).
- Per-rater sentinel self-consistency and borderline rate.

## 6. Secondary analyses (descriptive, no conclusions drawn from them alone)

- Authors only vs external raters only.
- External with criteria panel vs without.
- Borderline rate by stratum.
- Model accuracy vs human majority (not only vs the judge), per stratum.

## 7. What this audit does not change

- The audit is **descriptive**. It does not change the locked Phase 4 setup: Tier 0 (`clip-b16-v2`), primary VLM (Qwen3-VL-8B-Instruct-FP8), threshold c*, calibration or k*. The test set stays sealed.
- Rater names are never published; only R1–R9 and their group.
