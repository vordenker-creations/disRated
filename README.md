# disRated

> **Current Status (v0.1.0 — Active Research Prototype):** This package is an experimental evaluation harness under active development (currently at **Phase 2 of 4**), published early to reserve the `disrated` package name.
> * **Implemented & Verified:** Dataset ingestion (`VHD11K`), perceptual-hash and feature-space deduplication (`splits.v2.json`), crash-resumable `.jsonl` evaluation runners, dummy/NudeNet baselines, Tier 0 CLIP linear probes (`0.882` test ROC-AUC), and one-knob selective routing sweeps.
> * **Not Yet Implemented:** Tier 2 Vision-Language Model (VLM) escalation adapters, spatial region-of-interest (ROI) cropping, and a unified top-level `disrated` CLI binary (commands currently run via `python -m disrated.<module>`). Do not use this in production.
> 
> 

**`disRated`** is a headless evaluation harness and selective-classification (cascade triage) benchmark suite for visual Trust & Safety moderation.

Built around the **VHD11K** multimodal harm benchmark, `disRated` tests whether lightweight perceptual probes (Tier 0) can confidently triage obvious traffic and route only ambiguous edge cases (`|logit(p)| < c`) to expensive Vision-Language Models (Tier 2).

---

## Project Status & Roadmap

| Phase | Milestone | Status | Key Deliverables |
| --- | --- | --- | --- |
| **Phase 1** | **Harness, Split Hygiene & Tier 0 Baseline** | **Done** | Leak-remediated `splits.v2.json` (8,449 isolated components), SHA-256 provenance gates, NudeNet floor (`0.518` AUC), CLIP ViT-B/16 probe (`0.882` test AUC). |
| **Phase 2** | **Offline Risk–Coverage & Protocol Lock** | **Done** | Single-knob ` |
| **Phase 3** | **Tier 2 VLM Integration & Val Gate** | **Next** | Zero-shot VLM adapter, validation in-band superiority gate (`val` uncertain slice), locking threshold cutoff `c*`. |
| **Phase 4** | **4-Arm Held-Out Evaluation** | **Planned** | Paired bootstrap comparison across `probe-only`, `vlm-only`, `cascade`, and `random-control` on `test`. |

---

## Core Architecture & Design Principles

1. **Decoupled Inference & Scoring:** Model execution (`disrated.evalrun`) and metric computation (`disrated.score`) are strictly separated. Inference streams append-only, crash-resumable `.jsonl` records to disk alongside `.meta.json` configuration sidecars. Threshold sweeps, bootstrap confidence intervals, and routing simulations run offline in milliseconds without touching the GPU.
2. **Blind Pixel Ingest:** Dataset filenames and image headers frequently leak ground-truth labels (e.g., SDXL generation tags or prompt metadata). `load_clean()` decodes images to raw RGB pixel buffers and strips all EXIF, PNG text chunks, and file paths before any moderator sees the input.
3. **Graph-Isolated Split Hygiene:** Web-scraped and synthetic benchmarks suffer from near-duplicate contamination across splits. `disRated` constructs an undirected graph over prompt-index groups, 64-bit perceptual hashes (`pHash <= 8`), and CLIP feature proximity (`cosine >= 0.95`), extracting connected components so near-duplicates do not straddle `train`, `val`, or `test` at the audited thresholds.
4. **Cryptographic Split Provenance:** Every prediction record embeds the `sha256` digest of the split manifest used to generate it. The scorer verifies this digest on load and refuses to evaluate mismatched partitions.

---

## Installation

Install the published package from PyPI:

```bash
pip install disrated

```

For local development and full benchmark reproduction using [`uv`](https://www.google.com/search?q=https://github.com/astral-sh/uv):

```bash
git clone https://github.com/vordenker/disrated.git
cd disrated
uv sync

```

---

## Quickstart & Usage

### 1. Run Baseline Moderators

Run deterministic baselines or off-the-shelf detectors over a split. Runs are automatically resumable if interrupted:

```bash
# Deterministic pixel-hash random baseline
uv run python -m disrated.evalrun \
  --model random \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --split test

# NudeNet anatomical detector floor baseline
uv run python -m disrated.evalrun \
  --model nudenet \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --split all \
  --batch-size 16

```

### 2. Score Predictions

Compute binary classification metrics, ROC-AUC, Average Precision, coverage, and per-source (`concept` vs. `sdxl`) or per-concept breakdowns:

```bash
uv run python -m disrated.score results/nudenet-all.jsonl \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --split all \
  --concepts

```

### 3. Extract Embeddings & Fit Tier 0 Probe

Cache unit-normalised `openai/clip-vit-base-patch16` embeddings and fit the L2-regularised logistic regression probe across the isolated connected-component splits:

```bash
# Extract and cache 512-d image features
uv run python -m disrated.embed \
  --model openai/clip-vit-base-patch16 \
  --dataset /path/to/dataset_10000_1000 \
  --batch-size 32

# Tune regularization on val, write val/test/OOF prediction stores
uv run python -m disrated.probe \
  --embeddings embeddings/openai_clip-vit-base-patch16.npz \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --name clip-b16-v2

```

### 4. Risk–Coverage & Routing Analysis

Evaluate single-knob logit uncertainty routing (`|logit(p)| < c`) with 2,000-sample non-parametric bootstrap confidence intervals:

```bash
uv run python -m disrated.risk_coverage

```

### 5. Dataset Integrity & Leakage Audits

Re-run the perceptual hash, feature proximity, and connected-component split generators:

```bash
uv run python -m disrated.audit_hashes --dataset /path/to/dataset_10000_1000 --splits splits.v2.json
uv run python -m disrated.audit_clip --dataset /path/to/dataset_10000_1000 --splits splits.v2.json
uv run python -m disrated.rebuild_splits --dataset /path/to/dataset_10000_1000 --out splits.v2.json

```

---

## Empirical Baselines (VHD11K `splits.v2.json`)

All results below are evaluated against VHD11K's binary `decision` judge labels on the component-isolated `splits.v2.json` partition (`sha256: 07b1776d...`, 8,000 train / 1,000 val / 1,000 test across 8,449 connected components).

### Held-Out Test Split Performance (`N = 1,000`, `threshold = 0.50`)

| Model / Tier | Split | Cov | Acc | Prec | Recall | Spec | Macro-F1 | AP | ROC-AUC | Encoder Latency (ms/img) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `constant-0` (Always Safe) | `test` | 1.000 | 0.500 | - | 0.000 | 1.000 | 0.333 | 0.500 | 0.500 | - |
| `constant-1` (Always Harm) | `test` | 1.000 | 0.500 | 0.500 | 1.000 | 0.000 | 0.333 | 0.500 | 0.500 | - |
| `random` (Pixel SHA-256) | `test` | 1.000 | 0.496 | 0.496 | 0.482 | 0.510 | 0.496 | 0.504 | 0.514 | - |
| `nudenet` (5-Class Explicit) | `all` | 1.000 | 0.512 | 0.754 | 0.036 | 0.988 | 0.369 | 0.522 | 0.518 | 15.28 (CPU) |
| **`clip-b16-v2` (Tier 0 Probe)** | `test` | **1.000** | **0.803** | **0.816** | **0.782** | **0.824** | **0.803** | **0.884** | **0.882** | **11.58 (MPS)** |
| ├── `concept` cohort (`n=500`) | `test` | 1.000 | 0.808 | 0.824 | 0.784 | 0.832 | 0.808 | 0.890 | 0.883 | 11.58 (MPS) |
| └── `sdxl` cohort (`n=500`) | `test` | 1.000 | 0.798 | 0.809 | 0.780 | 0.816 | 0.798 | 0.877 | 0.880 | 11.58 (MPS) |

### Tier 0 One-Knob Selective Routing (`clip-b16-v2`)

Escalating the `k%` of validation samples closest to the decision boundary (`|logit(p)| < c_k`, with cutoff `c_k` transferred unchanged to `test`):

| Val Target (`k`) | Split | Realised Esc | Band `n` | Resolved Acc [95% CI] | In-Band Acc [95% CI] | Auto-Safe NPV | Auto-Harm PPV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0.10** | `val` | 0.100 | 100 | 0.799 [0.773, 0.825] | 0.450 [0.355, 0.550] | 0.800 | 0.797 |
| **0.10** | `test` | 0.092 | 92 | 0.829 [0.803, 0.853] | 0.543 [0.441, 0.640] | 0.812 | 0.848 |
| **0.20** | `val` | 0.200 | 200 | 0.830 [0.803, 0.855] | 0.500 [0.432, 0.571] | 0.831 | 0.829 |
| **0.20** | `test` | 0.164 | 164 | 0.853 [0.827, 0.876] | 0.549 [0.471, 0.627] | 0.843 | 0.863 |
| **0.30** | `val` | 0.300 | 300 | 0.856 [0.829, 0.881] | 0.550 [0.492, 0.605] | 0.853 | 0.859 |
| **0.30** | `test` | 0.266 | 266 | 0.871 [0.844, 0.894] | 0.617 [0.559, 0.673] | 0.858 | 0.884 |

---

## Repository Structure

```text
disrated/
├── HYPOTHESIS.md                    # Pre-specified evaluation protocol & falsification rules
├── pyproject.toml                   # Package metadata and dependencies
├── splits.v2.json                   # Component-isolated 80/10/10 split manifest
├── src/
│   └── disrated/
│       ├── data/
│       │   ├── splits.py            # Stratified group k-fold partitioning
│       │   └── vhd11k.py            # VHD11K CSV parser and integrity loader
│       ├── metrics/
│       │   └── scoring.py           # Binary metrics, slice aggregation, latency stats
│       ├── models/
│       │   ├── base.py              # Abstract BaseModerator contract
│       │   ├── dummy.py             # Constant and pixel-hash Random baselines
│       │   └── nudenet_mod.py       # NudeNet v3 ONNX detector wrapper
│       ├── audit_clip.py            # Cross-split cosine similarity & SDXL seed audit
│       ├── audit_components.py      # Connected-component size distribution inspector
│       ├── audit_dhash_residuals.py # Gradient hash residual inspector
│       ├── audit_duplicate_labels.py# Inter-duplicate label consistency audit
│       ├── audit_effect.py          # Leakage effect-size bootstrap analyser
│       ├── audit_hashes.py          # 64-bit pHash & dHash pairwise split auditor
│       ├── embed.py                 # Batched CLIP feature extractor (CUDA / MPS / CPU)
│       ├── evalrun.py               # CLI entrypoint for model inference runs
│       ├── probe.py                 # Logistic regression probe trainer & OOF evaluator
│       ├── rebuild_splits.py        # Graph connected-components split builder
│       ├── risk_coverage.py         # One-knob logit routing & bootstrap CI table
│       ├── runner.py                # Resumable JSONL runner & clean pixel loader
│       ├── schema.py                # Pydantic ModeratorOutput and Prediction schemas
│       └── score.py                 # CLI entrypoint for scoring and provenance checks
└── tests/
    ├── test_metrics.py              # Parity tests against scikit-learn references
    ├── test_runner.py               # Crash recovery, config lock, and pixel-strip tests
    └── test_vhd11k.py               # Filename regex and group split disjointness tests

```

---

## Running Tests

```bash
uv run pytest

```