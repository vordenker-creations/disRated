# disRated

> **Current Status (v0.1.2 — Active Research Prototype):** This package is an experimental evaluation harness under active development (currently at **Phase 3 of 4**), accompanying a VKU research benchmark study on selective visual moderation.
> * **Implemented & Verified:** Top-level `disrated` CLI, dataset ingestion (`VHD11K`), perceptual-hash and feature-space deduplication (`splits.v2.json`), crash-resumable `.jsonl` evaluation runners, dummy/NudeNet baselines, Tier 0 CLIP linear probes (`0.882` test ROC-AUC), one-knob selective routing sweeps, Apple Silicon Tier 2 VLM adapter (`d1-2b-mlx`), and Phase 3 validation gate (`disrated val-gate`).
> * **In Progress / Not Yet Implemented:** Additional Tier 2 VLM adapters (generative CoT / CUDA server backbones) and Phase 4 held-out `test` cascade evaluation (`splits.v2.json["test"]` remains sealed for Tier 2). Do not use this in production.
> 
> 

**`disRated`** is a headless evaluation harness and selective-classification (cascade triage) benchmark suite for visual Trust & Safety moderation.

Built around the **VHD11K** multimodal harm benchmark, `disRated` tests whether lightweight perceptual probes (Tier 0) can confidently triage obvious traffic and route only ambiguous edge cases (`|logit(p)| < c`) to expensive Vision-Language Models (Tier 2).

---

## Project Status & Roadmap

| Phase | Milestone | Status | Key Deliverables |
| --- | --- | --- | --- |
| **Phase 1** | **Harness, Split Hygiene & Tier 0 Baseline** | **Done** | Leak-remediated `splits.v2.json` (8,449 isolated components), SHA-256 provenance gates, NudeNet floor (`0.518` AUC), CLIP ViT-B/16 probe (`0.882` test AUC). |
| **Phase 2** | **Offline Risk–Coverage & Protocol Lock** | **Done** | Single-knob `\|logit(p)\| < c` routing sweeps, bootstrap CIs, pre-registered [`HYPOTHESIS.md`](HYPOTHESIS.md). |
| **Phase 3** | **Tier 2 VLM Integration & Val Gate** | **In Progress** | `LiquidAI/d1-2B-MLX-fp16` System-1 adapter, Platt calibration & in-band superiority gate (`disrated val-gate`). |
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

For local development and full benchmark reproduction using [`uv`](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/vordenker/disrated.git
cd disrated
uv sync                 # core harness (CPU / MPS / CUDA CLIP probes & scoring)
uv sync --extra mac     # Apple Silicon MLX Tier 2 VLM (d1-2b-mlx)
uv sync --extra cuda    # Linux CUDA FP8 Tier 2 VLM (qwen3-vl-8b-fp8)
```

---

## Quickstart & Usage

### 1. Run Baseline Moderators

Run deterministic baselines or off-the-shelf detectors over a split. Runs are automatically resumable if interrupted:

```bash
# Deterministic pixel-hash random baseline
uv run disrated evalrun \
  --model random \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --split val

# NudeNet anatomical detector floor baseline
uv run disrated evalrun \
  --model nudenet \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --split all \
  --batch-size 16
```

### 2. Score Predictions

Compute binary classification metrics, ROC-AUC, Average Precision, coverage, and per-source (`concept` vs. `sdxl`) or per-concept breakdowns:

```bash
uv run disrated score results/nudenet-all.jsonl \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --split all \
  --concepts
```

### 3. Extract Embeddings & Fit Tier 0 Probe

Cache unit-normalised `openai/clip-vit-base-patch16` embeddings and fit the L2-regularised logistic regression probe across the isolated connected-component splits:

```bash
# Extract and cache 512-d image features
uv run disrated embed \
  --model openai/clip-vit-base-patch16 \
  --dataset /path/to/dataset_10000_1000 \
  --batch-size 32

# Tune regularization on val, write val/test/OOF prediction stores
uv run disrated probe \
  --embeddings embeddings/openai_clip-vit-base-patch16.npz \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --name clip-b16-v2
```

Zero-shot CLIP baseline (no training) from the same cache; the prompt set is chosen on `val` by ROC-AUC before `test` is written:

```bash
uv run disrated zeroshot \
  --embeddings embeddings/openai_clip-vit-base-patch16.npz \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --name clip-b16-zeroshot
```

### 4. Risk–Coverage, Tier 2 Validation Gate & Routing Analysis

Evaluate single-knob logit uncertainty routing (`|logit(p)| < c`) and run the Phase 3 validation gate (`val` only):

```bash
uv run disrated risk-coverage \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json

uv run disrated val-gate \
  --dataset /path/to/dataset_10000_1000 \
  --splits splits.v2.json \
  --probe results/clip-b16-v2-val.jsonl \
  --vlm results/d1-2b-mlx-v2-val.jsonl
```

### 5. Dataset Integrity & Leakage Audits

Re-run the perceptual hash, feature proximity, and connected-component split generators:

```bash
uv run disrated audit-hashes --dataset /path/to/dataset_10000_1000 --splits splits.v2.json
uv run disrated audit-clip --dataset /path/to/dataset_10000_1000 --splits splits.v2.json
uv run disrated rebuild-splits --dataset /path/to/dataset_10000_1000 --out splits.v2.json
```

---

## Empirical Baselines (VHD11K `splits.v2.json`)

All results below are evaluated against VHD11K's binary `decision` judge labels on the component-isolated `splits.v2.json` partition (`sha256: 07b1776d...`, 8,000 train / 1,000 val / 1,000 test across 8,449 connected components).

### Held-Out Test Split Performance (`N = 1,000`, `threshold = 0.50`)

| Model / Tier | Split | Cov | Acc | Prec | Recall | Spec | Macro-F1 | AP | ROC-AUC | Encoder Latency (ms/img) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `constant-0` (Always Safe) | `test` | 1.000 | 0.500 | - | 0.000 | 1.000 | 0.333 | 0.500 | 0.500 | - |
| `constant-1` (Always Harm) | `test` | 1.000 | 0.500 | 0.500 | 1.000 | 0.000 | 0.333 | 0.500 | 0.500 | - |
| `random` (Pixel SHA-256) | `test` | 1.000 | 0.508 | 0.508 | 0.502 | 0.514 | 0.508 | 0.505 | 0.506 | 1.17 (CPU) |
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

## Benchmark Scope & Adapting Beyond VHD11K

`disRated` `v0.1.x` is intentionally scoped as a controlled evaluation harness for **VHD11K** (`all_10000_evenHarmfulUnharmful`), which pairs harmful images 1:1 against visually matched hard negatives across web (`concept`) and synthetic (`sdxl`) cohorts and defines ground truth via a 3-VLM debate judge. Accordingly, it models **post-pre-filter semantic triage** rather than raw ~1%-prevalence platform upload streams.

Adapting the core runner and routing pipeline to production or custom datasets involves four extensions:
1. **Asymmetric Funnel Thresholds (`[tau_clear, tau_block]`):** Replacing balanced symmetric `|logit(p)| < c` routing with prior- and cost-calibrated safe-clearing / escalation thresholds (see `disrated sweep-thresholds`).
2. **Multi-Head Policy Probes:** Fitting separate lightweight linear heads per policy category over cached embeddings rather than collapsing diverse harms into a single binary target.
3. **Log-Odds Fusion:** Combining Tier 0 and Tier 2 calibrated log-odds on escalated items rather than discarding Tier 0's margin.
4. **Custom Manifest Loader:** Plugging a generic `.jsonl`/`.csv` dataset loader alongside `disrated.data.vhd11k`.

---

## Repository Structure

```text
disrated/
├── CLAUDE.md                        # Methodological invariants & contributor rules
├── HYPOTHESIS.md                    # Pre-specified evaluation protocol & falsification rules
├── pyproject.toml                   # Package metadata and dependencies
├── splits.v2.json                   # Component-isolated 80/10/10 split manifest
├── src/
│   └── disrated/
│       ├── __init__.py              # Top-level `disrated` CLI router
│       ├── __main__.py              # `python -m disrated` entrypoint
│       ├── data/
│       │   ├── splits.py            # Stratified group k-fold partitioning
│       │   └── vhd11k.py            # VHD11K CSV parser and integrity loader
│       ├── metrics/
│       │   └── scoring.py           # Binary metrics, slice aggregation, latency stats
│       ├── models/
│       │   ├── base.py              # Abstract BaseModerator contract
│       │   ├── d1_mlx.py            # LiquidAI/d1-2B-MLX-fp16 System-1 VLM adapter
│       │   ├── dummy.py             # Constant and pixel-hash Random baselines
│       │   ├── nudenet_mod.py       # NudeNet v3 ONNX detector wrapper
│       │   └── qwen3_vl.py          # Qwen3-VL-8B-Instruct-FP8 single-pass logit adapter
│       ├── audit_clip.py            # Cross-split cosine similarity & SDXL seed audit
│       ├── audit_components.py      # Connected-component size distribution inspector
│       ├── audit_dhash_residuals.py # Gradient hash residual inspector
│       ├── audit_duplicate_labels.py# Inter-duplicate label consistency audit
│       ├── audit_effect.py          # Leakage effect-size bootstrap analyser
│       ├── audit_hashes.py          # 64-bit pHash & dHash pairwise split auditor
│       ├── embed.py                 # Batched vision feature extractor (CUDA / MPS / CPU)
│       ├── finetune.py              # Full fine-tuning baselines: ResNet-50, ConvNeXt-T, ViT-B/16, CLIP (RQ2)
│       ├── eval_test.py             # Phase 4 held-out test evaluator from locked val gate
│       ├── evalrun.py               # CLI entrypoint for model inference runs
│       ├── probe.py                 # Logistic regression probe trainer & OOF evaluator
│       ├── rebuild_splits.py        # Graph connected-components split builder
│       ├── risk_coverage.py         # One-knob logit routing & bootstrap CI table
│       ├── runner.py                # Resumable JSONL runner & clean pixel loader
│       ├── schema.py                # Pydantic ModeratorOutput and Prediction schemas
│       ├── score.py                 # CLI entrypoint for scoring and provenance checks
│       ├── summarize.py             # Multi-seed bootstrap CIs & paired run comparisons
│       ├── sweep_thresholds.py      # Two-knob asymmetric threshold grid search
│       ├── val_gate.py              # Phase 3 Platt calibration & validation gate lock
│       └── zeroshot.py              # Zero-shot CLIP baseline from cached embeddings (RQ1)
└── tests/
    ├── test_cli.py                  # Top-level CLI dispatch & help/version tests
    ├── test_d1_mlx.py               # d1-2B MLX adapter & blind-pixel contract tests
    ├── test_eval_test.py            # Phase 4 held-out evaluator & criteria unit tests
    ├── test_evalrun.py              # End-to-end evalrun, coverage & provenance hash tests
    ├── test_metrics.py              # Parity tests against scikit-learn references
    ├── test_qwen3_vl.py             # Qwen3-VL logit extraction & refusal fallback tests
    ├── test_rebuild_splits.py       # Cache ID alignment & shuffled-cache invariance tests
    ├── test_runner.py               # Crash recovery, config/split lock, pixel-strip tests
    ├── test_val_gate.py             # Cluster bootstrap, MC random-control & Platt tests
    ├── test_zeroshot.py             # Zero-shot scoring, prompt ensembling & val-only selection tests
    └── test_vhd11k.py               # Filename regex, taxonomy & split disjointness tests
```

---

## Running Tests

```bash
uv run pytest
```