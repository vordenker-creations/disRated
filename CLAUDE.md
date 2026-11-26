# CLAUDE.md — `disRated` Engineering Guidelines

This repository (`vordenker-work/disRated`) is first-party research software by **Nguyễn Hoàng Thiên Bảo** (`vor` / `vordenker`) and **Nguyễn Thị Thúy Vân** (`nttvan`) for the VKU Faculty of AI & Data Science thesis *"Nghiên cứu xây dựng mô hình phân loại hình ảnh độc hại"*.

All code changes, adapters, scripts, and evaluation runs must strictly obey the four locked methodological invariants defined in [`HYPOTHESIS.md`](HYPOTHESIS.md):

## 1. Locked Methodological Invariants (Non-Negotiable)

1. **`test` Remains Strictly Sealed:**
   Never run inference, inspect error cases, or tune prompts, hyperparameters, or thresholds on `splits.v2.json["test"]` ($N = 1,000$) prior to locking the validation gate (`results/phase3_val_gate.json`). All prompt adjustments, calibration, validation superiority checks, and scalar cutoff selection ($c^*$) happen exclusively on `splits.v2.json["val"]` ($N = 1,000$).
2. **Canonical Split Manifest & Provenance Gate:**
   All active runs must target `splits.v2.json` (`sha256: 07b1776d1d8ec1949e799859fe321e993361d6998040534601f81e4b0e3b09f2`, 8,000 train / 1,000 val / 1,000 test across 8,449 isolated connected components). `disrated.evalrun` embeds `splits_sha256` in prediction records and `disrated.score` verifies it and refuses mismatched splits.
3. **Blind Pixel Ingest (`load_clean()`):**
   Every image fed to a moderator or feature extractor must pass through `disrated.runner.load_clean()`, which decodes images to stripped RGB `PIL.Image.Image` buffers (`img.info.clear()`, no `.filename`) so models cannot exploit EXIF metadata, PNG text chunks, or filename labels (`sdxl_extend_by_llama3_harmful_...`).
4. **Decoupled Inference & Scoring:**
   `disrated.evalrun` / `disrated.runner.run` only streams append-only, crash-resumable `.jsonl` records (`Prediction`) paired with `.meta.json` config sidecars. Leave `ModeratorOutput.verdict = None` for continuous-score moderators so offline threshold sweeps, paired bootstraps, and routing simulations in `disrated.score` and `disrated.risk_coverage` remain decoupled.

## 2. Core Contracts & Commands

- **Moderator interface:** Subclass `disrated.models.base.BaseModerator` (`name: str`, `config() -> dict[str, Any]`, `predict(images: list[Image.Image]) -> list[ModeratorOutput]`).
- **Prediction schema (`disrated.schema`):** `ModeratorOutput` fields are `score` ($\in [0, 1]$), `verdict`, `refused`, `raw_output`, `extra`. `runner.run()` wraps each into `Prediction` with `sample_id`, `latency_ms`, and `error`.
- **Tests:** Run `uv run pytest` after edits. Keep Apple-only packages (`mlx-vlm>=0.7.11`) isolated under `[project.optional-dependencies] mac` and CUDA FP8 VLM packages (`accelerate>=1.0.0`, `kernels>=0.17.0,<0.18.0`) under `[project.optional-dependencies] cuda` so each hardware target stays clean (`uv sync --extra mac` on Apple Silicon; `uv sync --extra cuda` on Linux CUDA nodes like `vor-lab`).
