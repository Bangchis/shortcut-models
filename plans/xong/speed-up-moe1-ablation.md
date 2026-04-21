---
status: done
finished: 2026-04-22
started: 2026-04-22
---

# Goal

Speed up `moe1-ablation` training/inference overhead across all stages without reducing the ablation settings, without skipping Phase 2 visualization, and while keeping the important summary statistics.

# Approach

Keep every configured run. Optimize per-step compute by avoiding heavy diagnostics unless explicitly requested, avoid building activation dictionaries when not logged, and run the MoE source network in TPU-friendly bf16 for ablation while preserving float32 outputs for losses/stats.

# Tasks

- [x] Add a `train_metrics_level` flag and use summary diagnostics for `moe1-ablation` runs.
- [x] Stop constructing DiT activation dictionaries unless `return_activations=True`.
- [x] Add optional SourceMoE compute dtype and set ablation SourceMoE compute to bf16.
- [x] Keep all existing Phase 1B/Phase 2/Phase 3 visualization and important stats paths intact.
- [x] Compile/check touched files and close the plan.
