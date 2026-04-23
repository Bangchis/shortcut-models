---
status: done
started: 2026-04-23
finished: 2026-04-23
---

# Goal

Implement greedy hyperparameter tuning for `moe1-naive-k-ablation` with TPU v4-16 / 2-worker safe orchestration.

# Approach

- Extend the existing `moe1_naive_k_ablation.py` runner instead of using `moe1-ablation`.
- Add greedy stages for K, tau, balance, entropy, weight decay, and source var target.
- Keep checkpoints disabled and run final-only FID.
- Make GMM fitting and visualization process-0 only, while training subprocesses run on all workers in sync.

# Tasks

- [x] Inspect current mode, eval, inference, and data prep behavior.
- [x] Add/adjust flags for greedy tuning and final-only FID.
- [x] Implement greedy runner, candidate caching, stage summaries, and selection.
- [x] Add TPU multihost guards/barriers and batch validation.
- [x] Verify plots/summary outputs and no-checkpoint behavior.
- [x] Compile/smoke-test where possible.
