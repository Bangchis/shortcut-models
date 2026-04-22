---
status: done
started: 2026-04-23
finished: 2026-04-23
---

# Goal

Add a new `moe1-naive-k-ablation` mode that sweeps MoE source GMM K values, trains a naive baseline, avoids checkpoint saves, runs lightweight eval, and produces easy-to-compare source/data/path visualizations and summary stats.

# Approach

Implement a standalone runner based on `train.py --mode=train` and `model.train_type=naive-moe-source`, not on the existing `moe1-ablation` phase/composed workflow. Add the minimal flags needed by the Kaggle command, make periodic eval lightweight, and reuse plotting primitives where practical.

# Tasks

- [x] Add flags and train.py dispatch for `moe1-naive-k-ablation`.
- [x] Make `helper_eval.py` support lightweight eval visual/log and configurable periodic FID generations.
- [x] Implement the new runner with GMM fitting, per-K MoE train, naive train, no checkpoint saves, and final source/flow dumps.
- [x] Add visualization and summary outputs for per-K, baseline, combined plots, alignment, and ODE path stats.
- [x] Compile/check touched files and close the plan.
