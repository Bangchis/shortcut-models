---
status: done
started: 2026-04-28
finished: 2026-04-28
---

# Goal

Add clearer stage-1 GMM health metrics, including min cluster fraction, so GMM ablation runs are easier to judge before MoE training.

# Approach

- Inspect current `data_prep.py` GMM metrics and WandB logging.
- Add interpretable cluster occupancy, variance, NLL, and stability metrics to `metrics.json` and WandB.
- Surface important GMM metrics in the `moe1-naive-k-ablation` stage outputs.
- Validate syntax and summarize dangerous grid values.

# Tasks

- [x] Inspect current GMM metrics.
- [x] Add stage-1 GMM health metrics.
- [x] Surface metrics in ablation summaries.
- [x] Run syntax checks.
- [x] Move plan to `plans/xong/`.
