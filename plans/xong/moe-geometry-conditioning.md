---
status: done
started: 2026-04-28
finished: 2026-04-28
---

# Goal
Add CNN-based geometry conditioning to `naive-moe-source` so DiT sees fixed GMM mode center `mu_k` and angular prototype `s_bar[k,a]` in addition to ID embeddings and `rho`.

# Approach
- Add a small CNN geometry encoder in `model.py`.
- Pass a spatial geometry tensor `concat(mu_k, s_bar[k,a])` to DiT during train/eval/inference.
- Add config flags and metrics for geometry conditioning.
- Update math/run docs and run syntax checks.

# Tasks
- [x] Add CNN geometry encoder to DiT.
- [x] Build and pass geometry maps in train/eval/inference.
- [x] Add metrics/docs/config commands.
- [x] Run checks, commit, and push.
