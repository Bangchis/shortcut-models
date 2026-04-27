---
status: done
started: 2026-04-28
finished: 2026-04-28
---

# Goal

Add built-in list/grid ablation support for GMM regularizer configs in `moe1-naive-k-ablation`, instead of only accepting one fixed `--moe1_gmm_*` config per run.

# Approach

- Inspect existing flag parsing and ablation job generation.
- Add list flags for sigma target, variance pull beta, and pi uniform beta.
- Make the ablation runner materialize those list values into configs and pass each config to GMM fitting.
- Validate syntax and provide the exact run command.

# Tasks

- [x] Confirm current single-value GMM flags are wired through.
- [x] Add list flags and config generation.
- [x] Wire list values into `_run_gmm` subprocess calls.
- [x] Run syntax/import sanity checks.
- [x] Move plan to `plans/xong/` with completion metadata.
