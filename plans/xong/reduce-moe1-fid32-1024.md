---
status: done
finished: 2026-04-22
started: 2026-04-22
---

# Goal

Reduce `moe1-ablation` final inference/ranking from FID 128-step with 4096 samples to FID 32-step with 1024 samples.

# Approach

Centralize the ablation FID timestep/sample settings and metric key so all ranking, tables, M17 selection, and master WandB summary stay consistent.

# Tasks

- [x] Add centralized `ABLATION_FID_TIMESTEPS`, `ABLATION_FID_GENERATIONS`, and metric-key helpers.
- [x] Update train/inference args to use 32 timesteps and 1024 generations.
- [x] Replace hard-coded `fid128_4096` table/ranking/summary references with the configured metric key.
- [x] Compile/check the touched Python file.
