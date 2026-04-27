---
status: done
started: 2026-04-28
finished: 2026-04-28
---

## Goal

Replace `naive-moe-source` with the source-base angular/radius MoE method while keeping DiT flow matching in original VAE latent space.

## Approach

- Refactor GMM EM to use explicit uniform-pi interpolation and safe multiplicative variance scaling.
- Extend preprocessing with angular spherical k-means and lognormal radius statistics.
- Replace the old router/expert `SourceMoE` with a FiLM-conditioned ResCNN source network.
- Update training, eval, and inference to build source bases from fixed stats and compute posterior/alignment losses in standardized GMM space.

## Tasks

- [x] Implement GMM utility math and source-code helpers
- [x] Extend `data_prep.py` two-stage stats generation and logging
- [x] Replace `moe_source.py` with source-base ResCNN
- [x] Update `train.py` naive-moe-source training path and metrics
- [x] Update eval/inference source sampling
- [x] Run syntax checks
