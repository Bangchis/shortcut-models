---
status: done
started: 2026-04-27
finished: 2026-04-27
---

## Goal

Improve offline GMM pretraining stability for larger K, especially K=32, by reducing dead components and variance floor collapse without changing default behavior.

## Approach

Add optional regularization knobs to GMM EM:
- uniform mixture weight prior for `pi`
- diagonal variance prior toward standardized unit variance
- configurable low-count component reset threshold
- additional traces/logs for dead components, floor hits, min counts, and KL-to-uniform

## Tasks

- [x] Add optional EM regularization flags in `data_prep.py`
- [x] Extend `fit_diag_gmm` with pi/variance regularizers and traces
- [x] Save and log new diagnostics
- [x] Run syntax checks
