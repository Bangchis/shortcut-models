# Projected Diag GMM (Stage A/B/C Refactor)

This document describes the current `train_type="projected_diag_gmm"` implementation.

## Overview

Training is now a single run with 3 explicit stages:

1. **Stage A (GMM warmup)**  
   Learn prior `psi={pi_logits, mu, r_raw}` on shell-normalized targets.
2. **Stage B (FM Gaussian warmup)**  
   Train flow model with Gaussian source only.
3. **Stage C (Joint)**  
   Hard top-1 routing, chi-radius source, and joint loss:
   `L_FM + lambda_bal * L_bal + lambda_var * L_var`.

Key constraints:

- No `L_mix` in Stage C.
- Hard top-1 routing (argmax) is enforced.
- No gradient through argmax decision.
- Prior is **not** updated every step in Stage C:
  gradients are accumulated for `gmm_prior_accum_steps` (default 10), averaged, then one update is applied.
- Eval/inference always use **online prior** (`params['prior']`), not prior EMA.
- Model EMA remains active when `model.use_ema=1`.

## Notation

- `x0`: source latent.
- `x1`: real target latent.
- `D = H * W * C` (for `32x32x4`, `D=4096`).
- `R0 = sqrt(D)`.
- Shell target:
  `y = R0 * x1 / (||x1|| + eps_proj)`.

Prior:

- `pi = softmax(pi_logits)`.
- `sigma = softplus(r_raw)`.
- `var = sigma^2 + eps_cov`.

## Stage Objectives

### Stage A: GMM warmup (prior-only update)

- Router posterior is computed on shell targets `y`.
- Loss:
  - `L_mix_pre_norm = -mean(log p_mix(y)) / D`
  - `L_bal = sum_k (mean_q_k - 1/K)^2`
  - `L_var = mean_{k,d} (log(var_{k,d}))^2`
  - `L_A = w_mix * L_mix_pre_norm + w_bal * L_bal + w_var * L_var`

### Stage B: FM Gaussian warmup (model-only update)

- Source:
  - `x0 ~ N(0, I)` (scaled by `gmm_stage_b_noise_std`)
  - Optional random pairing `x1 <- x1[perm]` (`gmm_stage_b_random_pair=1`)
- Loss:
  - `L_B = L_FM`

### Stage C: Joint

- Routing on shell target `y1`.
- Hard top-1 mode:
  - `k* = argmax_k q(k|y1)` with stop-gradient on index.
- Source:
  - `y0 = mu[k*] + sigma[k*] * eps`, `eps ~ N(0, I)`
  - `s0 = y0 / (||y0|| + eps_proj)`
  - `u ~ ChiSquare(D), R = sqrt(u)` (so `R ~ Chi(D)`)
  - `x0 = R * s0`
- Loss:
  - `L_C = L_FM + w_bal_joint * L_bal + w_var_joint * L_var`
  - No `L_mix` in Stage C.

## Prior Accumulation in Stage C

- Flow model updates every step.
- Prior gradients are accumulated for `N = gmm_prior_accum_steps`.
- On every `N`-th accumulated step:
  - apply one prior update with averaged gradient.
- On non-apply steps:
  - prior update is forced to zero (including AdamW decay path).

## Configs

### New primary configs

- `model.gmm_stage_a_iters`
- `model.gmm_stage_b_iters`
- `model.gmm_prior_accum_steps`
- `model.gmm_stage_a_mix_weight`
- `model.gmm_stage_a_bal_weight`
- `model.gmm_stage_a_var_weight`
- `model.gmm_joint_bal_weight`
- `model.gmm_joint_var_weight`
- `model.gmm_stage_b_noise_std`
- `model.gmm_stage_b_random_pair`

### Existing configs still used

- `model.gmm_num_modes`
- `model.gmm_top_m` (kept for compatibility, but hard top-1 is enforced)
- `model.gmm_use_router_cond`
- `model.gmm_proj_eps`
- `model.gmm_cov_eps`

### Legacy configs

Legacy warmup flags are ignored by the stage-based implementation:

- `gmm_use_warmup`
- `gmm_warmup_iters`
- `gmm_warmup_mode`

## Main Logged Metrics

- `training/gmm_stage_id` (0=A, 1=B, 2=C)
- `training/in_gmm_stage_a`, `training/in_gmm_stage_b`, `training/in_gmm_stage_c`
- `training/loss_flow`
- `training/loss_mix_pre`, `training/loss_mix_pre_raw` (Stage A only; zero outside A)
- `training/loss_bal` (A/C)
- `training/loss_varreg` (A/C)
- `training/prior_update_applied`
- `training/prior_accum_count`
- `training/prior_accum_progress`
- `training/source_radius_mean`, `training/source_radius_std`

## Inference / Eval

- Source sampling uses the same prior family with `R ~ Chi(D)` radius.
- Prior params are read from online `params['prior']`.
- Model forward can still use EMA model params depending on `model.use_ema`.
