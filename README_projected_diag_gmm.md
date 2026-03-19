# Projected Diagonal GMM Source Prior

## 1. Overview — What Was Implemented

This adds a new training mode `train_type="projected_diag_gmm"` to the DiT/flow-matching pipeline. Instead of sampling source noise from a standard Gaussian N(0, I), this mode learns a **diagonal Gaussian Mixture Model (GMM)** on the target latent space. Sources are sampled from the GMM, **projected onto a sphere**, then scaled by a random per-sample radius.

### New Files Created

| File | Description |
|------|-------------|
| `utils/projected_diag_gmm.py` | Pure JAX utility functions for GMM (router posterior, source sampling, projection, losses) |
| `utils/train_state_projected_gmm.py` | Extended TrainState with nested `{"model": ..., "prior": ...}` params |
| `model_router_cond.py` | DiT variant with optional router conditioning embedding |
| `baselines/targets_projected_diag_gmm.py` | Time/label/radius sampling (gradient-free operations) |
| `helper_inference_projected_gmm.py` | Inference with GMM source initialization |
| `helper_eval_projected_gmm.py` | Evaluation/FID with GMM source initialization |

### Patches to Existing Files

- **`train.py`**: Added GMM config fields, model creation branch, new init with prior params, new update branch with GMM loss_fn, eval/inference dispatch

---

## 2. Training Flow (Step-by-Step)

```
Input: batch of images
         |
    [Stable VAE encode]
         |
    x_1 = latent [B, 32, 32, 4]
         |
    +----|----+
    |         |
    |    [Outside loss_fn - gradient free]
    |    1. Sample t ~ Uniform{0..T-1}/T
    |    2. Sample dt_base = log2(T)
    |    3. Apply label dropout for CFG
    |    4. Sample radius r ~ logit-normal on [63.75, 64.25]
    |         |
    |    [Inside loss_fn - differentiable]
    |    5. Flatten x_1 -> [B, 4096]
    |    6. Router posterior: q(k|x_1) = softmax(log pi_k + log N(x_1; mu_k, Sigma_k))
    |    7. Top-M selection -> pick M modes with highest q
    |    8. Sample source per mode: y = mu_k + sigma_k * eps, eps ~ N(0,I)
    |    9. Project: x0_dir = y / ||y||
    |    10. Scale: x0 = stop_gradient(r) * x0_dir
    |    11. Build flow: x_t = (1-t)*x0 + t*x1, v_t = x1 - x0
    |    12. Reshape [B*M, H, W, C], forward DiT once
    |    13. MSE weighted by q_top -> L_FM
    |    14. L_mix = -mean(log p_mix(x_1))  [full-K posterior]
    |    15. L_bal = sum((mean_q_k - 1/K)^2)  [full-K posterior]
    |    16. L = L_FM + lambda_mix * L_mix + lambda_bal * L_bal
    |         |
    [jax.grad -> gradients for both model AND prior params]
         |
    [optax update -> new params]
         |
    [EMA update]
```

### Why Prior Must Be Inside loss_fn

In the existing codebase, `get_targets()` runs **outside** `loss_fn()`. Since `jax.grad` only differentiates through `loss_fn`, the source/router/prior would not receive gradients if computed outside. The GMM mode computes everything inside `loss_fn` so that:

- `L_FM` gradients flow through `x_t -> x_0 -> safe_project -> mu_k, sigma_k` (reparameterization trick)
- `L_mix` gradients flow through `log_mixprob -> pi_logits, mu, sigma`
- `L_bal` gradients flow through `q_full -> pi_logits, mu, sigma`

### Param Tree Structure

```
Naive:  params = {FinalLayer_0: ..., DiTBlock_0: ..., ...}
GMM:    params = {"model": {FinalLayer_0: ..., ...}, "prior": {pi_logits, mu, r_raw}}
```

The optimizer sees the full nested tree. Both model and prior params get updated by a single `jax.grad` call.

---

## 3. Inference Flow (Step-by-Step)

```
1. Load prior_params from EMA
2. Sample mode k ~ Categorical(softmax(pi_logits))
3. Sample y = mu_k + sigma_k * eps, project onto sphere, scale by r
4. x = x_0 (GMM source)
5. for t in [0, 1/T, 2/T, ..., (T-1)/T]:
       v = model(x, t, dt, labels)
       x = x + v * (1/T)          # Euler step
6. Decode x through VAE -> image
```

The only difference from naive inference is step 2-3 (source initialization). The Euler integration loop is identical.

---

## 4. Differences from Naive Mode

| Aspect | Naive | Projected Diag GMM |
|--------|-------|-------------------|
| Source prior | N(0, I) | Learned diagonal GMM, projected + scaled |
| Target computation | Outside loss_fn | Inside loss_fn (differentiable) |
| Param tree | Flat model params | Nested `{"model": ..., "prior": ...}` |
| TrainState class | `TrainStateEma` | `TrainStateProjectedGMMEma` |
| Loss | MSE only | L_FM + lambda_mix * L_mix + lambda_bal * L_bal |
| Inference source | `jax.random.normal(...)` | Sample from learned GMM + project |

---

## 5. Quick Start

Minimal configuration:

```bash
python train.py \
    --model.train_type=projected_diag_gmm \
    --model.gmm_num_modes=8 \
    --model.gmm_top_m=1 \
    --model.gmm_use_router_cond=0 \
    --model.gmm_use_warmup=0 \
    --batch_size=32 \
    --max_steps=100000
```

---

## 6. Config Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `gmm_num_modes` | int | 8 | Number of GMM components (K) |
| `gmm_top_m` | int | 1 | Top-M modes per sample for flow loss |
| `gmm_use_router_cond` | int | 0 | Enable router conditioning in model (0/1) |
| `gmm_mix_weight` | float | 1.0 | Weight for mixture NLL loss |
| `gmm_mix_normalize_by_dim` | int | 1 | Normalize `loss_mix` by latent dim D (recommended to avoid loss scale domination) |
| `gmm_bal_weight` | float | 0.01 | Weight for balance loss |
| `gmm_varreg_weight` | float | 0.01 | Weight for variance regularizer to keep diag covariance near identity |
| `gmm_fm_pretrain_iters` | int | 0 | First N steps run FM-only pretrain (Gaussian source + no mix/bal/varreg) |
| `gmm_fm_pretrain_noise_std` | float | 1.0 | Std of Gaussian source used during FM-only pretrain |
| `gmm_fm_pretrain_random_pair` | int | 1 | Randomly permute x1 pairing during FM-only pretrain (0/1) |
| `gmm_proj_eps` | float | 1e-6 | Epsilon for safe projection |
| `gmm_cov_eps` | float | 1e-6 | Covariance floor for diagonal Gaussian |
| `gmm_use_warmup` | int | 0 | Enable GMM warm-up phase (0/1) |
| `gmm_warmup_iters` | int | 0 | Number of warm-up iterations |
| `gmm_warmup_mode` | str | "mix_only" | Warm-up loss: "mix_only" or "mix_bal" |
| `gmm_radius_low` | float | 63.75 | Lower bound for radius sampling |
| `gmm_radius_high` | float | 64.25 | Upper bound for radius sampling |
| `gmm_radius_mu` | float | 0.0 | Mean of logit-normal for radius |
| `gmm_radius_sigma` | float | 0.25 | Std of logit-normal for radius |
| `gmm_use_prior_ema` | int | 1 | Use EMA prior params for inference (0/1) |
| `gmm_stop_gradient_q_top` | int | 0 | Stop gradient through q_top in weighted MSE (0/1) |

---

## 7. Mathematical Formulation

### Prior Parameters

```
phi = {pi_logits [K], mu [K, D], r_raw [K, D]}
pi = softmax(pi_logits)
sigma = softplus(r_raw)
var = sigma^2 + eps_cov
```

### Router Posterior (Dense, Full-K)

```
q(k | x_1) = pi_k * N(x_1; mu_k, diag(var_k)) / sum_j pi_j * N(x_1; mu_j, diag(var_j))
```

### Top-M Selection

```
S_M(x_1) = top-M modes by q(k|x_1)
q_top(k) = q(k|x_1) / sum_{j in S_M} q(j|x_1)   for k in S_M
```

### Source Sampling

```
y_0^(k) = mu_k + sigma_k * eps,    eps ~ N(0, I)
x0_dir^(k) = y_0^(k) / sqrt(||y_0^(k)||^2 + eps_proj^2)
x_0^(k) = stop_gradient(r_i) * x0_dir^(k)
```

Where `r_i ~ logit-normal([r_low, r_high])` is per-sample, mode-independent.

### Flow Pair

```
x_t = (1 - (1-1e-5)*t) * x_0 + t * x_1
v_t = x_1 - (1-1e-5) * x_0
```

### Losses

```
L_FM = (1/B) * sum_i sum_{k in S_M(i)} q_top(k|x1_i) * ||v_theta(x_t_i^k, t_i) - v_t_i^k||^2

L_mix_raw = -(1/B) * sum_i log(sum_k pi_k * N(x1_i; mu_k, Sigma_k))
L_mix = L_mix_raw / D    (when gmm_mix_normalize_by_dim=1)

L_bal = sum_k (mean_q_k - 1/K)^2

L_VarReg = mean_{k,d} 0.5 * (var_{k,d} - 1 - log(var_{k,d}))
where var_{k,d} = softplus(r_raw_{k,d})^2 + eps_cov

L = L_FM + lambda_mix * L_mix + lambda_bal * L_bal + lambda_varreg * L_VarReg
```

Note: L_mix and L_bal use the **full dense posterior** (all K modes), not the top-M sparse version.

---

## 8. Warm-Up

When `gmm_use_warmup=1`, the first `gmm_warmup_iters` steps train only the prior (no flow loss):

- **`mix_only`**: `L = lambda_mix * L_mix` (default). Trains the GMM to fit the data distribution before asking the model to learn flow from it.
- **`mix_bal`**: `L = lambda_mix * L_mix + lambda_bal * L_bal`. Also encourages balanced mode usage during warm-up.

After warm-up, switches to the full loss automatically (via `jnp.where` for JIT compatibility).

## 8.1 FM-Only Pretrain (Gaussian + Random Pairing)

When `gmm_fm_pretrain_iters > 0`, the first N steps use:

- `x0 ~ N(0, I)` (scaled by `gmm_fm_pretrain_noise_std`)
- optional random pairing `x1 <- x1[perm]` when `gmm_fm_pretrain_random_pair=1`
- only `L_FM` active (all auxiliary losses are disabled in this phase)
- prior is frozen (no prior gradient/update, including no AdamW decay update on prior params)

After this phase, training automatically switches back to standard projected-GMM losses.

---

## 9. Router Conditioning

When `gmm_use_router_cond=1`:

- The model uses `model_router_cond.py` (DiT with `RouterCondEmbedder`)
- During training: the model receives a sparse `[B, K]` vector with the top-M posterior probabilities
- During inference: since there's no x_1 to compute posterior, the model receives the prior probabilities `softmax(pi_logits)` as conditioning
- The conditioning is added to the existing `c = te + ye + dte` as `c = te + ye + dte + ce`

---

## 10. `gmm_stop_gradient_q_top`

Controls whether gradient from flow loss flows through `q_top` into the router:

- **`0` (default)**: Gradient flows through. The router co-adapts with the model, learning to route based on reconstruction quality.
- **`1`**: `stop_gradient(q_top)`. Router trained only by L_mix and L_bal. Use this if you observe router collapse (all samples assigned to one mode).

---

## 11. Monitoring & Debugging

### Key Metrics to Watch

| Metric | What It Means | Red Flag |
|--------|---------------|----------|
| `router_entropy` | Average entropy of q(k\|x_1) | Near 0 = router collapsed to one mode |
| `router_usage_min` | Min average mode usage | Near 0 = some modes are dead |
| `router_usage_max` | Max average mode usage | Near 1 = one mode dominates |
| `router_sigma_min` | Min sigma across all modes | Near 0 = prior degenerating |
| `source_x0_norm_mean` | Mean norm of source after projection+radius | Should be ~1.0 |
| `source_proj_denom_min` | Min projection denominator | Near 0 = source near origin |
| `loss_mix` | GMM NLL | Should decrease during warm-up |
| `loss_mix_raw` | Unnormalized GMM NLL | Can be very large because it scales with latent dimension D |
| `loss_mix_norm_factor` | Mix normalization divisor | Typically equals latent dim D |
| `loss_bal` | Balance penalty | Should be small (~0.01) |
| `loss_varreg` | Unit-variance regularization | Should decrease/stabilize when covariance drifts |
| `prior_var_dev_abs_mean` | Mean absolute deviation of variance from 1 | Near 0 = covariance close to identity |
| `has_nan_loss` | NaN detected in loss | 1.0 = something exploded |
| `has_nan_router` | NaN in router posterior | 1.0 = numerical instability |

### Common Issues

- **Router collapse**: All `router_usage_*` concentrate on one mode. Try: increase `gmm_bal_weight`, enable warm-up, set `gmm_stop_gradient_q_top=1`.
- **Sigma collapse**: `router_sigma_min` goes to 0. Increase `gmm_cov_eps`.
- **NaN in training**: Check `has_nan_*` metrics. Usually caused by sigma too small or log-prob overflow. Increase `gmm_cov_eps` or `gmm_proj_eps`.
- **Loss doesn't decrease**: Check `in_gmm_warmup` to verify warm-up ended. Check `loss_flow` vs `loss_mix` to see which component is stuck.

---

## 12. Recommended Experiment Sequence

1. **Baseline (M=1, no warmup, no router_cond)**:
   ```
   --model.gmm_top_m=1 --model.gmm_use_warmup=0 --model.gmm_use_router_cond=0
   ```
   Verify: loss decreases, metrics are finite, no NaN.

2. **Add warm-up**:
   ```
   --model.gmm_use_warmup=1 --model.gmm_warmup_iters=5000 --model.gmm_warmup_mode=mix_only
   ```
   Verify: loss_mix decreases during warm-up, then loss_flow starts training.

3. **Increase M**:
   ```
   --model.gmm_top_m=2
   ```
   Note: effective batch for model forward becomes B*M. May need to reduce batch_size.

4. **Enable router conditioning**:
   ```
   --model.gmm_use_router_cond=1
   ```
   Verify: model creates RouterCondEmbedder params. Inference uses prior probs.

5. **If router collapses**, try:
   ```
   --model.gmm_stop_gradient_q_top=1 --model.gmm_bal_weight=0.1
   ```
