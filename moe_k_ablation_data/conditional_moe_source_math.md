---
title: Conditional Source Coupling for naive-moe-source
date: 2026-04-28
---

# Goal

Replace the old learned SourceCNN source with a conditional Flow Matching setup:

```text
v_theta(x_t, t)        -> old unconditional velocity
v_theta(x_t, t, c)     -> new conditional velocity
```

where:

```text
c = (k, a, rho)
```

- `k`: diagonal GMM mode.
- `a`: angular submode inside mode `k`.
- `rho`: standardized log-radius code.

The key point is that the condition is visible to the DiT velocity model during
both training and inference. This is intended to reduce
`Var[V | x_t, t, c]`, not only to create a nicer source.

# Offline Stats

Latents are original VAE latents `x`. GMM/code fitting uses standardized
coordinates:

```text
x_std = (flat(x) - mean_data) / (std_data + eps)
```

Fit a diagonal GMM:

```text
p_GMM(x_std) = sum_k pi_k N(x_std; mu_k, diag(var_k))
q(k | x_std) = posterior under this GMM
```

The GMM M-step uses:

```text
pi_new = (1 - beta_pi) * pi_em + beta_pi / K
```

and multiplicative variance scaling:

```text
scale_k = (var_target / (mean_j var_em[k,j] + var_eps)) ^ beta_var
scale_k = clip(scale_k, scale_min, scale_max)
var_new[k,j] = max(var_em[k,j] * scale_k, var_floor)
```

No random dead-component reinit is used. Dead/low-count components are logged.

# Local Coordinates

For each sample assigned to mode `k`:

```text
u = diag(var_k)^(-eta/2) * (x_std - mu_k)
r = ||u||
s = u / (||u|| + eps)
```

Run spherical k-means on `s` inside each GMM mode and store centers:

```text
s_bar[k,a]
```

Fit radius stats:

```text
log r | k,a ~ N(m_r[k,a], sigma_r[k,a]^2)
rho = (log r - m_r[k,a]) / sigma_r[k,a]
```

# Training Joint Distribution

Training samples:

```text
x1 ~ p_data
k = argmax q(k | x1_std)
a = argmax_a s(x1)^T s_bar[k,a]
rho = (log r1 - m_r[k,a]) / sigma_r[k,a]
x0 ~ p(x0 | k,a,rho)
```

So:

```text
q(x0, x1, c) = p_data(x1) q(c | x1) p(x0 | c)
```

with `c=(k,a,rho)`.

# Analytic Conditional Source

Given `c=(k,a,rho)`, recover radius:

```text
r = exp(m_r[k,a] + sigma_r[k,a] * rho)
```

Sample a broad local direction around the angular prototype:

```text
eps_dir ~ N(0, I)
eps_tan = eps_dir - <eps_dir, s_bar> s_bar
eps_tan = eps_tan / sqrt(d)
s0 = normalize(s_bar[k,a] + sigma_dir * eps_tan)
```

The `1/sqrt(d)` term is important in high dimension; otherwise Gaussian
tangent noise dominates only because the dimension is large.

Build source in standardized GMM space:

```text
x0_std = mu_k + diag(var_k)^(eta/2) * (kappa * r * s0)
```

Then convert back to original VAE latent scale:

```text
x0 = unflatten(x0_std * (std_data + eps) + mean_data)
```

Flow Matching stays in original VAE latent space:

```text
x_t = (1 - t) x0 + t x1
v* = x1 - x0
```

The DiT input is:

```text
(x_t, t, k, a, rho)
```

and the loss is:

```text
L = E || v_theta(x_t, t, k, a, rho) - (x1 - x0) ||^2
```

There is no SourceCNN, no posterior alignment loss, no variance floor loss,
and no angular alignment loss in this conditional analytic variant.

# Inference

Inference samples from the condition prior:

```text
k ~ Cat(pi)
a ~ Cat(pi_a | k)
rho ~ N(0, 1)
r = exp(m_r[k,a] + sigma_r[k,a] rho)
s0 = normalize(s_bar[k,a] + sigma_dir * eps_tan / sqrt(d))
x0_std = mu_k + diag(var_k)^(eta/2) * (kappa * r * s0)
x0 = unstandardize(x0_std)
```

Then solve:

```text
dx/dt = v_theta(x, t, k, a, rho),  x(0)=x0
```

with the same `c=(k,a,rho)` at every ODE step.

# Why This Avoids The Previous Failure

The old source-base implementation tried to make `x0` smart, but the DiT still
effectively had to learn a mixed field if the condition was not strong enough.
If many incompatible paths pass through nearby `x_t`, the target velocity has
large conditional variance:

```text
Var[V | x_t, t]
```

The new objective asks for:

```text
Var[V | x_t, t, c]
```

That only helps if:

1. `c` is passed to the DiT in training.
2. The same `c` is passed to the DiT in inference.
3. `x0 | c` is not too close to `x1`, and not over-concentrated on a single
   angular ray.

# Metrics To Watch

Useful indicators:

```text
training/source/x0_minus_x1_norm
training/source/source_center_cos
training/source/target_center_cos
training/source/source_target_direction_cos
training/condition/rho_std
training/condition/effective_codes_batch
training/entangle/same_c_pair_count
training/entangle/xt_pair_dist_same_c
training/entangle/v_pair_dist_same_c
training/entangle/path_lipschitz_proxy
training/entangle/velocity_cos_same_c
training/activations/moe_condition_embed
```

`training_summary.csv` is written from the full scalar training metric dict. If
new scalar metrics appear after the first row, the writer expands the CSV header
and preserves existing rows, so important diagnostics are not dropped.

Interpretation:

- If `x0_minus_x1_norm` is too small, the source is too close and the path is
  too trivial.
- If `source_center_cos` is much larger than `target_center_cos`, source
  directions are too concentrated on prototypes.
- If `path_lipschitz_proxy` is high inside the same `(k,a)` code, paths remain
  entangled even after conditioning.
- If `moe_condition_embed` stays tiny and changing `c` has no effect, DiT is
  ignoring the condition.

# Suggested Hyperparameters

Baseline:

```text
K = 16
A = 4
eta = 0.5
kappa = 1.4
source_direction_noise = 2.0
```

Sweep if baseline is weak:

```text
kappa in {1.2, 1.4, 1.6}
source_direction_noise in {1.5, 2.0, 2.5}
eta in {0.0, 0.5}
```

If source is too close to data, increase `kappa`. If source directions are too
prototype-locked, increase `source_direction_noise`. If paths cross too much,
try lower `source_direction_noise` or increase angular resolution `A=8`.

# Kaggle Commands

Prepare stats:

```bash
%cd /kaggle/working/shortcut-models

!git checkout moe1-angle
!git pull --ff-only

!uv run python data_prep.py \
  --dataset_name=celebahq256 \
  --tfds_data_dir=/kaggle/input/shortcut-celebahq256/tensorflow_datasets \
  --batch_size=64 \
  --gmm_save_path=/kaggle/working/source_stats_k16_a4_eta05.npz \
  --gmm_latent_cache_path=/kaggle/working/source_stats_k16_a4_eta05_latents.npy \
  --gmm_fit_samples=-1 \
  --gmm_num_modes=16 \
  --gmm_em_iters=100 \
  --gmm_em_restarts=3 \
  --gmm_em_chunk_size=1024 \
  --gmm_pi_uniform_beta=0.01 \
  --gmm_var_update=scale \
  --gmm_var_beta=0.05 \
  --gmm_var_target=1.0 \
  --gmm_var_floor=1e-4 \
  --gmm_var_eps=1e-8 \
  --gmm_var_scale_min=0.5 \
  --gmm_var_scale_max=2.0 \
  --local_eta=0.5 \
  --angular_num_submodes=4 \
  --angular_min_cluster_size=100 \
  --radius_min_log_std=0.05 \
  --wandb.entity=Fingerprint_Recognition \
  --wandb.project=moe1-source-prep \
  --wandb.service_wait=300 \
  --wandb.start_method=thread \
  --wandb.disable_stats=1 \
  --wandb.save_code=0
```

Train:

```bash
%cd /kaggle/working/shortcut-models

!git checkout moe1-angle
!git pull --ff-only

!uv run python train.py \
  --mode=train \
  --dataset_name=celebahq256 \
  --tfds_data_dir=/kaggle/input/shortcut-celebahq256/tensorflow_datasets \
  --fid_stats=/kaggle/input/shortcut-celebahq256/data/celeba256_fidstats_ours.npz \
  --batch_size=64 \
  --max_steps=250000 \
  --log_interval=1000 \
  --eval_interval=25000 \
  --save_interval=999999999 \
  --save_dir=/kaggle/working/moe1_cond_source_k14_dn20_ckpt/ \
  --summary_csv_path=/kaggle/working/moe1_cond_source_k14_dn20_ckpt/training_summary.csv \
  --summary_csv_steps=1,1000,5000,10000,25000,50000,100000,150000,200000,250000 \
  --summary_csv_wandb_upload=1 \
  --wandb.entity=Fingerprint_Recognition \
  --wandb.project=moe1-conditional-source \
  --wandb.service_wait=300 \
  --wandb.start_method=thread \
  --wandb.disable_stats=1 \
  --wandb.save_code=0 \
  --model.hidden_size=768 \
  --model.patch_size=2 \
  --model.depth=12 \
  --model.num_heads=12 \
  --model.mlp_ratio=4 \
  --model.sharding=dp \
  --model.train_type=naive-moe-source \
  --model.source_variant=conditional-analytic \
  --model.gmm_stats_path=/kaggle/working/source_stats_k16_a4_eta05.npz \
  --model.gmm_num_modes=16 \
  --model.angular_num_submodes=4 \
  --model.local_eta=0.5 \
  --model.source_kappa=1.4 \
  --model.source_direction_noise=2.0 \
  --model.cfg_scale=0 \
  --model.class_dropout_prob=1 \
  --model.num_classes=1 \
  --model.denoise_timesteps=128 \
  --model.lr=2e-4 \
  --model.beta1=0.9 \
  --model.beta2=0.999 \
  --model.weight_decay=0.01 \
  --model.loss_post_weight=0.0 \
  --model.loss_var_weight=0.0 \
  --model.loss_align_weight=0.0
```

Inference/FID-only run from a checkpoint:

```bash
!uv run python train.py \
  --mode=inference \
  --dataset_name=celebahq256 \
  --tfds_data_dir=/kaggle/input/shortcut-celebahq256/tensorflow_datasets \
  --fid_stats=/kaggle/input/shortcut-celebahq256/data/celeba256_fidstats_ours.npz \
  --batch_size=64 \
  --load_dir=/kaggle/working/moe1_cond_source_k14_dn20_ckpt/<checkpoint_dir>/ \
  --save_dir=/kaggle/working/moe1_cond_source_k14_dn20_eval/ \
  --inference_timesteps=32 \
  --inference_generations=4096 \
  --inference_cfg_scale=0 \
  --model.hidden_size=768 \
  --model.patch_size=2 \
  --model.depth=12 \
  --model.num_heads=12 \
  --model.mlp_ratio=4 \
  --model.sharding=dp \
  --model.train_type=naive-moe-source \
  --model.source_variant=conditional-analytic \
  --model.gmm_stats_path=/kaggle/working/source_stats_k16_a4_eta05.npz \
  --model.gmm_num_modes=16 \
  --model.angular_num_submodes=4 \
  --model.local_eta=0.5 \
  --model.source_kappa=1.4 \
  --model.source_direction_noise=2.0 \
  --model.cfg_scale=0 \
  --model.class_dropout_prob=1 \
  --model.num_classes=1 \
  --model.denoise_timesteps=128
```

`git rev-parse --short HEAD` is optional. It only records the exact commit in
the notebook logs so a W&B run can be traced back to code.
