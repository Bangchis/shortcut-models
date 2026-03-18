###############################
#
#  Projected Diagonal GMM utilities for source prior.
#  All computations in float32 for numerical stability.
#
###############################

import jax
import jax.numpy as jnp
import numpy as np


def flatten_latent(x):
    """Reshape spatial latent to flat vector.
    [B, H, W, C] -> [B, D] or [B, M, H, W, C] -> [B, M, D].
    """
    if x.ndim == 4:
        B = x.shape[0]
        return x.reshape(B, -1)
    elif x.ndim == 5:
        B, M = x.shape[0], x.shape[1]
        return x.reshape(B, M, -1)
    else:
        raise ValueError(f"flatten_latent: unexpected ndim={x.ndim}, shape={x.shape}")


def unflatten_latent(z, spatial_shape):
    """Reshape flat vector back to spatial latent.
    [B, D] -> [B, H, W, C] or [B, M, D] -> [B, M, H, W, C].
    spatial_shape: (H, W, C).
    """
    H, W, C = spatial_shape
    if z.ndim == 2:
        B = z.shape[0]
        return z.reshape(B, H, W, C)
    elif z.ndim == 3:
        B, M = z.shape[0], z.shape[1]
        return z.reshape(B, M, H, W, C)
    else:
        raise ValueError(f"unflatten_latent: unexpected ndim={z.ndim}, shape={z.shape}")


def safe_project(y, eps_proj=1e-6):
    """Project vectors onto approximate unit sphere.
    y: [..., D] -> [..., D].
    Returns (projected, denom_min) where denom_min is min denominator for debugging.
    """
    y = y.astype(jnp.float32)
    sq_norm = jnp.sum(y ** 2, axis=-1, keepdims=True)  # [..., 1]
    denom = jnp.sqrt(sq_norm + eps_proj ** 2)  # [..., 1]
    projected = y / denom
    denom_min = jnp.min(denom)
    return projected, denom_min


def sigma_from_raw(r_raw):
    """Convert unconstrained r_raw to positive sigma via softplus.
    r_raw: [K, D] -> [K, D].
    """
    return jax.nn.softplus(r_raw.astype(jnp.float32))


def diag_gaussian_logprob(x_flat, mu, r_raw, eps_cov=1e-6):
    """Log-probability of x under diagonal Gaussian components.
    x_flat: [B, D], mu: [K, D], r_raw: [K, D].
    Returns: logprob [B, K].
    """
    x_flat = x_flat.astype(jnp.float32)
    mu = mu.astype(jnp.float32)

    sigma = sigma_from_raw(r_raw)  # [K, D]
    var = sigma ** 2 + eps_cov  # [K, D]
    log_var = jnp.clip(jnp.log(var), -20.0, 20.0)  # Prevent overflow

    # Broadcast: x_flat [B, 1, D] - mu [1, K, D] -> [B, K, D]
    diff = x_flat[:, None, :] - mu[None, :, :]
    mahal = diff ** 2 / var[None, :, :]  # [B, K, D]

    # sum over D
    log_norm = -0.5 * jnp.sum(log_var + jnp.log(2 * jnp.pi), axis=-1)  # [K]
    log_exp = -0.5 * jnp.sum(mahal, axis=-1)  # [B, K]
    logprob = log_norm[None, :] + log_exp  # [B, K]
    return logprob


def compute_router_posterior(x1_flat, prior_params, eps_cov=1e-6):
    """Compute router posterior q(k|x_1) using full GMM density.
    x1_flat: [B, D].
    prior_params: dict with pi_logits [K], mu [K,D], r_raw [K,D].
    Returns: q_full [B,K], log_mixprob [B], stats dict.
    """
    pi_logits = prior_params['pi_logits'].astype(jnp.float32)
    mu = prior_params['mu']
    r_raw = prior_params['r_raw']

    log_pi = jax.nn.log_softmax(pi_logits)  # [K]
    log_comp = diag_gaussian_logprob(x1_flat, mu, r_raw, eps_cov)  # [B, K]

    log_joint = log_pi[None, :] + log_comp  # [B, K]
    log_mixprob = jax.nn.logsumexp(log_joint, axis=-1)  # [B]
    q_full = jax.nn.softmax(log_joint, axis=-1)  # [B, K]

    # Debug stats
    sigma = sigma_from_raw(r_raw)
    pi = jax.nn.softmax(pi_logits)
    entropy = -jnp.sum(q_full * jnp.log(q_full + 1e-10), axis=-1)  # [B]
    stats = {
        'router_entropy': jnp.mean(entropy),
        'router_qmax_mean': jnp.mean(jnp.max(q_full, axis=-1)),
        'router_pi_min': jnp.min(pi),
        'router_pi_max': jnp.max(pi),
        'router_sigma_mean': jnp.mean(sigma),
        'router_sigma_min': jnp.min(sigma),
        'router_sigma_max': jnp.max(sigma),
        'has_nan_router': jnp.any(jnp.isnan(q_full)).astype(jnp.float32),
    }
    return q_full, log_mixprob, stats


def select_top_m(q_full, top_m):
    """Select top-M modes per sample and renormalize.
    q_full: [B, K].
    Returns: top_idx [B, M], q_top [B, M] (renormalized).
    """
    top_vals, top_idx = jax.lax.top_k(q_full, top_m)  # [B, M] each
    q_top = top_vals / jnp.sum(top_vals, axis=-1, keepdims=True)  # [B, M]
    return top_idx, q_top


def sample_projected_sources(key, prior_params, top_idx, eps_proj=1e-6):
    """Sample from selected GMM components and project.
    key: PRNG key.
    prior_params: dict with mu [K,D], r_raw [K,D].
    top_idx: [B, M].
    Returns: x0_dir_bmd [B, M, D], stats dict.
    """
    mu = prior_params['mu'].astype(jnp.float32)  # [K, D]
    r_raw = prior_params['r_raw']
    sigma = sigma_from_raw(r_raw)  # [K, D]

    # Gather selected components: [K, D] -> [B, M, D]
    mu_sel = mu[top_idx]      # [B, M, D]
    sigma_sel = sigma[top_idx]  # [B, M, D]

    B, M, D = mu_sel.shape
    eps = jax.random.normal(key, (B, M, D), dtype=jnp.float32)

    # Reparameterization trick
    y = mu_sel + sigma_sel * eps  # [B, M, D]

    # Project onto sphere
    x0_dir, proj_denom_min = safe_project(y, eps_proj)

    stats = {
        'source_proj_denom_min': proj_denom_min,
        'has_nan_source': jnp.any(jnp.isnan(x0_dir)).astype(jnp.float32),
    }
    return x0_dir, stats


def apply_sample_radius(x0_dir_bmd, r_scalar_b):
    """Scale projected source by per-sample radius with stop_gradient.
    x0_dir_bmd: [B, M, D].
    r_scalar_b: [B].
    Returns: x0_flat_bmd [B, M, D].
    """
    r = jax.lax.stop_gradient(r_scalar_b)  # [B]
    return x0_dir_bmd * r[:, None, None]


def build_sparse_router_cond(top_idx, q_top, num_modes):
    """Build sparse router conditioning vector.
    top_idx: [B, M], q_top: [B, M].
    Returns: [B, num_modes] with q_top values scattered at top_idx positions.
    """
    B = top_idx.shape[0]
    cond = jnp.zeros((B, num_modes), dtype=jnp.float32)
    batch_idx = jnp.arange(B)[:, None]  # [B, 1]
    cond = cond.at[batch_idx, top_idx].set(q_top)
    return cond


def compute_mix_loss(log_mixprob):
    """Negative log-likelihood of data under mixture.
    log_mixprob: [B].
    Returns: scalar.
    """
    return -jnp.mean(log_mixprob)


def compute_bal_loss(q_full):
    """Balance loss to encourage uniform mode usage.
    q_full: [B, K].
    Returns: scalar.
    """
    K = q_full.shape[-1]
    mean_q = jnp.mean(q_full, axis=0)  # [K]
    target = 1.0 / K
    return jnp.sum((mean_q - target) ** 2)


def compute_var_reg_loss(r_raw, eps_cov=1e-6):
    """Variance regularization toward unit covariance.
    Uses KL[N(mu, diag(var)) || N(mu, I)] per-dimension, averaged over [K, D]:
        0.5 * (var - 1 - log(var))
    This is zero at var=1 and positive otherwise.
    """
    sigma = sigma_from_raw(r_raw)  # [K, D]
    var = sigma ** 2 + eps_cov     # [K, D]
    log_var = jnp.clip(jnp.log(var), -20.0, 20.0)
    kl_dim = 0.5 * (var - 1.0 - log_var)
    return jnp.mean(kl_dim)
