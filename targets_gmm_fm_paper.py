"""
GMM-FM Paper targets using pre-assigned clusters.

This is the paper-correct implementation that uses cached latents and
pre-assigned cluster IDs, eliminating the need for posterior computation
during training.

Key differences from targets_gmm_fm.py:
- x1 is already encoded (cached latents, not pixels)
- k_vec is pre-assigned (hard partition, not posterior sampling)
- x0 sampled directly from k_vec (NO p(k|x1) computation!)
"""

import jax
import jax.numpy as jnp
import numpy as np

from utils.gmm_prior import GMMPrior


def get_targets(FLAGS, key, train_state, images_latent, k_vec, gmm_prior, force_t=-1, force_dt=-1):
    """
    GMM-FM paper targets using pre-assigned clusters.

    Args:
        FLAGS: Configuration flags
        key: JAX random key
        train_state: Training state (unused, kept for API compatibility)
        images_latent: (B, H, W, C) - already encoded latents (x1)
        k_vec: (B,) int32 - pre-assigned cluster IDs
        gmm_prior: GMMPrior object
        force_t: Force specific t value (for eval)
        force_dt: Force specific dt value (for eval)

    Returns:
        x_t: (B, H, W, C) - interpolated latents
        v_t: (B, H, W, C) - target velocity
        t: (B,) - timesteps
        dt_base: (B,) - dt values
        labels: (B,) - unconditional labels (cluster IDs passed separately via k_vec)
        info: dict - logging info
    """
    time_key, noise_key = jax.random.split(key, 2)
    info = {}

    # x1 is already in latent space (cached)
    x1 = images_latent
    B = x1.shape[0]

    # Sample x0 from pre-assigned clusters (NO posterior computation!)
    # This is the key difference from targets_gmm_fm.py
    flat_shape = (B, -1)
    x0_flat = sample_x0_from_cluster_ids(gmm_prior, noise_key, k_vec)
    x0 = x0_flat.reshape(x1.shape)

    # Sample t (same as naive FM)
    t = jax.random.randint(
        time_key, (B,),
        minval=0, maxval=FLAGS.model['denoise_timesteps']
    ).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']

    # Force t override (for eval)
    force_t_vec = jnp.ones(B, dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec >= 0, force_t_vec, t)
    t_full = t.reshape((-1, 1, 1, 1))

    # Standard flow matching interpolation
    eps = 1e-5
    x_t = (1 - (1 - eps) * t_full) * x0 + t_full * x1
    v_t = x1 - (1 - eps) * x0

    # dt_base (same as naive)
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(B, dtype=jnp.int32) * dt_flow

    # Force dt override (for eval)
    force_dt_i = jnp.asarray(force_dt, dtype=jnp.int32)
    dt_base = jnp.where(
        force_dt_i != -1,
        jnp.full((B,), force_dt_i, dtype=jnp.int32),
        dt_base
    )

    # Return unconditional labels (cluster IDs passed separately as k parameter to model)
    labels = jnp.ones(B, dtype=jnp.int32) * FLAGS.model['num_classes']  # Unconditional token

    # Logging: cluster assignment stats
    K = int(gmm_prior.pi.shape[0])
    counts = jnp.bincount(k_vec, length=K)
    total = jnp.sum(counts)

    # Entropy of cluster distribution in batch
    p_batch = counts / (total + 1e-8)
    entropy = -jnp.sum(p_batch * jnp.log(p_batch + 1e-8))

    info['gmm/assign_entropy'] = entropy
    info['gmm/assign_max_frac'] = jnp.max(counts) / (total + 1e-8)
    info['gmm/num_unique_clusters'] = jnp.sum(counts > 0)

    # Velocity magnitude (for debugging)
    info['v_magnitude_target'] = jnp.sqrt(jnp.mean(jnp.square(v_t)))
    info['x0_magnitude'] = jnp.sqrt(jnp.mean(jnp.square(x0)))
    info['x1_magnitude'] = jnp.sqrt(jnp.mean(jnp.square(x1)))

    return x_t, v_t, t, dt_base, labels, info


@jax.jit
def sample_x0_from_cluster_ids(
    gmm_prior: GMMPrior,
    key: jax.Array,
    k_vec: jnp.ndarray  # (B,) int32
) -> jnp.ndarray:
    """
    Sample x0 from specific cluster IDs.

    NO posterior computation - just gather mu[k], var[k] and sample.

    Args:
        gmm_prior: GMMPrior object
        key: JAX random key
        k_vec: (B,) int32 - cluster IDs

    Returns:
        x0: (B, D) - sampled noise from clusters
    """
    # Gather cluster parameters
    mu_k = jnp.take(gmm_prior.mu, k_vec, axis=0)    # (B, D)
    var_k = jnp.take(gmm_prior.var, k_vec, axis=0)  # (B, D)

    # Sample from N(mu_k, var_k)
    eps = jax.random.normal(key, mu_k.shape, dtype=mu_k.dtype)
    x0 = mu_k + jnp.sqrt(jnp.maximum(var_k, 1e-12)) * eps

    return x0
