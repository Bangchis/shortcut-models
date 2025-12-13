# targets_khoat_fm.py
import jax
import jax.numpy as jnp
import numpy as np


def _log2_int(n: int) -> int:
    k = int(np.log2(int(n)))
    if (1 << k) != int(n):
        raise ValueError(f"denoise_timesteps must be a power of two, got {n}")
    return k


def sample_dt_base_curriculum(dt_key, B, K, stage, rho):
    """
    Sample dt_base with curriculum learning (mass transfer).

    Args:
        dt_key: JAX random key
        B: batch size
        K: maximum dt_base (log2(denoise_timesteps))
        stage: current curriculum stage (0 to K)
        rho: mass transfer parameter (0 to 1)

    Returns:
        dt_base: sampled dt values [B,] in range [K-stage, K]

    Strategy:
        - Allowed set S = {K, K-1, ..., K-stage}
        - p(K) = 1 - rho (most mass at K initially)
        - p(others in S) = rho / stage (distributed uniformly)
    """
    # Minimum allowed dt_base
    allowed_min = K - stage

    # Create probability distribution over all possible dt values [0..K]
    idx = jnp.arange(K + 1, dtype=jnp.int32)

    # Mask for "other dts" in allowed range: [allowed_min .. K-1]
    mask_other = (idx >= allowed_min) & (idx <= (K - 1))

    # Avoid division by zero when stage=0
    stage_f = jnp.maximum(stage.astype(jnp.float32), 1.0)
    rho = jnp.where(stage == 0, 0.0, rho)

    # Build probability vector
    p = jnp.zeros((K + 1,), dtype=jnp.float32)
    p = p.at[K].set(1.0 - rho)  # Mass at K
    p = p + mask_other.astype(jnp.float32) * (rho / stage_f)  # Distribute rho to others

    # Convert to logits for categorical sampling (avoid log(0))
    logits = jnp.where(p > 0, jnp.log(p), -1e9)

    # Sample B independent samples
    dt_base = jax.random.categorical(dt_key, logits, shape=(B,))

    return dt_base.astype(jnp.int32)


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1,
                kfm_stage=jnp.array(0, jnp.int32), kfm_rho=jnp.array(0.0, jnp.float32)):
    """
    Khoat Flow Matching (training phase) with curriculum learning:
      - sample dt_base with curriculum (mass transfer strategy)
      - sample aligned t on grid: t = m / 2^{dt_base}
      - x_t = (1 - (1-eps)*t)*x0 + t*x1
      - v_t = x1 - (1-eps)*x0
      - return (x_t, v_t, t, dt_base, labels_dropped, info)
    """
    # RNG
    dt_key, t_key, x0_key, label_key = jax.random.split(key, 4)
    info = {}

    B = images.shape[0]
    N = int(FLAGS.model['denoise_timesteps'])
    K = _log2_int(N)

    eps = float(FLAGS.model.get('kfm_eps', 1e-5))

    # ===== 1) Sample dt_base with curriculum learning =====
    dt_base = sample_dt_base_curriculum(dt_key, B, K, kfm_stage, kfm_rho)

    # force_dt override (used in eval sweeps)
    force_dt_i = jnp.asarray(force_dt, dtype=jnp.int32)
    dt_base = jnp.where(force_dt_i != -1, jnp.full((B,), force_dt_i, dtype=jnp.int32), dt_base)

    # ===== 2) Sample aligned t =====
    # t in {0, 1/2^k, ..., (2^k-1)/2^k} where k = dt_base
    dt_sections = jnp.power(2, dt_base).astype(jnp.float32)  # (B,)
    u = jax.random.uniform(t_key, (B,), minval=0.0, maxval=1.0)  # safe per-element
    m = jnp.floor(u * dt_sections).astype(jnp.int32)
    t = m.astype(jnp.float32) / dt_sections

    # force_t override
    force_t_f = jnp.asarray(force_t, dtype=jnp.float32)
    t = jnp.where(force_t_f != -1.0, jnp.full((B,), force_t_f, dtype=jnp.float32), t)
    t_full = t[:, None, None, None]

    # ===== 3) Flow-matching construction =====
    x0 = jax.random.normal(x0_key, images.shape)
    x1 = images
    x_t = (1.0 - (1.0 - eps) * t_full) * x0 + t_full * x1
    v_t = x1 - (1.0 - eps) * x0

    # ===== 4) CFG label dropout (reuse shortcut behavior) =====
    drop_p = float(FLAGS.model['class_dropout_prob'])
    labels_dropout = jax.random.bernoulli(label_key, drop_p, (B,))
    labels_dropped = jnp.where(labels_dropout, jnp.int32(FLAGS.model['num_classes']), labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == jnp.int32(FLAGS.model['num_classes']))

    # ===== 5) useful logs / sanity + curriculum monitoring =====
    info['dt_base_mean'] = jnp.mean(dt_base.astype(jnp.float32))
    info['dt_base_std'] = jnp.std(dt_base.astype(jnp.float32))
    info['dt_base_max_ratio'] = jnp.mean(dt_base == K)  # Fraction at maximum dt
    k_grid = t * jnp.power(2.0, dt_base.astype(jnp.float32))
    info['grid_abs_err'] = jnp.mean(jnp.abs(k_grid - jnp.round(k_grid)))

    # Curriculum tracking
    info['kfm_stage'] = kfm_stage.astype(jnp.float32)
    info['kfm_rho'] = kfm_rho

    return x_t, v_t, t, dt_base, labels_dropped, info
