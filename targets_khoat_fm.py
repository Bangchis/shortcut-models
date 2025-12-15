# targets_khoat_fm.py
import jax
import jax.numpy as jnp
import numpy as np


def _log2_int(n: int) -> int:
    k = int(np.log2(int(n)))
    if (1 << k) != int(n):
        raise ValueError(f"denoise_timesteps must be a power of two, got {n}")
    return k


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    """
    Khoat Flow Matching (training phase):
      - sample dt_base -> d = 2^{-dt_base} with P_min selecting d_min
      - sample aligned t on grid: t = m / 2^{dt_base}
      - stratified t=0: kfm_t0_ratio of batch forced to t=0
      - x_t = (1 - (1-eps)*t)*x0 + t*x1
      - v_t: t=0 -> (1/α)x1 + ((α-d)/(α·d))x0
             t>0 -> (1/α)(x1 - (1-eps)*x0)
      - return (x_t, v_t, t, dt_base, labels_dropped, info)
    """
    # RNG

    dt_key, t_key, x0_key, label_key, cat_key = jax.random.split(key, 5)
    info = {}

    B = images.shape[0]
    N = int(FLAGS.model['denoise_timesteps'])
    K = _log2_int(N)

    # Defaults (you asked: default only)
    dt_min = int(FLAGS.model.get('kfm_dt_min_exp', 0))
    dt_max_cfg = int(FLAGS.model.get('kfm_dt_max_exp', -1))
    dt_max = K if dt_max_cfg == -1 else int(dt_max_cfg)
    dt_min = max(0, min(dt_min, dt_max))
    dt_max = max(dt_min, min(dt_max, K))

    P_min = float(FLAGS.model.get('kfm_p_min', 0.5))
    eps = float(FLAGS.model.get('kfm_eps', 1e-5))

    # ===== 1) Sample dt_base (Linear Distribution) =====
    # 50% (P_min): dt_base = dt_max (d_min = 1/128)
    # 50% remaining: Linear distribution favoring smaller steps
    # P(dt_base=k) ∝ weight(k) = k - dt_min + 1
    choose_min = jax.random.bernoulli(dt_key, p=P_min, shape=(B,))
    dt_base = jnp.full((B,), dt_max, dtype=jnp.int32)

    if dt_max > dt_min:
        # Number of other levels: [dt_min, dt_min+1, ..., dt_max-1]
        num_others = dt_max - dt_min
        other_values = jnp.arange(dt_min, dt_max, dtype=jnp.int32)

        # Linear weights: dt_base small (d large) -> low weight
        #                 dt_base large (d small) -> high weight
        # weights = [1, 2, 3, ..., num_others]
        weights = jnp.arange(1, num_others + 1, dtype=jnp.float32)
        logits = jnp.log(weights)

        # Sample indices [0, 1, ..., num_others-1]
        sampled_indices = jax.random.categorical(cat_key, logits, shape=(B,))
        sampled_others = other_values[sampled_indices]

        # Merge: if choose_min, use dt_max; else use sampled value
        dt_base = jnp.where(choose_min, dt_base, sampled_others)

    # force_dt override (used in eval sweeps)
    force_dt_i = jnp.asarray(force_dt, dtype=jnp.int32)
    dt_base = jnp.where(force_dt_i != -1, jnp.full((B,),
                        force_dt_i, dtype=jnp.int32), dt_base)

    # ===== 2) Sample aligned t with stratified t=0 =====
    # t in {0, 1/2^k, ..., (2^k-1)/2^k} where k = dt_base
    t0_ratio = float(FLAGS.model.get('kfm_t0_ratio', 0.125))

    # Split keys for stratification
    perm_key, grid_key = jax.random.split(t_key)

    # Sample grid-aligned t for all samples
    dt_sections = jnp.power(2, dt_base).astype(jnp.float32)  # (B,)
    u = jax.random.uniform(grid_key, (B,), minval=0.0, maxval=1.0)
    m = jnp.floor(u * dt_sections).astype(jnp.int32)
    t = m.astype(jnp.float32) / dt_sections

    # Stratified t=0: force n_t0 samples to t=0
    n_t0 = int(B * t0_ratio)
    indices = jax.random.permutation(perm_key, B)
    mask_t0 = indices < n_t0  # If n_t0=0, all False
    t = jnp.where(mask_t0, 0.0, t)

    # force_t override
    force_t_f = jnp.asarray(force_t, dtype=jnp.float32)
    t = jnp.where(force_t_f != -1.0, jnp.full((B,),
                  force_t_f, dtype=jnp.float32), t)
    t_full = t[:, None, None, None]

    # ===== 3) Flow-matching construction =====
    x0 = jax.random.normal(x0_key, images.shape)
    x1 = images
    x_t = (1.0 - (1.0 - eps) * t_full) * x0 + t_full * x1

    # v_target with t=0 special case
    alpha = float(FLAGS.model.get('kfm_alpha', 0.9))
    d = jnp.power(2.0, -dt_base.astype(jnp.float32)
                  )[:, None, None, None]  # (B,1,1,1)

    # Velocity formulas (different for t=0 and t>0)
    # For t>0: v_t = (x1 - (1-eps)*x0) / alpha
    # For t=0: v_t = (alpha - 1)/alpha * (1-eps)*x0 + (1/alpha) * x1
    v_t_regular = (1.0/alpha) * (x1 - (1.0 - eps) * x0)

    # Special formula for t=0
    v_t_at_zero = ((alpha - 1.0) / alpha) * \
        (1.0 - eps) * x0 + (1.0 / alpha) * x1

    # Use different formulas based on whether t=0 or t>0
    v_t = jnp.where(t_full < 1e-6, v_t_at_zero, v_t_regular)

    # ===== 4) CFG label dropout (reuse shortcut behavior) =====
    drop_p = float(FLAGS.model['class_dropout_prob'])
    labels_dropout = jax.random.bernoulli(label_key, drop_p, (B,))
    labels_dropped = jnp.where(labels_dropout, jnp.int32(
        FLAGS.model['num_classes']), labels)
    info['dropped_ratio'] = jnp.mean(
        labels_dropped == jnp.int32(FLAGS.model['num_classes']))

    # ===== 5) useful logs / sanity =====
    info['dt_base_mean'] = jnp.mean(dt_base.astype(jnp.float32))
    info['pmin_hit_ratio'] = jnp.mean(dt_base == jnp.int32(dt_max))  # Should ≈ 0.5
    # Log ratio of largest step (dt_base=0, d=1) if applicable
    if dt_min == 0:
        info['dt_max_step_ratio'] = jnp.mean(dt_base == 0)  # Largest step d=1
    info['t0_ratio_actual'] = jnp.mean(t < 1e-6)
    k_grid = t * jnp.power(2.0, dt_base.astype(jnp.float32))
    info['grid_abs_err'] = jnp.mean(jnp.abs(k_grid - jnp.round(k_grid)))

    return x_t, v_t, t, dt_base, labels_dropped, info
