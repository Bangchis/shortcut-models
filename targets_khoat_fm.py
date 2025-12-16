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

    # Defaults
    dt_min = int(FLAGS.model.get('kfm_dt_min_exp', 0))
    dt_max_cfg = int(FLAGS.model.get('kfm_dt_max_exp', -1))
    dt_max = K if dt_max_cfg == -1 else int(dt_max_cfg)
    dt_min = max(0, min(dt_min, dt_max))
    dt_max = max(dt_min, min(dt_max, K))

    P_min = float(FLAGS.model.get('kfm_p_min', 0.5))
    eps = float(FLAGS.model.get('kfm_eps', 1e-5))
    
    # Bật flag này trong config để dùng logic chia slot cố định
    deterministic = bool(FLAGS.model.get('kfm_deterministic_for_d', 0))

    # ===== 1) Sample dt_base =====
    if deterministic:
        # --- DETERMINISTIC MODE: Fixed Slot Allocation ---
        
        # 1. Chia phe 50/50 chuẩn xác (Integer division)
        n_flow = B // 2
        n_shortcut = B - n_flow

        # 2. Tạo slot cho Flow Matching (d_min)
        dt_flow = jnp.full((n_flow,), dt_max, dtype=jnp.int32)

        if dt_max > dt_min:
            # 3. Tạo slot cho Shortcut (Custom Distribution)
            shortcut_bases = jnp.arange(dt_min, dt_max, dtype=jnp.int32)
            num_levels = shortcut_bases.shape[0]

            # --- CUSTOM WEIGHTS: [3, 4, 4, 5, 5, 5, 6] ---
            # Weights này ưu tiên bước nhỏ (6) nhưng giữ đủ bước lớn (3)
            custom_weights = jnp.array([3., 4., 4., 5., 5., 5., 6.], dtype=jnp.float32)

            # Fallback: Nếu số level không phải 7 (ví dụ đổi timesteps), dùng Linear
            weights = jnp.where(num_levels == 7,
                                custom_weights,
                                jnp.arange(1, num_levels + 1, dtype=jnp.float32))

            total_weight = jnp.sum(weights)

            # Tính số lượng slot cơ bản (Floor)
            base_counts = jnp.floor((weights / total_weight) * n_shortcut).astype(jnp.int32)
            base_counts = jnp.maximum(base_counts, 1)  # Safety: Tối thiểu 1

            # Xử lý phần dư: Cộng vào đầu mảng (Bước lớn)
            current_sum = jnp.sum(base_counts)
            diff = n_shortcut - current_sum
            
            # Tạo mask cộng vào các index đầu tiên [0, 1, 2...] nếu thừa
            indices = jnp.arange(num_levels)
            extra_mask = (indices < diff).astype(jnp.int32)
            
            final_counts = base_counts + extra_mask

            # Tạo mảng dt_shortcut thông qua repeat
            dt_shortcut = jnp.repeat(shortcut_bases, final_counts, total_repeat_length=n_shortcut)

            # Ghép lại
            dt_base_combined = jnp.concatenate([dt_flow, dt_shortcut])
        else:
            # Trường hợp dt_min == dt_max (hiếm gặp)
            dt_shortcut = jnp.full((n_shortcut,), dt_min, dtype=jnp.int32)
            dt_base_combined = jnp.concatenate([dt_flow, dt_shortcut])

        # 4. Xáo trộn ngẫu nhiên vị trí
        perm = jax.random.permutation(dt_key, B)
        dt_base = dt_base_combined[perm]

    else:
        # --- STOCHASTIC MODE (Original logic) ---
        # (Giữ lại làm fallback hoặc so sánh)
        choose_min = jax.random.bernoulli(dt_key, p=P_min, shape=(B,))
        dt_base = jnp.full((B,), dt_max, dtype=jnp.int32)

        if dt_max > dt_min:
            num_others = dt_max - dt_min
            other_values = jnp.arange(dt_min, dt_max, dtype=jnp.int32)
            weights = jnp.arange(1, num_others + 1, dtype=jnp.float32)
            logits = jnp.log(weights)
            sampled_indices = jax.random.categorical(cat_key, logits, shape=(B,))
            sampled_others = other_values[sampled_indices]
            dt_base = jnp.where(choose_min, dt_base, sampled_others)

    # force_dt override
    force_dt_i = jnp.asarray(force_dt, dtype=jnp.int32)
    dt_base = jnp.where(force_dt_i != -1, jnp.full((B,),
                        force_dt_i, dtype=jnp.int32), dt_base)

    # ===== 2) Sample aligned t with stratified t=0 =====
    t0_ratio = float(FLAGS.model.get('kfm_t0_ratio', 0.125))
    perm_key, grid_key = jax.random.split(t_key)

    dt_sections = jnp.power(2, dt_base).astype(jnp.float32)
    u = jax.random.uniform(grid_key, (B,), minval=0.0, maxval=1.0)
    m = jnp.floor(u * dt_sections).astype(jnp.int32)
    t = m.astype(jnp.float32) / dt_sections

    n_t0 = int(B * t0_ratio)
    indices = jax.random.permutation(perm_key, B)
    mask_t0 = indices < n_t0
    t = jnp.where(mask_t0, 0.0, t)

    force_t_f = jnp.asarray(force_t, dtype=jnp.float32)
    t = jnp.where(force_t_f != -1.0, jnp.full((B,),
                  force_t_f, dtype=jnp.float32), t)
    t_full = t[:, None, None, None]

    # ===== 3) Flow-matching construction =====
    x0 = jax.random.normal(x0_key, images.shape)
    x1 = images
    x_t = (1.0 - (1.0 - eps) * t_full) * x0 + t_full * x1

    # v_target
    alpha = float(FLAGS.model.get('kfm_alpha', 0.9))
    d = jnp.power(2.0, -dt_base.astype(jnp.float32))[:, None, None, None]

    v_t_regular = (1.0/alpha) * (x1 - (1.0 - eps) * x0)
    v_t_at_zero = ((alpha - 1.0) / alpha) * (1.0 - eps) * x0 + (1.0 / alpha) * x1
    v_t = jnp.where(t_full < 1e-6, v_t_at_zero, v_t_regular)

    # ===== 4) CFG label dropout =====
    drop_p = float(FLAGS.model['class_dropout_prob'])
    labels_dropout = jax.random.bernoulli(label_key, drop_p, (B,))
    labels_dropped = jnp.where(labels_dropout, jnp.int32(
        FLAGS.model['num_classes']), labels)
    info['dropped_ratio'] = jnp.mean(
        labels_dropped == jnp.int32(FLAGS.model['num_classes']))

    # ===== 5) Logs =====
    info['sampling_mode'] = 1.0 if deterministic else 0.0
    info['dt_base_mean'] = jnp.mean(dt_base.astype(jnp.float32))
    info['pmin_hit_ratio'] = jnp.mean(dt_base == jnp.int32(dt_max))
    
    # Log check số lượng d=1 (index 0) và d=1/64 (index 6)
    if dt_min == 0:
        info['count_dt_1'] = jnp.sum(dt_base == 0)      # Kỳ vọng: 3 (với Batch 64)
        info['count_dt_1_64'] = jnp.sum(dt_base == 6)   # Kỳ vọng: 6 (với Batch 64)

    return x_t, v_t, t, dt_base, labels_dropped, info