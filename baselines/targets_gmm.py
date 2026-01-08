import jax
import jax.numpy as jnp
import numpy as np


def get_targets(FLAGS, key, train_state, images, labels, gmm_stats, force_t=-1, force_dt=-1):
    """
    GMM-Prior Flow Matching Targets.
    Features:
    1. Hard Assignment using Log-Likelihood (Mahalanobis Distance).
    2. Importance Sampling Weights Calculation.
    3. Handle latent dataset mode (pre-computed x_0).

    Args:
        FLAGS: Config flags
        key: JAX random key
        train_state: Training state
        images: Input images [B, H, W, C]
        labels: Class labels [B]
        gmm_stats: Dict with keys 'means', 'covs', 'weights', 'empirical_probs'
        force_t: Override t value (-1 = random)
        force_dt: Override dt value (-1 = use default)

    Returns:
        x_t: Interpolated samples [B, H, W, C]
        v_t: Velocity targets [B, H, W, C]
        t: Time values [B]
        dt_base: dt values [B]
        labels_dropped: Labels with dropout [B]
        info: Dict with 'loss_weights', 'max_is_weight', 'dropped_ratio'
    """
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    B = images.shape[0]

    # === HANDLE LATENT DATASET MODE ===
    # If x_0 is pre-computed in dataset, skip GMM sampling
    if 'latent' in FLAGS.dataset_name:
        x_0 = images[..., :images.shape[-1] // 2]
        x_1 = images[..., images.shape[-1] // 2:]

        # Set dummy IS weights = 1 (no importance sampling for pre-computed data)
        info['loss_weights'] = jnp.ones(B)
        info['max_is_weight'] = 1.0

    # === GMM PRIOR MODE ===
    # === GMM PRIOR MODE ===
    else:
        x_1 = images

        # === 1. HARD ASSIGNMENT (Log-Likelihood) ===
        # Flatten images: [B, D]
        # Robustly calculate D - the dimension of the flattened image
        x_flat = x_1.reshape(B, -1)
        D = x_flat.shape[1] 

        # Flatten GMM params: [K, D]
        # Robustly reshape to (-1, D) to handle any prefix dimensions (like 1 from replication)
        means = gmm_stats['means'].reshape(-1, D)
        covs = gmm_stats['covs'].reshape(-1, D)
        weights = gmm_stats['weights'].flatten() # [K]
        
        # Recalculate K from the flattened weights
        K = weights.shape[0]

        # Tính Log-Likelihood Score:
        # Score_k = log(pi_k) - 0.5 * sum(log(sigma_k^2)) - 0.5 * sum((x - mu_k)^2 / sigma_k^2)

        # Term 1: Log Weights [K]
        log_weights = jnp.log(weights + 1e-10)

        # Term 2: Log Determinant [K] (Sum log variances for diagonal covariance)
        # Thêm epsilon nhỏ để tránh log(0)
        log_det = jnp.sum(jnp.log(covs + 1e-10), axis=-1)

        # Term 3: Mahalanobis Distance [B, K]
        # Broadcasting: [B, 1, D] - [1, K, D] -> [B, K, D]
        diff = x_flat[:, None, :] - means[None, :, :]
        # Chia cho variance: [B, K, D]
        mahalanobis = jnp.sum((diff ** 2) / (covs[None, :, :] + 1e-10), axis=-1)

        # Tổng hợp Score [B, K]
        log_probs = log_weights[None, :] - 0.5 * log_det[None, :] - 0.5 * mahalanobis

        # Hard Assignment: Chọn cụm k có score cao nhất
        cluster_ids = jnp.argmax(log_probs, axis=-1)  # [B]

        # === 2. IMPORTANCE SAMPLING WEIGHTS ===
        # Lấy Pi (Model Belief)
        model_pi = jnp.take(weights, cluster_ids)

        # Lấy P_emp (Data Reality) từ stats
        emp_prob = gmm_stats['empirical_probs'].flatten()
        emp_prob_selected = jnp.take(emp_prob, cluster_ids)

        # Tính Weight: w = Pi / P_emp
        # Nếu GMM gán trọng số cao (Pi) cho vùng ít dữ liệu (P_emp thấp) -> Weight lớn
        raw_weights = model_pi / (emp_prob_selected + 1e-8)

        # Normalize weights trong batch để giữ cho Loss scale ổn định (Mean ~ 1)
        # Điều này cực kỳ quan trọng để không làm Learning Rate bị sai lệch
        loss_weights = raw_weights / (jnp.mean(raw_weights) + 1e-8)

        # Lưu vào info để train.py sử dụng
        info['loss_weights'] = loss_weights
        # Log để theo dõi xem có weight nào quá lớn không (dấu hiệu GMM fit lỗi)
        info['max_is_weight'] = jnp.max(loss_weights)

        # === 3. SAMPLE SOURCE FROM GMM ===
        # Lấy params của cụm được chọn (đã reshape flat [K, D])
        batch_means_flat = jnp.take(means, cluster_ids, axis=0)  # [B, D]
        batch_vars_flat = jnp.take(covs, cluster_ids, axis=0)   # [B, D]
        batch_stds_flat = jnp.sqrt(batch_vars_flat)

        # Sample x0 ~ N(mu_k, sigma_k)
        eps_flat = jax.random.normal(noise_key, x_flat.shape)
        x_0_flat = batch_means_flat + batch_stds_flat * eps_flat

        # Reshape x0 về không gian ảnh [B, H, W, C] để flow matching
        x_0 = x_0_flat.reshape(images.shape)

    # === 4. STANDARD FLOW MATCHING INTERPOLATION ===
    # Sample t
    t = jax.random.randint(time_key, (B,), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']

    # Override t if force_t is specified
    if force_t != -1:
        force_t_vec = jnp.ones(B, dtype=jnp.float32) * force_t
        t = jnp.where(force_t_vec != -1, force_t_vec, t)

    t_full = t[:, None, None, None]

    # Path: x_t = (1-(1-eps)*t)*x_0 + t*x_1
    # Use same epsilon as naive flow matching
    eps_flow = 1e-5
    x_t = (1 - (1 - eps_flow) * t_full) * x_0 + t_full * x_1

    # Velocity: v_t = x_1 - (1-eps)*x_0
    v_t = x_1 - (1 - eps_flow) * x_0

    # === 5. LABEL DROPOUT FOR CLASSIFIER-FREE GUIDANCE ===
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # === 6. DT_BASE (for compatibility with shortcut models) ===
    # For naive/gmm-prior mode, dt_base is always maximum (log2(denoise_timesteps))
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(B, dtype=jnp.int32) * dt_flow

    return x_t, v_t, t, dt_base, labels_dropped, info
