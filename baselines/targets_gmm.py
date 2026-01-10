import jax
import jax.numpy as jnp
import numpy as np
from utils.dct_utils import dct_reduce, idct_power_law


def get_targets(FLAGS, key, train_state, images, labels, gmm_stats, force_t=-1, force_dt=-1):
    """
    GMM-Prior Flow Matching Targets.
    Features:
    1. Hard Assignment using Log-Likelihood (Mahalanobis Distance).
    2. Importance Sampling Weights Calculation.
    3. Handle latent dataset mode (pre-computed x_0).
    4. JIT-safe (no python control flow on tracers).
    """
    label_key, time_key, noise_key, dct_noise_key = jax.random.split(key, 4)
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
    else:
        x_1 = images  # [B, 32, 32, 4]

        # === 1. COMPRESS INPUT WITH DCT ===
        # Nén ảnh input để tìm cụm trong không gian nén
        keep_size = FLAGS.model['dct_keep_size']
        z_flat = dct_reduce(x_1, keep_size=keep_size)  # [B, keep_size^2*4]
        D = z_flat.shape[1]  # Should be keep_size^2*4

        # === 2. GMM ASSIGNMENT (on DCT-compressed space) ===
        # GMM params: [K, D] where D = keep_size^2*4
        means = gmm_stats['means'].reshape(-1, D)  # [K, D]
        covs = gmm_stats['covs'].reshape(-1, D)    # [K, D]
        weights = gmm_stats['weights'].flatten()    # [K]

        K = weights.shape[0]

        # Tính Log-Likelihood Score (Mahalanobis Distance)
        log_weights = jnp.log(weights + 1e-10)
        log_det = jnp.sum(jnp.log(covs + 1e-10), axis=-1)

        # Mahalanobis Distance [B, K]
        diff = z_flat[:, None, :] - means[None, :, :]  # [B, K, 256]
        mahalanobis = jnp.sum((diff ** 2) / (covs[None, :, :] + 1e-10), axis=-1)

        # Log Probs [B, K]
        log_probs = log_weights[None, :] - 0.5 * log_det[None, :] - 0.5 * mahalanobis

        # Hard Assignment
        cluster_ids = jnp.argmax(log_probs, axis=-1)  # [B]

        # === 3. IMPORTANCE SAMPLING WEIGHTS ===
        model_pi = jnp.take(weights, cluster_ids)
        emp_prob = gmm_stats['empirical_probs'].flatten()
        emp_prob_selected = jnp.take(emp_prob, cluster_ids)

        raw_weights = model_pi / (emp_prob_selected + 1e-8)
        loss_weights = raw_weights / (jnp.mean(raw_weights) + 1e-8)

        info['loss_weights'] = loss_weights
        info['max_is_weight'] = jnp.max(loss_weights)

        # === 4. SAMPLE & RECONSTRUCT WITH POWER LAW ===
        # Sample trong không gian nén [B, D]
        batch_means = jnp.take(means, cluster_ids, axis=0)  # [B, D]
        batch_stds = jnp.sqrt(jnp.take(covs, cluster_ids, axis=0))  # [B, D]

        z_0 = batch_means + batch_stds * jax.random.normal(noise_key, z_flat.shape)

        # Khôi phục x_0 bằng Power Law Noise (Pink Noise)
        x_0 = idct_power_law(dct_noise_key, z_0, keep_size=keep_size, alpha=1.0, noise_scale=1.0)  # [B, 32, 32, 4]

    # === 4. STANDARD FLOW MATCHING INTERPOLATION ===
    # Sample t
    t = jax.random.randint(time_key, (B,), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']

    # Override t if force_t is specified (using jnp.where for JIT compat)
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
