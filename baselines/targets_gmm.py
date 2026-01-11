import jax
import jax.numpy as jnp
import numpy as np


def remove_replication_dim(arr):
    """
    Remove the leading replication dimension added by jax.device_put_replicated().

    When GMM statistics are replicated across devices, they get an extra leading
    dimension equal to the number of devices. This function detects and removes it.

    Args:
        arr: Array that may have a replication dimension

    Returns:
        Array with replication dimension removed (if it existed)

    Examples:
        [8, 4096] -> [4096]           (1D array replicated)
        [8, 64, 4096] -> [64, 4096]   (2D array replicated)
        [8, K, 64, 64] -> [K, 64, 64] (3D array replicated)
    """
    device_count = jax.local_device_count()

    # Check if first dimension matches device count (indicates replication)
    if arr.shape[0] == device_count and arr.ndim >= 2:
        # Take the first slice - all slices are identical due to replication
        return arr[0]

    return arr


def get_targets(FLAGS, key, train_state, images, labels, gmm_stats, force_t=-1, force_dt=-1):
    """
    GMM-Prior Flow Matching Targets.
    Features:
    1. Hard Assignment using Log-Likelihood (Mahalanobis Distance).
    2. Importance Sampling Weights Calculation.
    3. Handle latent dataset mode (pre-computed x_0).
    4. JIT-safe (no python control flow on tracers).
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

    # === GMM PRIOR MODE (WITH PCA) ===
    else:
        x_1 = images

        # === 0. EXTRACT PCA PARAMS ===
        # Flatten images to pixel space: [B, 4096]
        x_flat = x_1.reshape(B, -1)

        # PCA parameters for projection - remove replication dimension if present
        pca_comps = remove_replication_dim(gmm_stats['pca_components'])  # [pca_dim, 4096]
        pca_mean = remove_replication_dim(gmm_stats['pca_mean'])         # [4096]

        # GMM parameters (in PCA space) - remove replication dimension if present
        means = remove_replication_dim(gmm_stats['means'])      # [K, pca_dim]
        covs = remove_replication_dim(gmm_stats['covs'])        # [K, pca_dim]
        weights = remove_replication_dim(gmm_stats['weights'])  # [K]

        # Ensure weights is 1D (flatten in case of any remaining dimensions)
        weights = weights.flatten()

        K = weights.shape[0]
        pca_dim = means.shape[1]

        # === 1. FORWARD PCA PROJECTION ===
        # Project to PCA space: z = (x - mean) @ V.T
        x_centered = x_flat - pca_mean
        z_flat = jnp.dot(x_centered, pca_comps.T)  # [B, pca_dim]

        # === 2. HARD ASSIGNMENT (in PCA space) ===
        # Compute Log-Likelihood Score in PCA space (pca_dim instead of 4096)
        # Score_k = log(pi_k) - 0.5 * sum(log(sigma_k^2)) - 0.5 * sum((z - mu_k)^2 / sigma_k^2)

        # Term 1: Log Weights [K]
        log_weights = jnp.log(weights + 1e-10)

        # Term 2: Log Determinant [K] (Sum log variances for diagonal covariance)
        log_det = jnp.sum(jnp.log(covs + 1e-10), axis=-1)

        # Term 3: Mahalanobis Distance [B, K] in PCA space
        # Broadcasting: [B, 1, pca_dim] - [1, K, pca_dim] -> [B, K, pca_dim]
        diff = z_flat[:, None, :] - means[None, :, :]
        mahalanobis = jnp.sum((diff ** 2) / (covs[None, :, :] + 1e-10), axis=-1)

        # Aggregate Score [B, K]
        log_probs = log_weights[None, :] - 0.5 * log_det[None, :] - 0.5 * mahalanobis

        # Hard Assignment: Choose cluster k with highest score
        cluster_ids = jnp.argmax(log_probs, axis=-1)  # [B]

        # === 3. IMPORTANCE SAMPLING WEIGHTS ===
        # Lấy Pi (Model Belief)
        model_pi = jnp.take(weights, cluster_ids)

        # Lấy P_emp (Data Reality) từ stats - remove replication dimension if present
        emp_prob = remove_replication_dim(gmm_stats['empirical_probs']).flatten()
        emp_prob_selected = jnp.take(emp_prob, cluster_ids)

        # Tính Weight: w = Pi / P_emp
        # Nếu GMM gán trọng số cao (Pi) cho vùng ít dữ liệu (P_emp thấp) -> Weight lớn
        raw_weights = model_pi / (emp_prob_selected + 1e-8)

        # Normalize weights trong batch để giữ cho Loss scale ổn định (Mean ~ 1)
        # Điều này cực kỳ quan trọng để không làm Learning Rate bị sai lệch
        loss_weights = raw_weights / (jnp.mean(raw_weights) + 1e-8)

        # Store in info for train.py to use
        info['loss_weights'] = loss_weights
        info['max_is_weight'] = jnp.max(loss_weights)

        # === 4. SAMPLE x_0 IN PCA SPACE & INVERSE PROJECT ===
        # Get cluster parameters in PCA space [K, pca_dim]
        batch_means_pca = jnp.take(means, cluster_ids, axis=0)  # [B, pca_dim]
        batch_vars_pca = jnp.take(covs, cluster_ids, axis=0)    # [B, pca_dim]
        batch_stds_pca = jnp.sqrt(batch_vars_pca)

        # Sample z_0 ~ N(mu_k, sigma_k) in PCA space
        eps_pca = jax.random.normal(noise_key, z_flat.shape)  # [B, pca_dim]
        z_0 = batch_means_pca + batch_stds_pca * eps_pca     # [B, pca_dim]

        # Inverse PCA projection: x = z @ V + mean
        # [B, pca_dim] @ [pca_dim, 4096] -> [B, 4096]
        x_0_flat = jnp.dot(z_0, pca_comps) + pca_mean

        # Reshape to image space [B, H, W, C] for flow matching
        x_0 = x_0_flat.reshape(images.shape)

    # === 5. STANDARD FLOW MATCHING INTERPOLATION ===
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

    # === 6. LABEL DROPOUT FOR CLASSIFIER-FREE GUIDANCE ===
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # === 7. DT_BASE (for compatibility with shortcut models) ===
    # For naive/gmm-prior mode, dt_base is always maximum (log2(denoise_timesteps))
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(B, dtype=jnp.int32) * dt_flow

    return x_t, v_t, t, dt_base, labels_dropped, info
