"""
GMM-Prior Flow Matching Targets

Implements Algorithm 2 from "GMM-Prior Flow Matching" paper.
Similar to naive flow matching but samples x0 from GMM prior instead of N(0, I).
"""

import jax
import jax.numpy as jnp
import numpy as np
import os

from utils.gmm_prior import load_gmm_prior, sample_gmm_x0

# Global cache for GMM prior (loaded once per process)
_gmm_prior_cache = None
_gmm_cache_loaded_from = None


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    """
    Get targets for GMM-Prior Flow Matching training

    Follows Algorithm 2:
    1. Sample batch of clusters k ~ π and time t ~ U[0,1]
    2. Sample data pairs x1 ~ D_k (Data)
    3. Sample source noise x0 ~ N(μ_k, diag(σ²_k)) via GMM assignment
    4. Construct path: x_t = (1-t)x0 + t*x1; Target: u = x1 - x0
    5. Compute FM loss: ||v_θ(x_t, t) - u||²

    Args:
        FLAGS: config flags
        key: JAX random key
        train_state: training state (unused here)
        images: [B, H, W, C] data latents
        labels: [B] class labels
        force_t: forced time value (-1 = sample randomly)
        force_dt: forced dt value (-1 = use default)

    Returns:
        x_t: [B, H, W, C] interpolated latents
        v_t: [B, H, W, C] target velocity
        t: [B] time values
        dt_base: [B] dt values for model conditioning
        labels_dropped: [B] labels with CFG dropout
        info: dict with logging metrics
    """
    global _gmm_prior_cache, _gmm_cache_loaded_from

    # Split random keys
    label_key, time_key, noise_key, gmm_key = jax.random.split(key, 4)

    info = {}

    # ========================================
    # 1. Load GMM Prior (once per process)
    # ========================================
    gmm_cache_path = FLAGS.model.get('gmm_cache_path', 'gmm_cache/prior.npz')

    if _gmm_prior_cache is None:
        if not os.path.exists(gmm_cache_path):
            raise FileNotFoundError(
                f"[GMM-FM Error] GMM cache not found at: {gmm_cache_path}\n"
                f"Please run preprocessing first to fit and save GMM prior."
            )

        _gmm_prior_cache = load_gmm_prior(gmm_cache_path, verbose=True)
        _gmm_cache_loaded_from = gmm_cache_path
        print(f"[GMM-FM] Loaded GMM prior for targets (K={_gmm_prior_cache.pi.shape[0]})")

    prior = _gmm_prior_cache

    # ========================================
    # 2. Label Dropout for CFG (same as naive)
    # ========================================
    labels_dropout = jax.random.bernoulli(
        label_key,
        FLAGS.model['class_dropout_prob'],
        (labels.shape[0],)
    )
    labels_dropped = jnp.where(
        labels_dropout,
        FLAGS.model['num_classes'],  # Unconditional token
        labels
    )
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # ========================================
    # 3. Sample Time t ~ U[0, 1] (same as naive)
    # ========================================
    t = jax.random.randint(
        time_key,
        (images.shape[0],),
        minval=0,
        maxval=FLAGS.model['denoise_timesteps']
    ).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']

    # Allow forced time (for debugging/eval)
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]  # [B, 1, 1, 1]

    # ========================================
    # 4. Sample Flow Pairs x_t, v_t
    # ========================================

    # Check if dataset already contains latent pairs (x0, x1)
    if 'latent' in FLAGS.dataset_name:
        # Latent dataset: images = [x0, x1] concatenated
        x_0 = images[..., :images.shape[-1] // 2]
        x_1 = images[..., images.shape[-1] // 2:]

    else:
        # Standard dataset: x1 = data, x0 sampled from GMM

        x_1 = images  # Data latent

        # **KEY DIFFERENCE FROM NAIVE: Sample x0 from GMM prior**
        x_0 = sample_gmm_x0(
            prior,
            gmm_key,
            x_1,
            assign_mode=FLAGS.model.get('gmm_assign_mode', 'soft_sample'),
            temperature=FLAGS.model.get('gmm_resp_temperature', 1.0),
            debug=FLAGS.model.get('gmm_verbose', False)
        )

        # Debug: Log GMM-specific metrics
        if FLAGS.model.get('gmm_verbose', False):
            x0_norm = jnp.mean(jnp.linalg.norm(x_0.reshape(x_0.shape[0], -1), axis=1))
            x1_norm = jnp.mean(jnp.linalg.norm(x_1.reshape(x_1.shape[0], -1), axis=1))
            print(f"[GMM-FM Debug] x0 norm: {x0_norm:.4f}, x1 norm: {x1_norm:.4f}")

    # Construct path: x_t = (1 - (1-ε)t) x_0 + t x_1
    # Target velocity: v_t = x_1 - (1-ε) x_0
    eps = 1e-5  # Small epsilon to prevent singularity at t=1
    x_t = (1 - (1 - eps) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - eps) * x_0

    # ========================================
    # 5. Set dt_base for Model Conditioning
    # ========================================
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    # Allow forced dt (for debugging/eval)
    if force_dt >= 0:
        dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * force_dt

    # ========================================
    # 6. Additional GMM-Specific Logging
    # ========================================

    # Compute which component each sample is assigned to
    from utils.gmm_prior import gmm_logp

    logp = gmm_logp(prior, x_1)  # [B, K]
    r = jax.nn.softmax(logp / FLAGS.model.get('gmm_resp_temperature', 1.0), axis=-1)  # [B, K]

    # Assignment entropy (how uncertain is the assignment?)
    # Higher entropy = more uniform responsibilities = less clear assignment
    resp_entropy = -jnp.sum(r * jnp.log(r + 1e-10), axis=-1)  # [B]
    info['gmm/assignment_entropy_mean'] = jnp.mean(resp_entropy)
    info['gmm/assignment_entropy_std'] = jnp.std(resp_entropy)

    # Max responsibility (how confident is the top assignment?)
    max_resp = jnp.max(r, axis=-1)  # [B]
    info['gmm/max_responsibility_mean'] = jnp.mean(max_resp)

    # Number of active components (with responsibility > 0.1)
    active_components = jnp.sum(jnp.any(r > 0.1, axis=0))
    info['gmm/active_components'] = active_components

    # x0 vs x1 statistics
    x0_norm = jnp.mean(jnp.linalg.norm(x_0.reshape(x_0.shape[0], -1), axis=1))
    x1_norm = jnp.mean(jnp.linalg.norm(x_1.reshape(x_1.shape[0], -1), axis=1))
    info['gmm/x0_norm_mean'] = x0_norm
    info['gmm/x1_norm_mean'] = x1_norm
    info['gmm/norm_ratio'] = x0_norm / (x1_norm + 1e-10)

    # Standard info (same as naive)
    info['dt_base_mean'] = jnp.mean(dt_base.astype(jnp.float32))

    return x_t, v_t, t, dt_base, labels_dropped, info


def reset_gmm_cache():
    """
    Reset global GMM cache (useful for testing or reloading)
    """
    global _gmm_prior_cache, _gmm_cache_loaded_from
    _gmm_prior_cache = None
    _gmm_cache_loaded_from = None
    print("[GMM-FM] GMM cache reset")


if __name__ == "__main__":
    """Test targets_gmm_fm with dummy data"""
    print("Testing GMM-FM targets...")

    import ml_collections
    from utils.gmm_em import GMMParams

    # Create dummy FLAGS
    FLAGS = ml_collections.ConfigDict()
    FLAGS.model = ml_collections.ConfigDict()
    FLAGS.model.class_dropout_prob = 0.1
    FLAGS.model.denoise_timesteps = 128
    FLAGS.model.num_classes = 1000
    FLAGS.model.gmm_cache_path = '/tmp/test_gmm_targets.npz'
    FLAGS.model.gmm_assign_mode = 'soft_sample'
    FLAGS.model.gmm_resp_temperature = 1.0
    FLAGS.model.gmm_verbose = True
    FLAGS.dataset_name = 'imagenet256'

    # Create and save dummy GMM
    K, D = 10, 4096  # 10 components, 32x32x4
    dummy_params = GMMParams(
        pi=jnp.ones(K) / K,
        mu=jax.random.normal(jax.random.PRNGKey(0), (K, D)),
        var=jnp.ones((K, D)) * 0.5
    )

    from utils.gmm_prior import save_gmm_prior
    os.makedirs('/tmp', exist_ok=True)
    save_gmm_prior(dummy_params, FLAGS.model.gmm_cache_path, verbose=True)

    # Create dummy inputs
    batch_size = 4
    H, W, C = 32, 32, 4
    images = jax.random.normal(jax.random.PRNGKey(1), (batch_size, H, W, C))
    labels = jax.random.randint(jax.random.PRNGKey(2), (batch_size,), 0, 1000)

    # Call get_targets
    key = jax.random.PRNGKey(42)
    x_t, v_t, t, dt_base, labels_dropped, info = get_targets(
        FLAGS, key, None, images, labels
    )

    print("\n" + "="*60)
    print("Target Outputs:")
    print("="*60)
    print(f"x_t shape: {x_t.shape}")
    print(f"v_t shape: {v_t.shape}")
    print(f"t shape: {t.shape}, range: [{np.min(t):.4f}, {np.max(t):.4f}]")
    print(f"dt_base: {dt_base}")
    print(f"labels_dropped: {labels_dropped}")

    print("\n" + "="*60)
    print("Info Metrics:")
    print("="*60)
    for key, value in info.items():
        print(f"  {key:40s}: {value}")

    print("\n✓ GMM-FM targets test passed!")

    # Test different assignment modes
    print("\n" + "="*60)
    print("Testing Different Assignment Modes:")
    print("="*60)

    reset_gmm_cache()  # Reset cache

    for mode in ['soft_sample', 'moment', 'hard']:
        print(f"\n--- Mode: {mode} ---")
        FLAGS.model.gmm_assign_mode = mode
        FLAGS.model.gmm_verbose = False  # Less verbose

        x_t, v_t, t, dt_base, labels_dropped, info = get_targets(
            FLAGS, jax.random.PRNGKey(100 + hash(mode) % 100), None, images, labels
        )

        print(f"  x0 norm: {info['gmm/x0_norm_mean']:.4f}")
        print(f"  Assignment entropy: {info['gmm/assignment_entropy_mean']:.4f}")
        print(f"  Max responsibility: {info['gmm/max_responsibility_mean']:.4f}")

    print("\n✓ All modes tested successfully!")
