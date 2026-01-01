\
import jax
import jax.numpy as jnp
import numpy as np

from utils.gmm_prior import sample_x0_conditional


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1, gmm_prior=None):
    """
    Classic (naive) flow matching targets, except x0 is sampled from a learned
    diagonal-GMM prior in latent space.

    - Same t sampling + objective as baselines.targets_naive
    - Replace x0 ~ N(0,I) with x0 ~ GMM, conditioned on x1 via p(k|x1)
      (assignment mode configurable).
    """
    assert gmm_prior is not None, "gmm_prior must be provided for train_type='gmm-fm'"

    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    # classifier-free dropout (same as naive)
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # Sample t (same as naive)
    t = jax.random.randint(
        time_key, (images.shape[0],),
        minval=0, maxval=FLAGS.model['denoise_timesteps']
    ).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec >= 0, force_t_vec, t)

    t_full = t.reshape((-1, 1, 1, 1))

    # Define x1 (data) and sample x0 from GMM
    if 'latent' in FLAGS.dataset_name:
        # dataset provides some latent tensor; we treat the second half as x1 (same pattern as naive)
        x1 = images[..., images.shape[-1] // 2:]
    else:
        x1 = images

    B = x1.shape[0]
    flat = x1.reshape((B, -1))
    assign_mode = FLAGS.model.get('gmm_assign_mode', 'soft_sample')
    temperature = float(FLAGS.model.get('gmm_resp_temperature', 1.0))
    x0_flat, k = sample_x0_conditional(
        gmm_prior, noise_key, flat,
        assign_mode=assign_mode,
        temperature=temperature,
    )
    x0 = x0_flat.reshape(x1.shape)

    # Classic FM interpolation (same as naive)
    x_t = (1 - (1 - 1e-5) * t_full) * x0 + t_full * x1
    v_t = x1 - (1 - 1e-5) * x0

    # dt_base (same as naive)
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(x1.shape[0], dtype=jnp.int32) * dt_flow

    # mode stats for logging
    K = int(gmm_prior.pi.shape[0])
    counts = jnp.bincount(k, length=K)
    info['gmm/assign_entropy'] = -jnp.sum((counts / jnp.sum(counts + 1e-8)) * jnp.log((counts / jnp.sum(counts + 1e-8)) + 1e-8))
    info['gmm/assign_max_frac'] = jnp.max(counts) / jnp.sum(counts + 1e-8)

    return x_t, v_t, t, dt_base, labels_dropped, info
