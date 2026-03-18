import jax
import jax.numpy as jnp
import numpy as np


def sample_time_and_labels(key, images, FLAGS):
    """Sample timesteps and apply label dropout. Gradient-free.
    Reuses the same logic as targets_naive.py.
    """
    label_key, time_key = jax.random.split(key, 2)
    info = {}

    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (images.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'],
                               jnp.arange(images.shape[0]) * 0)  # placeholder, overridden by caller
    # Note: labels_dropped is computed by caller who passes actual labels.
    # This function only handles the dropout mask and time sampling.

    # Sample t.
    t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']

    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    return t, dt_base, info


def apply_label_dropout(key, labels, FLAGS):
    """Apply class dropout for classifier-free guidance."""
    labels_dropout = jax.random.bernoulli(key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    dropped_ratio = jnp.mean(labels_dropped == FLAGS.model['num_classes'])
    return labels_dropped, dropped_ratio


def sample_batch_radius(key, batch_size, r_low=0.9, r_high=1.1, r_mu=0.0, r_sigma=0.25):
    """Sample per-sample radius from logit-normal on [r_low, r_high].
    Returns: r [batch_size].
    """
    z = jax.random.normal(key, (batch_size,), dtype=jnp.float32)
    rho = jax.nn.sigmoid(r_mu + r_sigma * z)
    r = r_low + (r_high - r_low) * rho
    return r
