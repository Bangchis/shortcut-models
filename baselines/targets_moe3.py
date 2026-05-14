import jax
import jax.numpy as jnp
import numpy as np


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    label_key, time_key = jax.random.split(key)
    info = {}

    t = jax.random.randint(
        time_key,
        (images.shape[0],),
        minval=0,
        maxval=FLAGS.model['denoise_timesteps'],
    ).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]

    mid = images.shape[-1] // 2
    x_0 = images[..., :mid]
    x_1 = images[..., mid:]
    x_t = (1 - t_full) * x_0 + t_full * x_1
    v_t = x_1 - x_0

    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    if FLAGS.model.moe3_condition_on_k:
        labels_dropout = jax.random.bernoulli(
            label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
        labels = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
        info['moe3/dropped_ratio'] = jnp.mean(labels == FLAGS.model['num_classes'])
        info['moe3/condition_on_k'] = jnp.array(1.0, dtype=jnp.float32)
    else:
        labels = jnp.zeros(labels.shape, dtype=jnp.int32)
        info['moe3/dropped_ratio'] = jnp.array(0.0, dtype=jnp.float32)
        info['moe3/condition_on_k'] = jnp.array(0.0, dtype=jnp.float32)

    info['moe3/x0_norm'] = jnp.sqrt(jnp.mean(jnp.square(x_0)))
    info['moe3/x1_norm'] = jnp.sqrt(jnp.mean(jnp.square(x_1)))
    info['moe3/v_norm'] = jnp.sqrt(jnp.mean(jnp.square(v_t)))
    return x_t, v_t, t, dt_base, labels, info
