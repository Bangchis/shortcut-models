import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
from functools import partial
from absl import app, flags

from utils.projected_diag_gmm import sigma_from_raw, safe_project
from baselines.targets_projected_diag_gmm import sample_batch_radius


def sample_gmm_source(key, prior_params, images_shape, eps_proj=1e-6,
                       r_low=0.9, r_high=1.1, r_mu=0.0, r_sigma=0.25):
    """Sample source from GMM prior for inference.
    Returns: x0 [B, H, W, C].
    """
    B = images_shape[0]
    H, W, C = images_shape[1], images_shape[2], images_shape[3]
    D = H * W * C

    cat_key, gauss_key, radius_key = jax.random.split(key, 3)

    pi_logits = prior_params['pi_logits'].astype(jnp.float32)
    mu = prior_params['mu'].astype(jnp.float32)  # [K, D]
    sigma = sigma_from_raw(prior_params['r_raw'])  # [K, D]

    # Sample mode indices from categorical.
    log_pi = jax.nn.log_softmax(pi_logits)
    mode_idx = jax.random.categorical(cat_key, log_pi, shape=(B,))  # [B]

    # Gather selected components.
    mu_sel = mu[mode_idx]      # [B, D]
    sigma_sel = sigma[mode_idx]  # [B, D]

    # Sample from selected Gaussian.
    eps = jax.random.normal(gauss_key, (B, D), dtype=jnp.float32)
    y = mu_sel + sigma_sel * eps  # [B, D]

    # Project onto sphere.
    x0_dir, _ = safe_project(y, eps_proj)

    # Sample radius.
    r = sample_batch_radius(radius_key, B, r_low, r_high, r_mu, r_sigma)  # [B]
    x0_flat = x0_dir * r[:, None]

    # Reshape to spatial.
    x0 = x0_flat.reshape(B, H, W, C)
    return x0


def do_inference(
    FLAGS,
    train_state,
    step,
    dataset,
    dataset_valid,
    shard_data,
    vae_encode,
    vae_decode,
    update,
    get_fid_activations,
    imagenet_labels,
    visualize_labels,
    fid_from_stats,
    truth_fid_stats,
):
    with jax.spmd_mode('allow_all'):
        global_device_count = jax.device_count()
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token

        prior_params = train_state.get_prior_params(
            use_ema=bool(FLAGS.model.get('gmm_use_prior_ema', 1)))

        def process_img(img):
            img = jnp.squeeze(img)
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img

        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            if FLAGS.model['gmm_use_router_cond']:
                pi = jax.nn.softmax(prior_params['pi_logits'].astype(jnp.float32))
                router_cond = jnp.tile(pi[None, :], (images.shape[0], 1))
                output = call_fn(images, t, dt, labels, router_cond, train=False)
            else:
                output = call_fn(images, t, dt, labels, train=False)
            return output

        denoise_timesteps = FLAGS.inference_timesteps
        num_generations = FLAGS.inference_generations
        cfg_scale = FLAGS.inference_cfg_scale
        x0 = []
        x1 = []
        lab = []
        x_render = []
        activations = []
        images_shape = batch_images.shape
        print(f"[GMM] Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
        for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, fid_it)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)

            # GMM source init instead of jax.random.normal.
            x = sample_gmm_source(
                eps_key, prior_params, images_shape,
                FLAGS.model['gmm_proj_eps'],
                FLAGS.model['gmm_radius_low'], FLAGS.model['gmm_radius_high'],
                FLAGS.model['gmm_radius_mu'], FLAGS.model['gmm_radius_sigma'])

            labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
            x, labels = shard_data(x, labels)
            x0.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            delta_t = 1.0 / denoise_timesteps
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((images_shape[0], ), t)
                dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if cfg_scale == 1:
                    v = call_model(train_state, x, t_vector, dt_base, labels)
                elif cfg_scale == 0:
                    v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                else:
                    v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                    v_pred_label = call_model(train_state, x, t_vector, dt_base, labels)
                    v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                x = x + v * delta_t # Euler sampling.
            x1.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            lab.append(np.array(jax.experimental.multihost_utils.process_allgather(labels)))
            if FLAGS.model.use_stable_vae:
                x = vae_decode(x) # Image is in [-1, 1] space.
                if num_generations < 10000:
                    x_render.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
            x = jnp.clip(x, -1, 1)
            acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
            acts = jax.experimental.multihost_utils.process_allgather(acts)
            acts = np.array(acts)
            activations.append(acts)

        if jax.process_index() == 0:
            activations = np.concatenate(activations, axis=0)
            activations = activations.reshape((-1, activations.shape[-1]))
            mu1 = np.mean(activations, axis=0)
            sigma1 = np.cov(activations, rowvar=False)
            fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
            print(f"FID is {fid}")
            print(f"FID is {fid}")
            print(f"FID is {fid}")

            if FLAGS.save_dir is not None:
                os.makedirs(FLAGS.save_dir, exist_ok=True)
                x_render = np.concatenate(x_render, axis=0)
                np.save(FLAGS.save_dir + f'/x_render.npy', x_render)
