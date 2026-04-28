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
from gmm_utils import (
    build_conditional_source,
    build_moe_geometry,
    make_moe_condition,
)

flags.DEFINE_integer('inference_timesteps', 128, 'Number of timesteps for inference.')
flags.DEFINE_integer('inference_generations', 4096, 'Number of generations for inference.')
flags.DEFINE_float('inference_cfg_scale', 1.0, 'CFG scale for inference.')

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
    gmm_state=None,
):
    with jax.spmd_mode('allow_all'):
        global_device_count = jax.device_count()
        key = jax.random.PRNGKey(42 + jax.process_index())
        batch_images, batch_labels = next(dataset)
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
            batch_images = vae_encode(key, batch_images)
            valid_images = vae_encode(key, valid_images)
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token
        eps = jax.random.normal(key, batch_images.shape)
        source_standardize_eps = 1e-6
        source_local_eta = 0.5
        if FLAGS.model.train_type == 'naive-moe-source':
            source_standardize_eps = float(np.asarray(
                gmm_state.get('standardize_eps', np.array(1e-6, dtype=np.float32))))
            source_local_eta = float(np.asarray(
                gmm_state.get('local_eta', np.array(FLAGS.model.local_eta, dtype=np.float32))))

        def process_img(img):
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img
        
        @partial(jax.jit, static_argnames=('use_ema',))
        def call_model(
            train_state,
            images,
            t,
            dt,
            labels,
            moe_condition,
            moe_geometry,
            use_ema=True,
        ):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(
                images,
                t,
                dt,
                labels,
                moe_condition=moe_condition,
                moe_geometry=moe_geometry,
                train=False,
            )
            return output

        def sample_source_prior(sample_key):
            if FLAGS.model.train_type != 'naive-moe-source':
                latents = jax.random.normal(sample_key, images_shape)
                return shard_data(latents), None, None
            mode_key, angular_key, rho_key, direction_key = jax.random.split(
                sample_key, 4)
            sampled_modes = jax.random.categorical(
                mode_key,
                jnp.log(jnp.maximum(gmm_state['pi'], 1e-8)),
                shape=(images_shape[0],),
            )
            angular_probs = gmm_state['angular_pi'][sampled_modes]
            angular_codes = jax.random.categorical(
                angular_key,
                jnp.log(jnp.maximum(angular_probs, 1e-8)),
                axis=-1,
            )
            rho = jax.random.normal(rho_key, (images_shape[0],))
            x0, _, _, _, _ = build_conditional_source(
                direction_key,
                sampled_modes,
                angular_codes,
                rho,
                images_shape[1:],
                gmm_state['mean'],
                gmm_state['std'],
                source_standardize_eps,
                gmm_state['mu'],
                gmm_state['var'],
                gmm_state['angular_centers'],
                gmm_state['radius_log_mean'],
                gmm_state['radius_log_std'],
                source_local_eta,
                FLAGS.model['source_kappa'],
                FLAGS.model['source_direction_noise'],
                eps=FLAGS.model['source_eps'],
            )
            moe_condition = make_moe_condition(
                sampled_modes, angular_codes, rho)
            moe_geometry = build_moe_geometry(
                sampled_modes,
                angular_codes,
                images_shape[1:],
                gmm_state['mu'],
                gmm_state['angular_centers'],
            )
            return shard_data(x0, moe_condition, moe_geometry)
        
        if FLAGS.mode == 'interpolate':
            seed = 5
            eps0 = jax.random.normal(jax.random.PRNGKey(seed), batch_images[0].shape)
            eps1 = jax.random.normal(jax.random.PRNGKey(seed+1), batch_images[0].shape)
            labels = jnp.ones(FLAGS.batch_size,).astype(jnp.int32) * 555
            i = jnp.linspace(0, 1, FLAGS.batch_size)
            i_neg = np.sqrt(1-i**2)
            x = eps0[None] * i_neg[:, None, None, None] + eps1[None] * i[:, None, None, None]
            t_vector = jnp.full((FLAGS.batch_size, ), 0)
            dt_vector = jnp.zeros_like(t_vector)
            cfg_scale = FLAGS.inference_cfg_scale
            v = call_model(
                train_state, x, t_vector, dt_vector, labels, None, None)
            x = x + v * 1.0
            x = vae_decode(x) # Image is in [-1, 1] space.
            x_render = np.array(jax.experimental.multihost_utils.process_allgather(x))
            os.makedirs(FLAGS.save_dir, exist_ok=True)
            np.save(FLAGS.save_dir + f'/x_render.npy', x_render)
            breakpoint()

        denoise_timesteps = FLAGS.inference_timesteps
        num_generations = FLAGS.inference_generations
        cfg_scale = FLAGS.inference_cfg_scale
        alpha = float(FLAGS.model['kfm_alpha']) if FLAGS.model['train_type'] == 'khoat-fm' else 1.0
        x0 = []
        x1 = []
        lab = []
        x_render = []
        activations = []
        images_shape = batch_images.shape
        print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
        for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
            key = jax.random.PRNGKey(42)
            key = jax.random.fold_in(key, fid_it)
            key = jax.random.fold_in(key, jax.process_index())
            eps_key, label_key = jax.random.split(key)
            x, moe_condition, moe_geometry = sample_source_prior(eps_key)
            labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
            labels = shard_data(labels)
            x0_initial = x  # initial noise for ti==0 special-case
            x0.append(np.array(jax.experimental.multihost_utils.process_allgather(x)))
            delta_t = 1.0 / denoise_timesteps
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((images_shape[0], ), t)
                if FLAGS.model.train_type in ('naive', 'naive-moe-source'):
                    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow # Smallest dt.
                else: # shortcut
                    dt_flow = np.log2(denoise_timesteps).astype(jnp.int32)
                    dt_base = jnp.ones(images_shape[0], dtype=jnp.int32) * dt_flow
                    # print(dt_base)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if cfg_scale == 1:
                    v = call_model(
                        train_state, x, t_vector, dt_base,
                        labels, moe_condition, moe_geometry)
                elif cfg_scale == 0:
                    v = call_model(
                        train_state, x, t_vector, dt_base,
                        labels_uncond, moe_condition, moe_geometry)
                else:
                    v_pred_uncond = call_model(
                        train_state, x, t_vector, dt_base,
                        labels_uncond, moe_condition, moe_geometry)
                    v_pred_label = call_model(
                        train_state, x, t_vector, dt_base,
                        labels, moe_condition, moe_geometry)
                    v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                if FLAGS.model.train_type == 'khoat-fm':
                    # Algorithm 1 Sampling Phase (linear schedule: d = delta_t)
                    if ti == 0:
                        # x_d <- (1-alpha) x0 + alpha * d * v(x0, 0, d)
                        x = (1.0 - alpha) * x0_initial + alpha * (delta_t * v)
                    else:
                        # x_{t+d} <- x_t + alpha * d * v(x_t, t, d)
                        x = x + alpha * (delta_t * v)
                elif FLAGS.model.train_type == 'consistency':
                    eps = shard_data(jax.random.normal(jax.random.fold_in(eps_key, ti), images_shape))
                    x1pred = x + v * (1-t)
                    x = x1pred * (t+delta_t) + eps * (1-t-delta_t)
                else:
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

                # x0 = np.concatenate(x0, axis=0)
                # x1 = np.concatenate(x1, axis=0)
                # lab = np.concatenate(lab, axis=0)
                # os.makedirs(FLAGS.save_dir, exist_ok=True)
                # np.save(FLAGS.save_dir + f'/x0.npy', x0)
                # np.save(FLAGS.save_dir + f'/x1.npy', x1)
                # np.save(FLAGS.save_dir + f'/lab.npy', lab)
