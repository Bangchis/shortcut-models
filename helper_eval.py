import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from functools import partial
from gmm_utils import (
    assign_angular_codes,
    build_source_base,
    flatten_and_standardize,
    posterior_from_standardized,
    sample_lognormal_radius,
)

def eval_model(
    FLAGS,
    train_state,
    train_state_teacher,
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
        if 'latent' in FLAGS.dataset_name:
            eps_valid = valid_images[..., :valid_images.shape[-1]//2]
            batch_images = batch_images[..., batch_images.shape[-1]//2:]
            valid_images = valid_images[..., valid_images.shape[-1]//2:]
        batch_labels_sharded, valid_labels_sharded = shard_data(batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes']) # Null token
        eps = jax.random.normal(key, batch_images.shape)
        alpha = float(FLAGS.model['kfm_alpha']) if FLAGS.model['train_type'] == 'khoat-fm' else 1.0
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
        
        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            output = call_fn(images, t, dt, labels, train=False)
            return output

        @partial(jax.jit, static_argnums=(6,))
        def call_source(train_state, z, x_base, modes, angular_codes, log_radius, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_source_ema
            else:
                call_fn = train_state.call_source
            return call_fn(z, x_base, modes, angular_codes, log_radius)

        def sample_source_prior(sample_key, batch_shape):
            if FLAGS.model.train_type != 'naive-moe-source':
                return shard_data(jax.random.normal(sample_key, batch_shape))
            z_key, mode_key, angular_key, radius_key, x0_key = jax.random.split(sample_key, 5)
            z = jax.random.normal(z_key, batch_shape)
            sampled_modes = jax.random.categorical(
                mode_key,
                jnp.log(jnp.maximum(gmm_state['pi'], 1e-8)),
                shape=(batch_shape[0],),
            )
            angular_probs = gmm_state['angular_pi'][sampled_modes]
            angular_codes = jax.random.categorical(
                angular_key,
                jnp.log(jnp.maximum(angular_probs, 1e-8)),
                axis=-1,
            )
            radius, log_radius = sample_lognormal_radius(
                radius_key,
                sampled_modes,
                angular_codes,
                gmm_state['radius_log_mean'],
                gmm_state['radius_log_std'],
            )
            x_base, _ = build_source_base(
                sampled_modes,
                angular_codes,
                radius,
                batch_shape[1:],
                gmm_state['mean'],
                gmm_state['std'],
                source_standardize_eps,
                gmm_state['mu'],
                gmm_state['var'],
                gmm_state['angular_centers'],
                source_local_eta,
                eps=FLAGS.model['source_eps'],
            )
            z, x_base, sampled_modes, angular_codes, log_radius = shard_data(
                z, x_base, sampled_modes, angular_codes, log_radius)
            mu_x0, log_sigma, _ = call_source(
                train_state, z, x_base, sampled_modes, angular_codes, log_radius)
            eps_x0 = shard_data(jax.random.normal(x0_key, batch_shape))
            return mu_x0 + eps_x0 * jnp.exp(log_sigma)

        def sample_source_posterior(sample_key, latents):
            if FLAGS.model.train_type != 'naive-moe-source':
                return latents
            z_key, radius_key, x0_key = jax.random.split(sample_key, 3)
            latents_std = flatten_and_standardize(
                latents,
                gmm_state['mean'],
                gmm_state['std'],
                source_standardize_eps,
            )
            q = posterior_from_standardized(
                latents_std,
                gmm_state['log_pi'],
                gmm_state['mu'],
                gmm_state['var'],
            )
            sampled_modes = jnp.argmax(q, axis=-1).astype(jnp.int32)
            angular_codes, _ = assign_angular_codes(
                latents_std,
                sampled_modes,
                gmm_state['mu'],
                gmm_state['var'],
                gmm_state['angular_centers'],
                source_local_eta,
                eps=FLAGS.model['source_eps'],
            )
            angular_codes = angular_codes.astype(jnp.int32)
            radius, log_radius = sample_lognormal_radius(
                radius_key,
                sampled_modes,
                angular_codes,
                gmm_state['radius_log_mean'],
                gmm_state['radius_log_std'],
            )
            x_base, _ = build_source_base(
                sampled_modes,
                angular_codes,
                radius,
                latents.shape[1:],
                gmm_state['mean'],
                gmm_state['std'],
                source_standardize_eps,
                gmm_state['mu'],
                gmm_state['var'],
                gmm_state['angular_centers'],
                source_local_eta,
                eps=FLAGS.model['source_eps'],
            )
            z = jax.random.normal(z_key, latents.shape)
            z, x_base, sampled_modes, angular_codes, log_radius = shard_data(
                z, x_base, sampled_modes, angular_codes, log_radius)
            mu_x0, log_sigma, _ = call_source(
                train_state, z, x_base, sampled_modes, angular_codes, log_radius)
            eps_x0 = shard_data(jax.random.normal(x0_key, latents.shape))
            return mu_x0 + eps_x0 * jnp.exp(log_sigma)

        print("Training Loss per T.")
        if FLAGS.model.denoise_timesteps == 128:
            fig, axs = plt.subplots(5, 8, figsize=(15, 12))
            d_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            fig, axs = plt.subplots(3, 6, figsize=(15, 8))
            d_list = [0, 1, 2, 3, 4, 5]
        for d in d_list:
            infos = None

            # For KFM: sweep aligned t values on the grid of size 2^d
            if FLAGS.model['train_type'] == 'khoat-fm':
                grid_n = 2 ** int(d)
                if grid_n <= 32:
                    t_values = (np.arange(0, grid_n) / grid_n).tolist()
                else:
                    idx = np.linspace(0, grid_n - 1, 32, dtype=np.int32)
                    t_values = (idx / grid_n).tolist()
            else:
                t_values = (np.arange(0, 32) / 32).tolist()

            for t in t_values:
                batch_images_n, batch_labels_n = next(dataset)
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    batch_images_n = vae_encode(key, batch_images_n)
                batch_images_sharded, batch_labels_sharded = shard_data(batch_images_n, batch_labels_n)
                _, info = update(train_state, train_state_teacher, batch_images_sharded, batch_labels_sharded, force_t=float(t), force_dt=int(d))
                info = jax.experimental.multihost_utils.process_allgather(info)
                if infos is None:
                    infos = jax.tree_map(lambda x: [x], info)
                else:
                    infos = jax.tree_map(lambda x, y: y + [x], info, infos)
            time_axis = np.array(t_values)
            axs[0, d].plot(time_axis, infos['loss'])
            axs[0, d].set_title(f"All {d}")
            if FLAGS.model['train_type'] == 'shortcut':
                axs[1, d].plot(time_axis, infos['loss_flow'])
                axs[1, d].set_title(f"Flow {d}")
                axs[2, d].plot(time_axis, infos['loss_bootstrap'])
                axs[2, d].set_title(f"Bootstrap {d}")

            if jax.process_index() == 0:
                fig.tight_layout()
                wandb.log({f'mse': wandb.Image(fig)}, step=step)


        print("One-step Denoising at various t.")
        if 'latent' in FLAGS.dataset_name:
            eps = eps_valid
        if FLAGS.model.train_type == 'naive-moe-source':
            eps = sample_source_posterior(jax.random.fold_in(key, 17), valid_images)
            eps = jax.experimental.multihost_utils.process_allgather(eps)[0]
        for dt_type in ['flow', 'shortcut']:
            if len(jax.local_devices()) == 8:
                if dt_type == 'flow':
                    t = jnp.arange(8) / 8 # between 0 and 0.875
                    t = jnp.tile(t, valid_images.shape[0] // 8) # [batch, etc]
                    dt = 0
                    dt_base = jnp.ones_like(t) * np.log2(FLAGS.model.denoise_timesteps)
                elif dt_type == 'shortcut':
                    dt_base = jnp.array([0,0,0,1,2,3,4,5])
                    if FLAGS.model.denoise_timesteps == 128:
                        dt_base = jnp.array([0,1,2,3,4,5,6,7])
                    dt_base = jnp.tile(dt_base, valid_images.shape[0] // 8) # [batch, etc]
                    dt = 2.0 ** (-dt_base)
                    t = 1 - dt
                eps_tile = jnp.repeat(eps, 8, axis=0)[:valid_images.shape[0]]
                valid_images_tile = jnp.repeat(valid_images, 8, axis=0)[:valid_images.shape[0]]
                t_full = t[..., None, None, None]
                x_t = (1 - t_full) * eps_tile + t_full * valid_images_tile
                x_t, t, dt_base = shard_data(x_t, t, dt_base)
                v_pred = call_model(train_state, x_t, t, dt_base, valid_labels_sharded if FLAGS.model.cfg_scale != 0 else labels_uncond)
                x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
                x_t = jax.experimental.multihost_utils.process_allgather(x_t) # [devices, batch, H, W, C]
                x_1_pred = jax.experimental.multihost_utils.process_allgather(x_1_pred) # [devices, batch, H, W, C]
                valid_images_gather = jax.experimental.multihost_utils.process_allgather(shard_data(valid_images_tile)) # [devices, batch, H, W, C]
                if jax.process_index() == 0:
                    # valid_images_gather is [batchsize] wide. Every 8 corresponds to a timescale.
                    x_t, x_1_pred, valid_images_gather = x_t[0], x_1_pred[0], valid_images_gather[0] #-> (batch, H, W, C)
                    fig, axs = plt.subplots(8, 4*3, figsize=(30, 30))
                    
                    for j in range(min(4, valid_images_gather.shape[0] // 8)):
                        for k in range(8):
                            axs[k,3*j].imshow(process_img(valid_images_gather[j*8 + k]), vmin=0, vmax=1)
                            axs[k,3*j+1].imshow(process_img(x_t[j*8 + k]), vmin=0, vmax=1)
                            axs[k,3*j+2].imshow(process_img(x_1_pred[j*8 + k]), vmin=0, vmax=1)
                    wandb.log({f'reconstruction_{dt_type}': wandb.Image(fig)}, step=step)
                    plt.close(fig)

        print("Denoising at N steps")

        denoise_timesteps_list = [1, 2, 4, 8, 16, 32]
        if FLAGS.model.denoise_timesteps == 128:
            denoise_timesteps_list.append(128)
        if FLAGS.model.cfg_scale != 0:
            denoise_timesteps_list.append('cfg')
        for denoise_timesteps in denoise_timesteps_list:
            do_cfg = False
            if denoise_timesteps == 'cfg':
                denoise_timesteps = denoise_timesteps_list[-2]
                do_cfg = True
            all_x = []
            delta_t = 1.0 / denoise_timesteps
            if FLAGS.model.train_type == 'naive-moe-source':
                x = sample_source_prior(jax.random.fold_in(key, denoise_timesteps), eps.shape)
            else:
                x = shard_data(eps) # [batch, ...] (on all devices)
            x0_initial = x  # initial noise for ti==0 special-case
            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((eps.shape[0],), t)
                dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                    dt_base = jnp.zeros_like(t_vector)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if not do_cfg:
                    v = call_model(train_state, x, t_vector, dt_base, visualize_labels if FLAGS.model.cfg_scale != 0 else labels_uncond)
                else:
                    v_cond = call_model(train_state, x, t_vector, dt_base, visualize_labels)
                    v_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                    v = v_uncond + FLAGS.model.cfg_scale * (v_cond - v_uncond)

                if FLAGS.model['train_type'] == 'khoat-fm':
                    if ti == 0:
                        x = (1.0 - alpha) * x0_initial + alpha * (delta_t * v)
                    else:
                        x = x + alpha * (delta_t * v)
                else:
                    x = x + v * delta_t
                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    np_x = jax.experimental.multihost_utils.process_allgather(x)
                    all_x.append(np.array(np_x))
            all_x = np.stack(all_x, axis=1) # [batch, timesteps, etc..] ->  # [devices, timesteps, batch, H, W, C]
            all_x = all_x[0]  # -> (timesteps, batch, H, W, C)
            all_x = np.transpose(all_x, (1, 0, 2, 3, 4))  # -> (batch, timesteps, H, W, C)
            all_x = all_x[:, -8:]
            if jax.process_index() == 0:
                fig, axs = plt.subplots(8, 8, figsize=(30, 30))
                for j in range(8):
                    for t in range(min(8, all_x.shape[1])):
                        axs[t, j].imshow(process_img(all_x[j, t]), vmin=0, vmax=1)
                d_label = 'cfg' if do_cfg else denoise_timesteps
                wandb.log({f'sample_N/{d_label}': wandb.Image(fig)}, step=step)
                plt.close(fig)

        def do_fid_calc(cfg_scale, denoise_timesteps):
            activations = []
            images_shape = batch_images.shape
            num_generations = 50048 #to match with paper's config
            print(f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
            for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
                key = jax.random.PRNGKey(42)
                key = jax.random.fold_in(key, fid_it)
                key = jax.random.fold_in(key, jax.process_index())
                eps_key, label_key = jax.random.split(key)
                x = sample_source_prior(eps_key, images_shape)
                labels = jax.random.randint(label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                labels = shard_data(labels)
                x0_initial = x  # initial noise for ti==0 special-case
                delta_t = 1.0 / denoise_timesteps
                for ti in range(denoise_timesteps):
                    t = ti / denoise_timesteps # From x_0 (noise) to x_1 (data)
                    t_vector = jnp.full((images_shape[0], ), t)
                    dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                    if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                        dt_base = jnp.zeros_like(t_vector)
                    t_vector, dt_base = shard_data(t_vector, dt_base)
                    if cfg_scale == 1:
                        v = call_model(train_state, x, t_vector, dt_base, labels)
                    elif cfg_scale == 0:
                        v = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                    else:
                        v_pred_uncond = call_model(train_state, x, t_vector, dt_base, labels_uncond)
                        v_pred_label = call_model(train_state, x, t_vector, dt_base, labels)
                        v = v_pred_uncond + cfg_scale * (v_pred_label - v_pred_uncond)

                    if FLAGS.model['train_type'] == 'khoat-fm':
                        if ti == 0:
                            x = (1.0 - alpha) * x0_initial + alpha * (delta_t * v)
                        else:
                            x = x + alpha * (delta_t * v)
                    else:
                        x = x + v * delta_t # Euler sampling.
                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x) # Image is in [-1, 1] space.
                x = jax.image.resize(x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :] # [devices, batch//devices, 2048]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)
            return activations
        
        if FLAGS.fid_stats is not None:
            denoise_timesteps_list = [1, 4, 32]
            if FLAGS.model.denoise_timesteps == 128:
                denoise_timesteps_list.append(128)
            if FLAGS.model.cfg_scale != 0:
                denoise_timesteps_list.append('cfg')
            for denoise_timesteps in denoise_timesteps_list:
                if denoise_timesteps == 'cfg':
                    activations = do_fid_calc(FLAGS.model.cfg_scale, FLAGS.model.denoise_timesteps)
                else:
                    activations = do_fid_calc(1 if FLAGS.model.cfg_scale != 0 else 0, denoise_timesteps)
                if jax.process_index() == 0:
                    activations = np.concatenate(activations, axis=0)
                    activations = activations.reshape((-1, activations.shape[-1]))
                    mu1 = np.mean(activations, axis=0)
                    sigma1 = np.cov(activations, rowvar=False)
                    fid = fid_from_stats(mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                    print(f"FID for denoise_timesteps {denoise_timesteps} is {fid}")
                    wandb.log({f'fid/timesteps/{denoise_timesteps}': fid}, step=step)
