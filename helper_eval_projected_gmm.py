import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from functools import partial

from utils.projected_diag_gmm import sigma_from_raw, safe_project, sample_chi_radius


def sample_gmm_source(key, prior_params, images_shape, eps_proj=1e-6):
    """Sample source from GMM prior for eval.
    Returns: x0 [B, H, W, C].
    """
    B = images_shape[0]
    H, W, C = images_shape[1], images_shape[2], images_shape[3]
    D = H * W * C

    cat_key, gauss_key, radius_key = jax.random.split(key, 3)

    pi_logits = prior_params['pi_logits'].astype(jnp.float32)
    mu = prior_params['mu'].astype(jnp.float32)
    sigma = sigma_from_raw(prior_params['r_raw'])

    log_pi = jax.nn.log_softmax(pi_logits)
    mode_idx = jax.random.categorical(cat_key, log_pi, shape=(B,))

    mu_sel = mu[mode_idx]
    sigma_sel = sigma[mode_idx]

    eps = jax.random.normal(gauss_key, (B, D), dtype=jnp.float32)
    y = mu_sel + sigma_sel * eps

    x0_dir, _ = safe_project(y, eps_proj)

    r = sample_chi_radius(radius_key, B, D)
    x0_flat = x0_dir * r[:, None]

    x0 = x0_flat.reshape(B, H, W, C)
    return x0


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
        batch_labels_sharded, valid_labels_sharded = shard_data(
            batch_labels, valid_labels)
        labels_uncond = shard_data(jnp.ones(
            batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes'])

        prior_params = train_state.get_prior_params(use_ema=False)

        # GMM source noise (fixed seed for reproducibility).
        FIX_EVAL_NOISE_SEED = 42
        eval_key = jax.random.PRNGKey(FIX_EVAL_NOISE_SEED)
        eps_eval = sample_gmm_source(
            eval_key, prior_params, batch_images.shape,
            FLAGS.model['gmm_proj_eps'])
        eps = sample_gmm_source(
            key, prior_params, batch_images.shape,
            FLAGS.model['gmm_proj_eps'])

        def process_img(img):
            print(f"Debug: Original img shape: {img.shape}")
            if len(img.shape) > 3 or img.shape[0] > 1:
                img = img[0]
                print(f"Debug: Shape after taking [0]: {img.shape}")
            img = jnp.squeeze(img)
            print(f"Debug: Shape after general squeeze: {img.shape}")
            if img.shape[-1] == 1:
                img = jnp.squeeze(img, axis=-1)
                print(f"Debug: Shape after axis=-1 squeeze: {img.shape}")
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
                print(f"Debug: Shape after vae_decode: {img.shape}")
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            print(f"Debug: Final img shape for imshow: {img.shape}")
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

        print("Training Loss per T.")
        if FLAGS.model.denoise_timesteps == 128:
            fig, axs = plt.subplots(5, 8, figsize=(15, 12))
            d_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            fig, axs = plt.subplots(3, 6, figsize=(15, 8))
            d_list = [0, 1, 2, 3, 4, 5]
        for d in d_list:
            infos = None
            for t in np.arange(0, 32):
                t = t * (1.0 / 32)

                batch_images_n, batch_labels_n = next(dataset)
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    batch_images_n = vae_encode(key, batch_images_n)
                batch_images_sharded, batch_labels_sharded = shard_data(
                    batch_images_n, batch_labels_n)
                _, info = update(train_state, train_state_teacher,
                                 batch_images_sharded, batch_labels_sharded, force_t=t, force_dt=d)
                info = jax.experimental.multihost_utils.process_allgather(info)
                if infos is None:
                    infos = jax.tree_map(lambda x: [x], info)
                else:
                    infos = jax.tree_map(lambda x, y: y + [x], info, infos)
            time_axis = np.arange(0, 32) / 32
            axs[0, d].plot(time_axis, infos['loss'])
            axs[0, d].set_title(f"All {d}")
            # GMM mode logs loss_flow instead of loss_bootstrap.
            if 'loss_flow' in infos:
                axs[1, d].plot(time_axis, infos['loss_flow'])
                axs[1, d].set_title(f"Flow {d}")
            if 'loss_mix_pre' in infos:
                axs[2, d].plot(time_axis, infos['loss_mix_pre'])
                axs[2, d].set_title(f"MixPre {d}")

            if jax.process_index() == 0:
                fig.tight_layout()
                wandb.log({f'mse': wandb.Image(fig)}, step=step)

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

            # GMM source init (fixed seed).
            x = eps_eval
            B_local = eps_eval.shape[0]
            x = shard_data(x)

            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps
                t_vector = jnp.full((B_local,), t)
                dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if not do_cfg:
                    v = call_model(train_state, x, t_vector, dt_base,
                                   visualize_labels if FLAGS.model.cfg_scale != 0 else labels_uncond)
                else:
                    v_cond = call_model(
                        train_state, x, t_vector, dt_base, visualize_labels)
                    v_uncond = call_model(
                        train_state, x, t_vector, dt_base, labels_uncond)
                    v = v_uncond + FLAGS.model.cfg_scale * (v_cond - v_uncond)
                x = x + v * delta_t

                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    np_x = jax.experimental.multihost_utils.process_allgather(x)
                    all_x.append(np.array(np_x))
            all_x = np.stack(all_x, axis=1)
            all_x = all_x[:, -8:]

            if jax.process_index() == 0:
                num_viz_samples = min(8, all_x.shape[0])
                num_viz_timesteps = min(8, all_x.shape[1])
                fig, axs = plt.subplots(num_viz_timesteps, num_viz_samples, figsize=(
                    num_viz_samples * 3, num_viz_timesteps * 3))

                if num_viz_timesteps == 1 and num_viz_samples == 1:
                    pass
                elif num_viz_timesteps == 1:
                    axs = np.array(axs).reshape(1, -1)
                elif num_viz_samples == 1:
                    axs = axs.reshape(-1, 1)

                for t in range(num_viz_timesteps):
                    for j in range(num_viz_samples):
                        sample_img = process_img(all_x[j, t])
                        if num_viz_timesteps == 1 and num_viz_samples == 1:
                            axs.imshow(sample_img, vmin=0, vmax=1)
                        else:
                            axs[t, j].imshow(sample_img, vmin=0, vmax=1)
                            axs[t, j].axis('off')
                            axs[t, j].set_title(f't={t}, sample={j}')
                d_label = 'cfg' if do_cfg else denoise_timesteps
                wandb.log({f'sample_N/{d_label}': wandb.Image(fig)}, step=step)
                plt.close(fig)

        def do_fid_calc(cfg_scale, denoise_timesteps):
            activations = []
            images_shape = batch_images.shape
            num_generations = 4096
            print(
                f"[GMM] Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
            for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
                key = jax.random.PRNGKey(42)
                key = jax.random.fold_in(key, fid_it)
                key = jax.random.fold_in(key, jax.process_index())
                eps_key, label_key = jax.random.split(key)

                # GMM source init.
                x = sample_gmm_source(
                    eps_key, prior_params, images_shape,
                    FLAGS.model['gmm_proj_eps'])

                labels = jax.random.randint(
                    label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                x, labels = shard_data(x, labels)
                delta_t = 1.0 / denoise_timesteps
                for ti in range(denoise_timesteps):
                    t = ti / denoise_timesteps
                    t_vector = jnp.full((images_shape[0], ), t)
                    dt_base = jnp.ones_like(
                        t_vector) * np.log2(denoise_timesteps)
                    t_vector, dt_base = shard_data(t_vector, dt_base)
                    if cfg_scale == 1:
                        v = call_model(train_state, x, t_vector,
                                       dt_base, labels)
                    elif cfg_scale == 0:
                        v = call_model(train_state, x, t_vector,
                                       dt_base, labels_uncond)
                    else:
                        v_pred_uncond = call_model(
                            train_state, x, t_vector, dt_base, labels_uncond)
                        v_pred_label = call_model(
                            train_state, x, t_vector, dt_base, labels)
                        v = v_pred_uncond + cfg_scale * \
                            (v_pred_label - v_pred_uncond)
                    x = x + v * delta_t  # Euler sampling.
                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x)
                x = jax.image.resize(
                    x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :]
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
                    activations = do_fid_calc(
                        FLAGS.model.cfg_scale, FLAGS.model.denoise_timesteps)
                else:
                    activations = do_fid_calc(
                        1 if FLAGS.model.cfg_scale != 0 else 0, denoise_timesteps)
                if jax.process_index() == 0:
                    activations = np.concatenate(activations, axis=0)
                    activations = activations.reshape(
                        (-1, activations.shape[-1]))
                    mu1 = np.mean(activations, axis=0)
                    sigma1 = np.cov(activations, rowvar=False)
                    fid = fid_from_stats(
                        mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                    print(
                        f"FID for denoise_timesteps {denoise_timesteps} is {fid}")
                    wandb.log(
                        {f'fid/timesteps/{denoise_timesteps}': fid}, step=step)
