import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from functools import partial
import os
import csv
##############################################################################
from helper_eval_for_mb import stream_mbvar_and_csv, _nice_xticks, _plot_mbvar


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
            batch_labels.shape, dtype=jnp.int32) * FLAGS.model['num_classes'])  # Null token
        eps = jax.random.normal(key, batch_images.shape)

        # === FIXED NOISE cho "Denoising at N steps" (cùng batch nhiễu cho 80 đồ thị) ===
        FIX_EVAL_NOISE_SEED = 42
        eval_key = jax.random.PRNGKey(FIX_EVAL_NOISE_SEED)
        # eval_key = jax.random.fold_in(eval_key, jax.process_index())
        # ~N(0,I), cố định theo host
        eps_eval = jax.random.normal(eval_key, batch_images.shape)

        def process_img(img):
            # to JAX array
            img = jnp.asarray(img)

            # Nếu có batch (4D), lấy sample đầu; nếu 3D thì giữ nguyên
            if img.ndim == 4:
                img = img[0]
            elif img.ndim == 3:
                pass
            elif img.ndim == 2:
                # Trường hợp hiếm (H, W) -> thêm kênh giả
                img = img[..., None]
            else:
                raise ValueError(
                    f"Unexpected image shape for viz: {img.shape}")

            # KHÔNG squeeze thêm nữa để khỏi mất kênh
            # if img.shape[-1] == 1:  # (tuỳ dataset)
            #     img = jnp.squeeze(img, axis=-1)

            # Nếu đang dùng Stable VAE: img lúc này là latent (H, W, 4) -> decode ra pixel
            if FLAGS.model.use_stable_vae:
                # kỳ vọng img.shape[-1] == 4; nếu khác 4, nhiều khả năng bạn đã đưa pixel 3-ch vào đây
                img = vae_decode(img[None])[0]  # -> (H, W, 3)

            # Chuẩn hoá về [0,1] cho imshow
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)

            return np.array(img)

        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
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
            if FLAGS.model['train_type'] == 'shortcut':
                axs[1, d].plot(time_axis, infos['loss_flow'])
                axs[1, d].set_title(f"Flow {d}")
                axs[2, d].plot(time_axis, infos['loss_bootstrap'])
                axs[2, d].set_title(f"Bootstrap {d}")

            if jax.process_index() == 0:

                fig.tight_layout()
                wandb.log({f'mse': wandb.Image(fig)}, step=step)

        print("One-step Denoising at various t.")

        # Dùng biến cục bộ để KHÔNG ảnh hưởng eps_eval cho block N steps
        eps_one_step = eps_valid if 'latent' in FLAGS.dataset_name else eps

        for dt_type in ['flow', 'shortcut']:
            if len(jax.local_devices()) == 8:
                if dt_type == 'flow':
                    t = jnp.arange(8) / 8  # between 0 and 0.875
                    t = jnp.tile(t, valid_images.shape[0] // 8)  # [batch, etc]
                    dt = 0
                    dt_base = jnp.ones_like(
                        t) * np.log2(FLAGS.model.denoise_timesteps)
                elif dt_type == 'shortcut':
                    dt_base = jnp.array([0, 0, 0, 1, 2, 3, 4, 5])
                    if FLAGS.model.denoise_timesteps == 128:
                        dt_base = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
                    dt_base = jnp.tile(
                        dt_base, valid_images.shape[0] // 8)  # [batch, etc]
                    dt = 2.0 ** (-dt_base)
                    t = 1 - dt
                eps_tile = jnp.repeat(eps_one_step, 8, axis=0)[
                    :valid_images.shape[0]]
                valid_images_tile = jnp.repeat(valid_images, 8, axis=0)[
                    :valid_images.shape[0]]
                t_full = t[..., None, None, None]
                x_t = (1 - (1 - 1e-5) * t_full) * \
                    eps_tile + t_full * valid_images_tile
                x_t, t, dt_base = shard_data(x_t, t, dt_base)
                v_pred = call_model(
                    train_state, x_t, t, dt_base, valid_labels_sharded if FLAGS.model.cfg_scale != 0 else labels_uncond)
                x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
                x_t_host = np.array(jax.device_get(x_t))
                x1_host = np.array(jax.device_get(x_1_pred))
                valid_host = np.array(jax.device_get(valid_images_tile))

                # Flatten batch về [B_global, H, W, C]
                x_t_flat = x_t_host.reshape(-1, *x_t_host.shape[-3:])
                x1_flat = x1_host.reshape(-1, *x1_host.shape[-3:])
                valid_flat = valid_host.reshape(-1, *valid_host.shape[-3:])

                # Mỗi nhóm 8 ảnh tương ứng 8 mốc t; số nhóm khả dụng:
                groups = valid_flat.shape[0] // 8
                n_show = min(4, groups)

                if jax.process_index() == 0 and n_show > 0:
                    # 8 hàng, 3 cột mỗi sample
                    fig, axs = plt.subplots(
                        8, 3 * n_show, figsize=(6 * n_show, 24))

                    # đảm bảo axs luôn là 2D
                    axs = np.atleast_2d(axs)

                    for j in range(n_show):
                        for k in range(8):
                            base = j * 8 + k
                            axs[k, 3*j +
                                0].imshow(process_img(valid_flat[base]), vmin=0, vmax=1)
                            axs[k, 3*j +
                                1].imshow(process_img(x_t_flat[base]),   vmin=0, vmax=1)
                            axs[k, 3*j +
                                2].imshow(process_img(x1_flat[base]),    vmin=0, vmax=1)
                            axs[k, 3*j + 0].set_axis_off()
                            axs[k, 3*j + 1].set_axis_off()
                            axs[k, 3*j + 2].set_axis_off()

                    wandb.log(
                        {f"reconstruction_{dt_type}": wandb.Image(fig)}, step=step)
                    plt.close(fig)

        denoise_timesteps_list = [1, 2, 4, 8, 16, 32]
        if FLAGS.model.denoise_timesteps == 128:
            denoise_timesteps_list.append(128)
        if FLAGS.model.cfg_scale != 0:
            denoise_timesteps_list.append('cfg')
###################################################################################################

        for denoise_timesteps in denoise_timesteps_list:
            do_cfg = False
            if denoise_timesteps == 'cfg':
                denoise_timesteps = denoise_timesteps_list[-2]
                do_cfg = True

            all_x = []

            # [NEW] List để lưu giá trị Cosine Similarity qua từng bước t
            cosim_trajectory = []
            # [NEW] Biến lưu vector vận tốc bước trước đó
            v_prev = None

            delta_t = 1.0 / denoise_timesteps

            # BẮT ĐẦU TỪ CÙNG 1 BATCH NHIỄU CỐ ĐỊNH
            x = eps_eval
            B_local = eps_eval.shape[0]
            x = shard_data(x)

            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps
                t_vector = jnp.full((B_local,), t)
                dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                    dt_base = jnp.zeros_like(t_vector)

                t_vector, dt_base = shard_data(t_vector, dt_base)

                # --- CALL MODEL ---
                if not do_cfg:
                    v = call_model(train_state, x, t_vector, dt_base,
                                   visualize_labels if FLAGS.model.cfg_scale != 0 else labels_uncond)
                else:
                    v_cond = call_model(
                        train_state, x, t_vector, dt_base, visualize_labels)
                    v_uncond = call_model(
                        train_state, x, t_vector, dt_base, labels_uncond)
                    v = v_uncond + FLAGS.model.cfg_scale * (v_cond - v_uncond)

                # ==============================================================================
                # [NEW] TÍNH TOÁN TRAJECTORY STRAIGHTNESS (Cosine Sim v_t vs v_{t-1})
                # ==============================================================================
                if v_prev is not None:
                    # Flatten spatial dimensions: [Batch, H, W, C] -> [Batch, H*W*C]
                    flat_curr = v.reshape(v.shape[0], -1)
                    flat_prev = v_prev.reshape(v_prev.shape[0], -1)

                    # Tính Cosine Similarity
                    dot_prod = jnp.sum(flat_curr * flat_prev, axis=1)
                    norm_curr = jnp.linalg.norm(flat_curr, axis=1)
                    norm_prev = jnp.linalg.norm(flat_prev, axis=1)

                    # Thêm 1e-6 để tránh chia cho 0
                    cosim = dot_prod / (norm_curr * norm_prev + 1e-6)

                    # Lấy trung bình cộng của cả batch
                    mean_cosim = jnp.mean(cosim)

                    # Chuyển từ JAX array về CPU numpy float để vẽ đồ thị
                    cosim_trajectory.append(float(jax.device_get(mean_cosim)))

                # Cập nhật v_prev cho bước tiếp theo
                v_prev = v
                # ==============================================================================

                # Euler Update
                x = x + v * delta_t

                # Lưu ảnh để visualize (giữ nguyên logic cũ)
                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    np_x = jax.experimental.multihost_utils.process_allgather(
                        x)
                    all_x.append(np.array(np_x))

            all_x = np.stack(all_x, axis=1)  # (batch, timesteps, H, W, C)
            all_x = all_x[:, -8:]            # Last 8 timesteps

            # ==============================================================================
            # [NEW] VẼ & LOG ĐỒ THỊ COSINE SIMILARITY LÊN WANDB
            # ==============================================================================
            # Chỉ vẽ ở Host 0 và chỉ vẽ khi có dữ liệu (denoise_timesteps >= 2)
            if jax.process_index() == 0 and len(cosim_trajectory) > 0:
                fig_traj, ax_traj = plt.subplots(figsize=(8, 4))

                # Trục X: Index của bước chuyển (từ bước 0->1, 1->2, ...)
                steps_axis = np.arange(len(cosim_trajectory))

                ax_traj.plot(steps_axis, cosim_trajectory, marker='o',
                             markersize=4, linestyle='-', linewidth=1.5)

                # Thêm Step vào Title để dễ phân biệt khi nhìn lại ảnh
                ax_traj.set_title(
                    f'Trajectory Straightness (CosSim) @ Step {step}\nN={denoise_timesteps}, CFG={do_cfg}')
                ax_traj.set_xlabel('Step Index')
                ax_traj.set_ylabel('Cosine Similarity (v_t, v_{t+1})')
                ax_traj.set_ylim(0.0, 1.05)  # Cosim max là 1
                ax_traj.grid(True, which='both', linestyle='--', alpha=0.5)

                d_label = 'cfg' if do_cfg else denoise_timesteps
                wandb.log(
                    {f'trajectory/cosim_{d_label}': wandb.Image(fig_traj)}, step=step)
                plt.close(fig_traj)
            # ==============================================================================

            if jax.process_index() == 0:
                num_viz_samples = min(8, all_x.shape[0])  # Limit samples
                num_viz_timesteps = min(8, all_x.shape[1])  # Limit timesteps
                fig, axs = plt.subplots(num_viz_timesteps, num_viz_samples, figsize=(
                    num_viz_samples * 3, num_viz_timesteps * 3))

                # Fix reshape: Xử lý single Axes (1x1 subplot)
                if num_viz_timesteps == 1 and num_viz_samples == 1:
                    pass
                elif num_viz_timesteps == 1:
                    axs = np.array(axs).reshape(1, -1)
                elif num_viz_samples == 1:
                    axs = axs.reshape(-1, 1)

                for t in range(num_viz_timesteps):
                    for j in range(num_viz_samples):
                        sample_img = process_img(all_x[j, t])  # Single latent
                        if num_viz_timesteps == 1 and num_viz_samples == 1:
                            axs.imshow(sample_img, vmin=0, vmax=1)
                        else:
                            axs[t, j].imshow(sample_img, vmin=0, vmax=1)
                            axs[t, j].axis('off')
                            axs[t, j].set_title(f't={t}, sample={j}')

                d_label = 'cfg' if do_cfg else denoise_timesteps
                wandb.log({f'sample_N/{d_label}': wandb.Image(fig)}, step=step)
                plt.close(fig)

        csv_path = os.path.join(
            FLAGS.save_dir if FLAGS.save_dir is not None else '.', 'mbvar_eval.csv')

        # GỌI TRÊN TẤT CẢ HOSTS (không đặt trong if process_index==0)
        mbvar_results = stream_mbvar_and_csv(
            FLAGS=FLAGS,
            train_state=train_state,
            shard_data=shard_data,
            vae_decode=vae_decode,
            call_model=call_model,
            batch_shape=batch_images.shape,
            step=step,
            csv_path=csv_path,
            T_list=(1, 4, 32, 128),
            total_samples=1000,
            decode_to_pixel=False,
            labels_uncond=labels_uncond
        )

        # Chỉ host 0: vẽ đồ thị & upload CSV lên W&B
        if jax.process_index() == 0:
            for T, series in mbvar_results.items():
                if not series['mean']:
                    continue
                xs = np.arange(0, T+1, dtype=np.int32)
                fig2 = _plot_mbvar(xs, series['mean'], series['max'],
                                   series['min'], series['std'], T, step)
                wandb.log({
                    f"mbvar/plot/T{T}": wandb.Image(fig2),
                    f"mbvar/T{T}/mean_sigma2": series['mean'],
                    f"mbvar/T{T}/max_sigma2":  series['max'],
                    f"mbvar/T{T}/min_sigma2":  series['min'],
                    f"mbvar/T{T}/std_sigma2":  series['std'],
                }, step=step)
                plt.close(fig2)

            # ⬇️ Upload CSV lên W&B bằng Artifact (ngay tại đây)
            if os.path.exists(csv_path):
                art = wandb.Artifact(
                    f"mbvar_eval_step_{int(step)}", type="evaluation")
                art.add_file(csv_path)
                wandb.log_artifact(art)

        def do_fid_calc(cfg_scale, denoise_timesteps):
            activations = []
            images_shape = batch_images.shape
            num_generations = 4096
            print(
                f"Calc FID for CFG {cfg_scale} and denoise_timesteps {denoise_timesteps}")
            for fid_it in tqdm.tqdm(range(num_generations // FLAGS.batch_size)):
                key = jax.random.PRNGKey(42)
                key = jax.random.fold_in(key, fid_it)
                key = jax.random.fold_in(key, jax.process_index())
                eps_key, label_key = jax.random.split(key)
                x = jax.random.normal(eps_key, images_shape)
                labels = jax.random.randint(
                    label_key, (images_shape[0],), 0, FLAGS.model.num_classes)
                x, labels = shard_data(x, labels)
                delta_t = 1.0 / denoise_timesteps
                for ti in range(denoise_timesteps):
                    # From x_0 (noise) to x_1 (data)
                    t = ti / denoise_timesteps
                    t_vector = jnp.full((images_shape[0], ), t)
                    dt_base = jnp.ones_like(
                        t_vector) * np.log2(denoise_timesteps)
                    if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                        dt_base = jnp.zeros_like(t_vector)
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
                    x = vae_decode(x)  # Image is in [-1, 1] space.
                x = jax.image.resize(
                    x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                # [devices, batch//devices, 2048]
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
