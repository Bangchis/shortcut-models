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

# lưu ý x_cin ở đây có thể là y ở những file khác , nói đúng hơn là state sau khi đi qua ConditionInstanceNorm


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
    if jax.process_index() == 0:
        print(f"\n{'='*80}", flush=True)
        print(f"[EVAL] Starting eval_model at step={step}", flush=True)
        print(f"{'='*80}\n", flush=True)

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

            # model giờ trả (v, x_cin)
            v, x_cin = call_fn(images, t, dt, labels, train=False)
            return v, x_cin

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

                v_pred, x_cin = call_model(
                    train_state, x_t, t, dt_base,
                    valid_labels_sharded if FLAGS.model.cfg_scale != 0 else labels_uncond
                )

                # Version A: state base là x_cin, không phải x_t gốc
                x_1_pred = x_cin + v_pred * (1 - t[..., None, None, None])

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
            delta_t = 1.0 / denoise_timesteps
            # x = eps # [local_batch, ...]
            # x = shard_data(x) # [batch, ...] (on all devices)

            # BẮT ĐẦU TỪ CÙNG 1 BATCH NHIỄU CỐ ĐỊNH
            x = eps_eval                      # (thay vì: x = eps)
            B_local = eps_eval.shape[0]       # size trước khi shard
            x = shard_data(x)

            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps  # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((B_local,), t)  # (thay vì eps.shape[0])
                dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                    dt_base = jnp.zeros_like(t_vector)
                t_vector, dt_base = shard_data(t_vector, dt_base)
                if not do_cfg:
                    v, x_cin = call_model(
                        train_state, x, t_vector, dt_base,
                        visualize_labels if FLAGS.model.cfg_scale != 0 else labels_uncond
                    )
                else:
                    v_cond, x_cin_cond = call_model(
                        train_state, x, t_vector, dt_base, visualize_labels
                    )
                    v_uncond, x_cin_uncond = call_model(
                        train_state, x, t_vector, dt_base, labels_uncond
                    )
                    v = v_uncond + FLAGS.model.cfg_scale * (v_cond - v_uncond)
                    # về lý thuyết x_cin_cond == x_cin_uncond vì cùng (x,t) → dùng 1 cái
                    x_cin = x_cin_cond

                # Version A: bước từ state đã norm
                x = x_cin + v * delta_t

                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    np_x = jax.experimental.multihost_utils.process_allgather(
                        x)
                    all_x.append(np.array(np_x))
            all_x = np.stack(all_x, axis=1)  # (batch, timesteps, H, W, C)
            all_x = all_x[:, -8:]  # Last 8 timesteps

            if jax.process_index() == 0:
                num_viz_samples = min(8, all_x.shape[0])  # Limit samples
                num_viz_timesteps = min(8, all_x.shape[1])  # Limit timesteps
                fig, axs = plt.subplots(num_viz_timesteps, num_viz_samples, figsize=(
                    num_viz_samples * 3, num_viz_timesteps * 3))

                # Fix reshape: Xử lý single Axes (1x1 subplot)
                if num_viz_timesteps == 1 and num_viz_samples == 1:
                    # Single subplot: axs là Axes object, không cần reshape
                    pass
                elif num_viz_timesteps == 1:
                    # 1 row, multiple cols: axs là 1D array, reshape thành 2D (1, N)
                    axs = np.array(axs).reshape(1, -1)
                elif num_viz_samples == 1:
                    # Multiple rows, 1 col: axs là 2D với shape (M, 1), transpose nếu cần
                    axs = axs.reshape(-1, 1)

                for t in range(num_viz_timesteps):
                    for j in range(num_viz_samples):
                        sample_img = process_img(all_x[j, t])  # Single latent
                        if num_viz_timesteps == 1 and num_viz_samples == 1:
                            # Direct call cho single Axes
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
            if jax.process_index() == 0:
                print(f"[FID-P{jax.process_index()}] START do_fid_calc: CFG={cfg_scale}, T={denoise_timesteps}", flush=True)

            # Only show tqdm on process 0 to avoid I/O blocking
            iterator = range(num_generations // FLAGS.batch_size)
            if jax.process_index() == 0:
                iterator = tqdm.tqdm(iterator, desc=f"FID CFG={cfg_scale} T={denoise_timesteps}")

            for fid_it in iterator:
                # Log first iteration to confirm loop starts
                if fid_it == 0 and jax.process_index() == 0:
                    print(f"[FID-P{jax.process_index()}] First iteration of sampling loop", flush=True)

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
                        v, x_cin = call_model(
                            train_state, x, t_vector, dt_base, labels)
                    elif cfg_scale == 0:
                        v, x_cin = call_model(
                            train_state, x, t_vector, dt_base, labels_uncond)
                    else:
                        v_uncond, x_cin_uncond = call_model(
                            train_state, x, t_vector, dt_base, labels_uncond
                        )
                        v_label, x_cin_label = call_model(
                            train_state, x, t_vector, dt_base, labels
                        )
                        v = v_uncond + cfg_scale * (v_label - v_uncond)
                        x_cin = x_cin_label  # x_cin_uncond ≈ x_cin_label

                    x = x_cin + v * delta_t  # Euler sampling theo Version A.

                    # Diagnostic: Check x_cin (BatchNorm output) and x for NaN/Inf at first iteration
                    if fid_it == 0 and ti == 0 and jax.process_index() == 0:
                        x_cin_cpu = np.array(x_cin)
                        v_cpu = np.array(v)
                        x_cpu = np.array(x)
                        print(f"[FID-DEBUG] First sample, first step (t={t:.3f}):", flush=True)
                        print(f"[FID-DEBUG]   x_cin (BatchNorm output): has_nan={np.isnan(x_cin_cpu).any()}, has_inf={np.isinf(x_cin_cpu).any()}, min={np.min(x_cin_cpu):.4f}, max={np.max(x_cin_cpu):.4f}, mean={np.mean(x_cin_cpu):.4f}", flush=True)
                        print(f"[FID-DEBUG]   v (velocity): has_nan={np.isnan(v_cpu).any()}, has_inf={np.isinf(v_cpu).any()}, min={np.min(v_cpu):.4f}, max={np.max(v_cpu):.4f}, mean={np.mean(v_cpu):.4f}", flush=True)
                        print(f"[FID-DEBUG]   x (after step): has_nan={np.isnan(x_cpu).any()}, has_inf={np.isinf(x_cpu).any()}, min={np.min(x_cpu):.4f}, max={np.max(x_cpu):.4f}, mean={np.mean(x_cpu):.4f}", flush=True)

                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x)  # Image is in [-1, 1] space.
                x = jax.image.resize(
                    x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)

                # Diagnostic: Check final x before FID network at first iteration
                if fid_it == 0 and jax.process_index() == 0:
                    x_final_cpu = np.array(x)
                    print(f"[FID-DEBUG] First sample, final image (after resize & clip): has_nan={np.isnan(x_final_cpu).any()}, has_inf={np.isinf(x_final_cpu).any()}, min={np.min(x_final_cpu):.4f}, max={np.max(x_final_cpu):.4f}", flush=True)
                # [devices, batch//devices, 2048]
                acts = get_fid_activations(x)[..., 0, 0, :]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                acts = np.array(acts)
                activations.append(acts)

            if jax.process_index() == 0:
                print(f"[FID-P{jax.process_index()}] DONE do_fid_calc: CFG={cfg_scale}, T={denoise_timesteps}, collected {len(activations)} batches", flush=True)
            return activations

        if FLAGS.fid_stats is not None:
            if jax.process_index() == 0:
                print(f"\n[EVAL] Starting FID calculation section...", flush=True)

                # Diagnostic: Check batch_stats before FID calculation
                print(f"[EVAL-DEBUG] Checking batch_stats from train_state...", flush=True)
                if hasattr(train_state, 'batch_stats') and train_state.batch_stats is not None:
                    batch_stats = train_state.batch_stats
                    print(f"[EVAL-DEBUG] batch_stats keys: {batch_stats.keys()}", flush=True)

                    # Check if we have ConditionalBatchNormSpecialT stats
                    if 'ConditionalBatchNormSpecialT_0' in batch_stats:
                        bn_stats = batch_stats['ConditionalBatchNormSpecialT_0']
                        if 'mean' in bn_stats and 'var' in bn_stats:
                            running_mean = np.array(bn_stats['mean'])
                            running_var = np.array(bn_stats['var'])

                            print(f"[EVAL-DEBUG] BatchNorm running_mean: shape={running_mean.shape}, has_nan={np.isnan(running_mean).any()}, has_inf={np.isinf(running_mean).any()}", flush=True)
                            print(f"[EVAL-DEBUG] BatchNorm running_var: shape={running_var.shape}, has_nan={np.isnan(running_var).any()}, has_inf={np.isinf(running_var).any()}", flush=True)

                            # Check each special_t
                            K = running_mean.shape[0]
                            for k in range(K):
                                mean_k = running_mean[k]
                                var_k = running_var[k]
                                print(f"[EVAL-DEBUG] special_t[{k}]: mean range=[{mean_k.min():.4e}, {mean_k.max():.4e}], mean={mean_k.mean():.4e}", flush=True)
                                print(f"[EVAL-DEBUG] special_t[{k}]: var range=[{var_k.min():.4e}, {var_k.max():.4e}], mean={var_k.mean():.4e}", flush=True)

                                num_neg = np.sum(var_k < 0)
                                num_zero = np.sum(np.abs(var_k) < 1e-10)
                                num_large = np.sum(var_k > 1e6)
                                if num_neg > 0 or num_zero > 0 or num_large > 0:
                                    print(f"[EVAL-DEBUG] special_t[{k}] WARNING: neg={num_neg}, near_zero={num_zero}, large={num_large}", flush=True)
                        else:
                            print(f"[EVAL-DEBUG] batch_stats exists but missing mean/var", flush=True)
                    else:
                        print(f"[EVAL-DEBUG] ConditionalBatchNormSpecialT_0 not found in batch_stats", flush=True)
                else:
                    print(f"[EVAL-DEBUG] train_state has no batch_stats", flush=True)

            denoise_timesteps_list = [1, 4, 32]
            if FLAGS.model.denoise_timesteps == 128:
                denoise_timesteps_list.append(128)
            if FLAGS.model.cfg_scale != 0:
                denoise_timesteps_list.append('cfg')
            for denoise_timesteps in denoise_timesteps_list:
                if jax.process_index() == 0:
                    cfg_val = FLAGS.model.cfg_scale if denoise_timesteps == 'cfg' else (1 if FLAGS.model.cfg_scale != 0 else 0)
                    print(f"[FID] Starting T={denoise_timesteps}, cfg_scale={cfg_val}", flush=True)

                if denoise_timesteps == 'cfg':
                    activations = do_fid_calc(
                        FLAGS.model.cfg_scale, FLAGS.model.denoise_timesteps)
                else:
                    activations = do_fid_calc(
                        1 if FLAGS.model.cfg_scale != 0 else 0, denoise_timesteps)

                if jax.process_index() == 0:
                    print(f"[FID] Returned from do_fid_calc T={denoise_timesteps}, now computing stats...", flush=True)

                # Sync all processes before stats computation
                jax.experimental.multihost_utils.sync_global_devices(f"fid_sync_{denoise_timesteps}")

                if jax.process_index() == 0:
                    print(f"[FID] Concatenating activations T={denoise_timesteps}...", flush=True)
                    activations = np.concatenate(activations, axis=0)
                    activations = activations.reshape(
                        (-1, activations.shape[-1]))

                    # Diagnostic: Check activations for NaN/Inf
                    print(f"[FID-DEBUG] Activations: shape={activations.shape}, dtype={activations.dtype}", flush=True)
                    print(f"[FID-DEBUG] Activations: has_nan={np.isnan(activations).any()}, has_inf={np.isinf(activations).any()}", flush=True)
                    print(f"[FID-DEBUG] Activations: min={np.min(activations):.4f}, max={np.max(activations):.4f}, mean={np.mean(activations):.4f}, std={np.std(activations):.4f}", flush=True)

                    # Check per-channel stats
                    channel_means = np.mean(activations, axis=0)
                    channel_stds = np.std(activations, axis=0)
                    zero_std_channels = np.sum(channel_stds < 1e-8)
                    print(f"[FID-DEBUG] Channel stats: num_zero_std={zero_std_channels}, min_std={np.min(channel_stds):.4e}, max_std={np.max(channel_stds):.4e}", flush=True)
                    print(f"[FID-DEBUG] Channel means: min={np.min(channel_means):.4f}, max={np.max(channel_means):.4f}", flush=True)

                    if np.isnan(activations).any() or np.isinf(activations).any():
                        nan_indices = np.where(np.isnan(activations))
                        inf_indices = np.where(np.isinf(activations))
                        print(f"[FID-DEBUG] WARNING: Found NaN at {len(nan_indices[0])} positions, Inf at {len(inf_indices[0])} positions", flush=True)

                    print(f"[FID] Computing mean T={denoise_timesteps}...", flush=True)
                    mu1 = np.mean(activations, axis=0)
                    print(f"[FID] Computing covariance T={denoise_timesteps}...", flush=True)
                    sigma1 = np.cov(activations, rowvar=False)
                    print(f"[FID] Computing FID score T={denoise_timesteps}...", flush=True)
                    fid = fid_from_stats(
                        mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                    print(
                        f"[FID] FINISHED T={denoise_timesteps}: FID = {fid}", flush=True)
                    wandb.log(
                        {f'fid/timesteps/{denoise_timesteps}': fid}, step=step)
                    print(f"[FID] Logged to wandb T={denoise_timesteps}", flush=True)
