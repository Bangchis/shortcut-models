import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from functools import partial
from utils import summary_csv


def eval_moe3_fid(
    FLAGS,
    train_state,
    step,
    shard_data,
    vae_decode,
    get_fid_activations,
    fid_from_stats,
    truth_fid_stats,
    image_shape,
):
    if FLAGS.fid_stats is None or get_fid_activations is None:
        return

    from utils.moe3 import load_moe3_inference_state, assign_labels_from_noise

    with jax.spmd_mode('allow_all'):
        centroids, bias = None, None
        if FLAGS.model.moe3_condition_on_k:
            centroids, bias = load_moe3_inference_state(FLAGS.model.moe3_cache_dir)

        @partial(jax.jit, static_argnums=(5,))
        def call_model(train_state, images, t, dt, labels, use_ema=True):
            if use_ema and FLAGS.model.use_ema:
                call_fn = train_state.call_model_ema
            else:
                call_fn = train_state.call_model
            return call_fn(images, t, dt, labels, train=False)

        def do_fid_calc(denoise_timesteps):
            activations = []
            label_counts = np.zeros((FLAGS.model.num_classes,), dtype=np.int64)
            num_generations = 4096
            num_batches = num_generations // FLAGS.batch_size
            print(f"Calc moe3 FID for denoise_timesteps {denoise_timesteps}")
            for fid_it in tqdm.tqdm(range(num_batches)):
                key = jax.random.PRNGKey(42)
                key = jax.random.fold_in(key, fid_it)
                key = jax.random.fold_in(key, jax.process_index())
                eps_key, _ = jax.random.split(key)
                x = jax.random.normal(eps_key, image_shape)
                if FLAGS.model.moe3_condition_on_k:
                    labels = assign_labels_from_noise(x, centroids, bias)
                else:
                    labels = np.zeros((image_shape[0],), dtype=np.int32)
                label_counts += np.bincount(
                    np.asarray(labels), minlength=FLAGS.model.num_classes)
                x, labels = shard_data(x, labels)

                delta_t = 1.0 / denoise_timesteps
                for ti in range(denoise_timesteps):
                    t = ti / denoise_timesteps
                    t_vector = jnp.full((image_shape[0],), t)
                    dt_flow = np.log2(FLAGS.model.denoise_timesteps).astype(jnp.int32)
                    dt_base = jnp.ones(image_shape[0], dtype=jnp.int32) * dt_flow
                    t_vector, dt_base = shard_data(t_vector, dt_base)
                    v = call_model(train_state, x, t_vector, dt_base, labels)
                    x = x + v * delta_t

                if FLAGS.model.use_stable_vae:
                    x = vae_decode(x)
                x = jax.image.resize(
                    x, (x.shape[0], 299, 299, 3), method='bilinear', antialias=False)
                x = jnp.clip(x, -1, 1)
                acts = get_fid_activations(x)[..., 0, 0, :]
                acts = jax.experimental.multihost_utils.process_allgather(acts)
                activations.append(np.array(acts))
            return activations, label_counts

        denoise_timesteps_list = list(dict.fromkeys(
            [1, 4, 32, int(FLAGS.model.denoise_timesteps)]))
        for denoise_timesteps in denoise_timesteps_list:
            activations, label_counts = do_fid_calc(int(denoise_timesteps))
            if jax.process_index() == 0:
                activations = np.concatenate(activations, axis=0)
                activations = activations.reshape((-1, activations.shape[-1]))
                mu1 = np.mean(activations, axis=0)
                sigma1 = np.cov(activations, rowvar=False)
                fid = fid_from_stats(
                    mu1, sigma1, truth_fid_stats['mu'], truth_fid_stats['sigma'])
                priors = label_counts.astype(np.float32) / max(1, np.sum(label_counts))
                print(f"moe3 FID for denoise_timesteps {denoise_timesteps} is {fid}")
                fid_metrics = {
                    f'fid/timesteps/{denoise_timesteps}': fid,
                    f'fid/moe3_label_prior_min/{denoise_timesteps}': float(np.min(priors)),
                    f'fid/moe3_label_prior_max/{denoise_timesteps}': float(np.max(priors)),
                    f'fid/moe3_label_prior_std/{denoise_timesteps}': float(np.std(priors)),
                }
                wandb.log(fid_metrics, step=step)
                summary_csv.log_both(step, fid_metrics, phase='eval')


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
            # Debug tạm: kiểm tra shape gốc
            print(f"Debug: Original img shape: {img.shape}")

            # Fix chính: Nếu batched (len(shape) > 3 hoặc shape[0] >1), lấy sample đầu tiên cho viz
            if len(img.shape) > 3 or img.shape[0] > 1:
                img = img[0]  # Lấy first sample (32,32,4)
                print(f"Debug: Shape after taking [0]: {img.shape}")

            # Squeeze general và conditional (an toàn)
            img = jnp.squeeze(img)
            print(f"Debug: Shape after general squeeze: {img.shape}")

            if img.shape[-1] == 1:
                img = jnp.squeeze(img, axis=-1)
                print(f"Debug: Shape after axis=-1 squeeze: {img.shape}")

            if FLAGS.model.use_stable_vae:
                # Giờ img single, [None] → (1,32,32,4) OK
                img = vae_decode(img[None])[0]
                # Mong (256,256,3)
                print(f"Debug: Shape after vae_decode: {img.shape}")

            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)

            # Mong (256,256,3)
            print(f"Debug: Final img shape for imshow: {img.shape}")
            return img

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
                x_t = jax.experimental.multihost_utils.process_allgather(x_t)
                x_1_pred = jax.experimental.multihost_utils.process_allgather(
                    x_1_pred)
                valid_images_gather = jax.experimental.multihost_utils.process_allgather(
                    shard_data(valid_images_tile))
                if jax.process_index() == 0:
                    # valid_images_gather is [batchsize] wide. Every 8 corresponds to a timescale.
                    fig, axs = plt.subplots(8, 4*3, figsize=(30, 30))

                    for j in range(min(4, valid_images_gather.shape[0] // 8)):
                        for k in range(8):
                            axs[k, 3*j].imshow(process_img(
                                valid_images_gather[j*8 + k]), vmin=0, vmax=1)
                            axs[k, 3*j +
                                1].imshow(process_img(x_t[j*8 + k]), vmin=0, vmax=1)
                            axs[k, 3*j +
                                2].imshow(process_img(x_1_pred[j*8 + k]), vmin=0, vmax=1)
                    wandb.log(
                        {f'reconstruction_{dt_type}': wandb.Image(fig)}, step=step)
                    plt.close(fig)

        print("Denoising at N steps")

        # --- Helper: minibatch variance stats (global across devices/hosts) ---
###################################################################################################

        def _mb_var_stats_global(a):
            """
            a: [B, ...] (latent hoặc pixel).
            Trả về 4 số trên GLOBAL batch (sau allgather):
            - mb_var_mean : mean   over batch của per-sample spatial variance (σ²)
            - mb_var_max  : max    over batch của σ²
            - mb_var_min  : min    over batch của σ²
            - mb_var_std  : std    over batch của σ²  (== σ(σ²), cùng đơn vị với σ²)
            """
            a32 = a.astype(jnp.float32)
            red_axes = tuple(range(1, a32.ndim)) if a32.ndim >= 2 else ()
            mean_b = jnp.mean(a32, axis=red_axes)           # [B_local]
            mean2_b = jnp.mean(a32 * a32, axis=red_axes)     # [B_local]
            var_b = jnp.maximum(mean2_b - mean_b * mean_b,
                                0.0)  # per-sample σ²
            var_b_g = jax.experimental.multihost_utils.process_allgather(
                var_b).reshape(-1)
            return {
                "mb_var_mean": jnp.mean(var_b_g),
                "mb_var_max":  jnp.max(var_b_g),
                "mb_var_min":  jnp.min(var_b_g),   # NEW
                "mb_var_std":  jnp.std(var_b_g),   # NEW (σ(σ²))
            }
####################################################################################################

        def _nice_xticks(T: int):
            """
            Tạo mốc tick trục X (0..T) dễ đọc: ~9 mốc lớn, và mốc nhỏ mỗi bước (nếu T không quá lớn).
            """
            if T <= 16:
                major = np.arange(0, T + 1, 1)
                minor = None  # đã đủ dày
            else:
                step_major = max(1, T // 8)  # ~8 khoảng
                major = np.arange(0, T + 1, step_major)
                minor = np.arange(0, T + 1, 1)
            return major, minor

        def _plot_mbvar(xs, stats_mean, stats_max, stats_min, stats_std, T, step):
            """
            Vẽ 4 đường: mean σ², max σ², min σ², std[σ²] theo xs (0..T).
            Các đại lượng này đều cùng đơn vị (σ²), nên đặt chung một trục Y là hợp lý.
            """

            # hàm này hỗ trợ cho aesthetics
            def _best_ylim(series_list):
                """
                series_list: danh sách các list số (vd. [stats_mean, stats_max, stats_var]).
                Trả về (ymin, ymax) có đệm nhỏ cho nhìn rõ.
                """
                ymin = min(min(s) for s in series_list)
                ymax = max(max(s) for s in series_list)
                if ymax <= ymin:
                    ymax = ymin + 1e-6
                pad = 0.05 * (ymax - ymin)
                return ymin - pad, ymax + pad

            if MBVAR_RELATIVE:  # gần như không bao giờ dùng
                base_mean = stats_mean[0]
                base_max = stats_max[0]
                base_min = stats_min[0]
                base_std = stats_std[0]
                if MBVAR_PERCENT:  # gần như không bao giờ dùng
                    mean_s = [100.0*(v/(base_mean+1e-12)-1.0)
                              for v in stats_mean]
                    max_s = [100.0*(v/(base_max + 1e-12)-1.0)
                             for v in stats_max]
                    min_s = [100.0*(v/(base_min + 1e-12)-1.0)
                             for v in stats_min]
                    std_s = [100.0*(v/(base_std + 1e-12)-1.0)
                             for v in stats_std]
                    y_label = "Δ vs t=0 (%)"
                else:  # gần như không bao giờ dùng
                    mean_s = [v - base_mean for v in stats_mean]
                    max_s = [v - base_max for v in stats_max]
                    min_s = [v - base_min for v in stats_min]
                    std_s = [v - base_std for v in stats_std]
                    y_label = "Δ (đơn vị σ²)"
            else:  # mặc định
                mean_s, max_s, min_s, std_s = stats_mean, stats_max, stats_min, stats_std
                y_label = "minibatch variance / std(σ²)"

            fig = plt.figure(figsize=(8, 5))
            plt.plot(xs, mean_s, marker='o', linewidth=1.6, label="mean σ²")
            plt.plot(xs, max_s,  marker='o', linewidth=1.6, label="max σ²")
            plt.plot(xs, min_s,  marker='o', linewidth=1.6, label="min σ²")
            plt.plot(xs, std_s,  marker='o', linewidth=1.6, label="std[σ²]")

            major, minor = _nice_xticks(T)
            ax = plt.gca()
            ax.set_xlim(0, T)
            ax.set_xticks(major)
            if minor is not None:
                ax.set_xticks(minor, minor=True)
            ax.set_xlabel("denoising step k (0..T)")

            if MBVAR_YMODE == "log":
                ax.set_yscale("log")
                ax.grid(True, which='both', alpha=0.25)
            else:
                ymin, ymax = _best_ylim([mean_s, max_s, min_s, std_s])
                ax.set_ylim(ymin, ymax)
                ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
                ax.grid(True, which='major', alpha=0.3)
                ax.grid(True, which='minor', alpha=0.15)

            ax.set_ylabel(y_label)
            ax.set_title(f"MB variance vs step | T={T} | step={step}")
            ax.legend()
            return fig

        denoise_timesteps_list = [1, 2, 4, 8, 16, 32]
        if FLAGS.model.denoise_timesteps == 128:
            denoise_timesteps_list.append(128)
        if FLAGS.model.cfg_scale != 0:
            denoise_timesteps_list.append('cfg')
###################################################################################################
        # >>> ADDED FOR MB-VAR PLOTS
        Ts_interest = [1, 4, 32, 128]  # 4 đồ thị mỗi lần eval
        MBVAR_DECODE_TO_PIXEL = False  # bật True nếu muốn đo trên pixel-space
        # labels_for_stats = labels_uncond  # đo uncond để ổn định so sánh

        # --- Cấu hình vẽ MBVAR (bạn có thể chỉnh 3 biến này) ---
        MBVAR_YMODE = "tight"     # "tight" (mặc định), "log"
        # True => vẽ Δ so với t=0 (đơn vị tuyệt đối), hoặc phần trăm nếu muốn
        MBVAR_RELATIVE = False
        MBVAR_PERCENT = False     # Chỉ dùng nếu MBVAR_RELATIVE=True. True => vẽ % so với t=0
        # <<< ADDED

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

            # >>> ADDED FOR MB-VAR PLOTS
            collect_mbvar = (denoise_timesteps in Ts_interest) and (not do_cfg)
            if jax.process_index() == 0 and collect_mbvar:
                stats_mean, stats_max, stats_min, stats_std = [], [], [], []
                # --- ĐO TẠI t=0 (trước khi update) ---
                x0_for_stats = x
                if MBVAR_DECODE_TO_PIXEL and FLAGS.model.use_stable_vae:
                    # [-1,1] pixel-space (thường do tanh)
                    # gần như là không dùng do mô hình xử lí trên latent mà
                    x0_for_stats = vae_decode(x0_for_stats)
                s0 = _mb_var_stats_global(x0_for_stats)
                stats_mean.append(float(s0["mb_var_mean"]))
                stats_max.append(float(s0["mb_var_max"]))
                stats_min.append(float(s0["mb_var_min"]))
                stats_std.append(float(s0["mb_var_std"]))
            # <<< ADDED

            for ti in range(denoise_timesteps):
                t = ti / denoise_timesteps  # From x_0 (noise) to x_1 (data)
                t_vector = jnp.full((B_local,), t)  # (thay vì eps.shape[0])
                dt_base = jnp.ones_like(t_vector) * np.log2(denoise_timesteps)
                if FLAGS.model.train_type == 'livereflow' and denoise_timesteps < 128:
                    dt_base = jnp.zeros_like(t_vector)
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
                # giải nhiễu từng bước 1 và ngay sau đó thì tiến hành đo minibatch variance
                x = x + v * delta_t

                # >>> ADDED FOR MB-VAR PLOTS (đo sau mỗi bước)
                # --- SAU MỖI BƯỚC k=1..T ---
                # (bên trong vòng for ti in range(denoise_timesteps): sau x = x + v * delta_t)
                if collect_mbvar:
                    x_for_stats = x
                    if MBVAR_DECODE_TO_PIXEL and FLAGS.model.use_stable_vae:
                        x_for_stats = vae_decode(x_for_stats)
                    s = _mb_var_stats_global(x_for_stats)
                    if jax.process_index() == 0:
                        stats_mean.append(float(s["mb_var_mean"]))
                        stats_max.append(float(s["mb_var_max"]))
                        stats_min.append(float(s["mb_var_min"]))
                        stats_std.append(float(s["mb_var_std"]))
                # <<< ADDED

                if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == FLAGS.model.denoise_timesteps-1:
                    np_x = jax.experimental.multihost_utils.process_allgather(
                        x)
                    all_x.append(np.array(np_x))
            all_x = np.stack(all_x, axis=1)  # (batch, timesteps, H, W, C)
            all_x = all_x[:, -8:]  # Last 8 timesteps

            # >>> ADDED FOR MB-VAR PLOTS (vẽ 1 biểu đồ/ T)
            if jax.process_index() == 0 and collect_mbvar:
                xs = np.arange(0, denoise_timesteps + 1, dtype=np.int32)
                fig2 = _plot_mbvar(xs, stats_mean, stats_max,
                                   stats_min, stats_std, denoise_timesteps, step)
                wandb.log({
                    f"mbvar/plot/T{denoise_timesteps}": wandb.Image(fig2),
                    f"mbvar/T{denoise_timesteps}/mean_sigma2": stats_mean,
                    f"mbvar/T{denoise_timesteps}/max_sigma2":  stats_max,
                    f"mbvar/T{denoise_timesteps}/min_sigma2":  stats_min,   # NEW
                    f"mbvar/T{denoise_timesteps}/std_sigma2":  stats_std,   # NEW
                }, step=step)
                plt.close(fig2)
            # <<< ADDED

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
                    fid_metrics = {f'fid/timesteps/{denoise_timesteps}': fid}
                    wandb.log(fid_metrics, step=step)
                    summary_csv.log_both(step, fid_metrics, phase='eval')
