import os

from absl import app, flags
from ml_collections import config_flags
import jax
import numpy as np

from gmm_utils import (
    fit_diag_gmm,
    flatten_latents,
    save_gmm_stats,
)
from utils.datasets import get_dataset, get_num_examples
from utils.stable_vae import StableVAE
from utils.wandb import default_wandb_config, setup_wandb


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'celebahq256', 'Dataset name to preprocess.')
flags.DEFINE_string('tfds_data_dir', None, 'Optional TFDS data directory.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for latent extraction.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('gmm_save_path', '/kaggle/working/gmm_stats.npz', 'Output npz path for source stats.')
flags.DEFINE_string(
    'gmm_latent_cache_path',
    '/kaggle/working/gmm_latents.npy',
    'Memmap cache path for flattened train latents.',
)
flags.DEFINE_integer(
    'gmm_fit_samples',
    -1,
    'How many train samples to use. -1 means full train split.',
)
flags.DEFINE_integer('gmm_num_modes', 16, 'Number of diagonal GMM modes.')
flags.DEFINE_integer('gmm_em_iters', 100, 'Maximum EM iterations per restart.')
flags.DEFINE_integer('gmm_em_restarts', 3, 'Number of EM restarts.')
flags.DEFINE_integer('gmm_init_seed', 0, 'Seed for GMM initialization.')
flags.DEFINE_float('gmm_standardize_eps', 1e-6, 'Epsilon used for latent standardization.')
flags.DEFINE_float('gmm_pi_uniform_beta', 0.01, 'Convex interpolation weight from empirical pi to uniform pi.')
flags.DEFINE_enum('gmm_var_update', 'scale', ['none', 'scale'], 'Diagonal variance regularization update.')
flags.DEFINE_float('gmm_var_beta', 0.05, 'Exponent for multiplicative variance scale update.')
flags.DEFINE_float('gmm_var_target', 1.0, 'Target mean diagonal variance in standardized latent space.')
flags.DEFINE_float('gmm_var_floor', 1e-4, 'Minimum variance for every GMM dimension.')
flags.DEFINE_float('gmm_var_eps', 1e-8, 'Epsilon for variance scale update.')
flags.DEFINE_float('gmm_var_scale_min', 0.5, 'Minimum per-step component variance scale.')
flags.DEFINE_float('gmm_var_scale_max', 2.0, 'Maximum per-step component variance scale.')
flags.DEFINE_float('gmm_min_component_count', 1.0, 'Diagnostic threshold for dead/low-count components.')
flags.DEFINE_integer('gmm_kmeanspp_init', 1, 'Whether to use kmeans++-style initialization.')
flags.DEFINE_integer('gmm_em_chunk_size', 1024, 'Chunk size for GMM E-step/M-step accumulation.')
flags.DEFINE_integer('gmm_keep_latent_cache', 0, 'Whether to keep the latent cache file after fitting.')
flags.DEFINE_float('local_eta', 0.5, 'Local coordinate whitening exponent.')
flags.DEFINE_integer('angular_num_submodes', 4, 'Number of spherical k-means directions per GMM mode.')
flags.DEFINE_integer('angular_kmeans_iters', 30, 'Spherical k-means iterations.')
flags.DEFINE_integer('angular_min_cluster_size', 100, 'Minimum submode count before radius stats are trusted.')
flags.DEFINE_float('radius_min_log_std', 0.05, 'Minimum std for log-radius stats.')
flags.DEFINE_float('local_eps', 1e-8, 'Epsilon for local coordinate norms.')


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'shortcut',
    'name': 'gmm_prep_{dataset_name}',
})
config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)


def _resolve_cache_path():
    if FLAGS.gmm_latent_cache_path:
        return FLAGS.gmm_latent_cache_path
    base, _ = os.path.splitext(FLAGS.gmm_save_path)
    return base + '_latents.npy'


def _logsumexp_np(x, axis=-1, keepdims=False):
    max_x = np.max(x, axis=axis, keepdims=True)
    out = max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))
    if keepdims:
        return out
    return np.squeeze(out, axis=axis)


def _posterior_np(latents_std, pi, mu, var):
    dim = latents_std.shape[-1]
    log_pi = np.log(np.maximum(pi, 1e-8))
    log_det = np.sum(np.log(var), axis=-1)
    diff = latents_std[:, None, :] - mu[None, :, :]
    quad = np.sum((diff * diff) / var[None, :, :], axis=-1)
    log_prob = log_pi[None, :] - 0.5 * (
        dim * np.log(2.0 * np.pi) + log_det[None, :] + quad)
    log_norm = _logsumexp_np(log_prob, axis=-1, keepdims=True)
    return np.exp(log_prob - log_norm).astype(np.float32)


def _normalize_rows(x, eps=1e-8):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + eps)


def _spherical_kmeans(directions, num_clusters, iters, seed, eps=1e-8):
    rng = np.random.default_rng(seed)
    num_examples = directions.shape[0]
    if num_examples == 0:
        centers = rng.normal(size=(num_clusters, directions.shape[1])).astype(np.float32)
        return _normalize_rows(centers, eps), np.zeros((0,), dtype=np.int32), np.zeros((num_clusters,), dtype=np.int32)

    replace = num_examples < num_clusters
    init_idx = rng.choice(num_examples, size=num_clusters, replace=replace)
    centers = directions[init_idx].astype(np.float32)
    centers = _normalize_rows(centers, eps)
    assignments = np.zeros((num_examples,), dtype=np.int32)

    for _ in range(iters):
        scores = directions @ centers.T
        assignments = np.argmax(scores, axis=-1).astype(np.int32)
        new_centers = np.zeros_like(centers)
        for idx in range(num_clusters):
            mask = assignments == idx
            if np.any(mask):
                new_centers[idx] = np.mean(directions[mask], axis=0)
            else:
                new_centers[idx] = directions[int(rng.integers(0, num_examples))]
        centers = _normalize_rows(new_centers, eps)

    counts = np.bincount(assignments, minlength=num_clusters).astype(np.int32)
    return centers.astype(np.float32), assignments, counts


def _assign_gmm_modes(latent_cache, gmm_state):
    num_examples = latent_cache.shape[0]
    mode_assignments = np.zeros((num_examples,), dtype=np.int32)
    posterior_entropy_sum = 0.0
    hard_counts = np.zeros((FLAGS.gmm_num_modes,), dtype=np.int64)

    for start in range(0, num_examples, FLAGS.gmm_em_chunk_size):
        stop = min(start + FLAGS.gmm_em_chunk_size, num_examples)
        q = _posterior_np(
            np.asarray(latent_cache[start:stop], dtype=np.float32),
            gmm_state['pi'],
            gmm_state['mu'],
            gmm_state['var'],
        )
        modes = np.argmax(q, axis=-1).astype(np.int32)
        mode_assignments[start:stop] = modes
        hard_counts += np.bincount(modes, minlength=FLAGS.gmm_num_modes)
        posterior_entropy_sum += float(
            np.sum(-q * np.log(np.maximum(q, 1e-8))))

    return mode_assignments, hard_counts.astype(np.float32), posterior_entropy_sum / num_examples


def _fit_angular_radius_stats(latent_cache, mode_assignments, gmm_state, latent_shape):
    num_modes = FLAGS.gmm_num_modes
    num_submodes = FLAGS.angular_num_submodes
    latent_dim = latent_cache.shape[-1]
    angular_centers = np.zeros((num_modes, num_submodes, latent_dim), dtype=np.float32)
    angular_counts = np.zeros((num_modes, num_submodes), dtype=np.float32)
    angular_active = np.zeros((num_modes, num_submodes), dtype=np.float32)
    angular_pi = np.ones((num_modes, num_submodes), dtype=np.float32) / num_submodes
    radius_log_mean = np.zeros((num_modes, num_submodes), dtype=np.float32)
    radius_log_std = np.ones((num_modes, num_submodes), dtype=np.float32) * FLAGS.radius_min_log_std
    radius_counts = np.zeros((num_modes, num_submodes), dtype=np.float32)
    radius_backoff = np.ones((num_modes, num_submodes), dtype=np.float32)
    base_to_data_l2_sum = 0.0
    base_to_data_count = 0

    global_log_radius_mean = 0.5 * np.log(float(latent_dim))
    global_log_radius_std = max(FLAGS.radius_min_log_std, 0.1)

    for mode in range(num_modes):
        indices = np.where(mode_assignments == mode)[0]
        if indices.shape[0] == 0:
            rng = np.random.default_rng(FLAGS.seed + 10000 + mode)
            centers = rng.normal(size=(num_submodes, latent_dim)).astype(np.float32)
            angular_centers[mode] = _normalize_rows(centers, FLAGS.local_eps)
            radius_log_mean[mode] = global_log_radius_mean
            radius_log_std[mode] = global_log_radius_std
            continue

        x_mode = np.asarray(latent_cache[indices], dtype=np.float32)
        mode_mu = gmm_state['mu'][mode]
        mode_var = np.maximum(gmm_state['var'][mode], FLAGS.local_eps)
        local = (x_mode - mode_mu[None, :]) * (mode_var[None, :] ** (-0.5 * FLAGS.local_eta))
        radius = np.linalg.norm(local, axis=-1).astype(np.float32)
        directions = _normalize_rows(local, FLAGS.local_eps)
        centers, angle_assignments, counts = _spherical_kmeans(
            directions,
            num_submodes,
            FLAGS.angular_kmeans_iters,
            FLAGS.seed + 1000 + mode,
            eps=FLAGS.local_eps,
        )
        angular_centers[mode] = centers
        angular_counts[mode] = counts.astype(np.float32)
        active = counts >= FLAGS.angular_min_cluster_size
        angular_active[mode] = active.astype(np.float32)

        active_counts = counts.astype(np.float32) * active.astype(np.float32)
        if np.sum(active_counts) > 0:
            angular_pi[mode] = active_counts / np.sum(active_counts)
        else:
            angular_pi[mode] = (counts.astype(np.float32) + 1.0)
            angular_pi[mode] /= np.sum(angular_pi[mode])

        log_radius = np.log(np.maximum(radius, FLAGS.local_eps))
        mode_log_mean = float(np.mean(log_radius))
        mode_log_std = max(float(np.std(log_radius)), FLAGS.radius_min_log_std)

        selected_centers = centers[angle_assignments]
        x_base_std = mode_mu[None, :] + (
            mode_var[None, :] ** (0.5 * FLAGS.local_eta)
        ) * radius[:, None] * selected_centers
        base_to_data_l2_sum += float(np.sum(np.linalg.norm(x_base_std - x_mode, axis=-1)))
        base_to_data_count += int(indices.shape[0])

        for submode in range(num_submodes):
            sub_mask = angle_assignments == submode
            sub_count = int(np.sum(sub_mask))
            radius_counts[mode, submode] = sub_count
            if sub_count >= FLAGS.angular_min_cluster_size:
                sub_log_radius = log_radius[sub_mask]
                radius_log_mean[mode, submode] = float(np.mean(sub_log_radius))
                radius_log_std[mode, submode] = max(
                    float(np.std(sub_log_radius)), FLAGS.radius_min_log_std)
                radius_backoff[mode, submode] = 0.0
            else:
                radius_log_mean[mode, submode] = mode_log_mean
                radius_log_std[mode, submode] = mode_log_std
                radius_backoff[mode, submode] = 1.0

    base_to_data_l2_mean = base_to_data_l2_sum / max(base_to_data_count, 1)
    return {
        'latent_shape': np.asarray(latent_shape, dtype=np.int32),
        'angular_centers': angular_centers,
        'angular_pi': angular_pi,
        'angular_counts': angular_counts,
        'angular_active': angular_active,
        'radius_log_mean': radius_log_mean,
        'radius_log_std': radius_log_std,
        'radius_counts': radius_counts,
        'radius_backoff': radius_backoff,
        'source_base_l2_mean_std_space': np.array(base_to_data_l2_mean, dtype=np.float32),
        'angular_inactive_count': np.array(
            np.sum(1.0 - angular_active), dtype=np.float32),
        'radius_backoff_count': np.array(np.sum(radius_backoff), dtype=np.float32),
    }


def main(_):
    np.random.seed(FLAGS.seed)
    save_dir = os.path.dirname(FLAGS.gmm_save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    cache_path = _resolve_cache_path()
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if jax.process_index() == 0:
        setup_wandb(
            {
                'dataset_name': FLAGS.dataset_name,
                'gmm_num_modes': FLAGS.gmm_num_modes,
                'angular_num_submodes': FLAGS.angular_num_submodes,
                'local_eta': FLAGS.local_eta,
                'gmm_em_iters': FLAGS.gmm_em_iters,
                'gmm_em_restarts': FLAGS.gmm_em_restarts,
                'gmm_pi_uniform_beta': FLAGS.gmm_pi_uniform_beta,
                'gmm_var_update': FLAGS.gmm_var_update,
                'gmm_var_beta': FLAGS.gmm_var_beta,
                'gmm_var_target': FLAGS.gmm_var_target,
                'gmm_var_floor': FLAGS.gmm_var_floor,
                'gmm_fit_samples': FLAGS.gmm_fit_samples,
            },
            **FLAGS.wandb,
        )

    dataset = get_dataset(
        FLAGS.dataset_name,
        FLAGS.batch_size,
        True,
        debug_overfit=False,
        data_dir=FLAGS.tfds_data_dir,
        repeat=False,
    )
    total_train_examples = get_num_examples(
        FLAGS.dataset_name,
        True,
        data_dir=FLAGS.tfds_data_dir,
    )
    target_examples = total_train_examples if FLAGS.gmm_fit_samples <= 0 else min(
        total_train_examples, FLAGS.gmm_fit_samples)

    vae = StableVAE.create()
    vae_encode = jax.jit(vae.encode)
    vae_rng = jax.random.PRNGKey(FLAGS.seed)

    latent_cache = None
    running_sum = None
    running_sumsq = None
    count = 0
    latent_shape = None

    print(f"Encoding {target_examples} train latents for source preprocessing.")
    while count < target_examples:
        try:
            batch_images, _ = next(dataset)
        except StopIteration:
            break

        vae_rng, vae_key = jax.random.split(vae_rng)
        latents = vae_encode(vae_key, batch_images)
        latent_shape = latents.shape[1:]
        latents_flat = np.asarray(jax.device_get(flatten_latents(latents)), dtype=np.float32)

        if latent_cache is None:
            latent_dim = latents_flat.shape[-1]
            latent_cache = np.lib.format.open_memmap(
                cache_path,
                mode='w+',
                dtype=np.float32,
                shape=(target_examples, latent_dim),
            )
            running_sum = np.zeros((latent_dim,), dtype=np.float64)
            running_sumsq = np.zeros((latent_dim,), dtype=np.float64)

        take = min(target_examples - count, latents_flat.shape[0])
        if take <= 0:
            break
        latents_slice = latents_flat[:take]
        latent_cache[count:count + take] = latents_slice
        running_sum += latents_slice.sum(axis=0, dtype=np.float64)
        running_sumsq += np.square(latents_slice, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += take

    if count == 0:
        raise ValueError("No train latents were extracted for source preprocessing.")

    if count < target_examples:
        latent_cache.flush()
        latent_cache = np.load(cache_path, mmap_mode='r+')
        latent_cache = latent_cache[:count]
        target_examples = count

    mean = (running_sum / target_examples).astype(np.float32)
    var = (running_sumsq / target_examples) - np.square(mean, dtype=np.float64)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)

    for start in range(0, target_examples, FLAGS.gmm_em_chunk_size):
        stop = min(start + FLAGS.gmm_em_chunk_size, target_examples)
        latent_cache[start:stop] = (
            latent_cache[start:stop] - mean[None, :]
        ) / (std[None, :] + FLAGS.gmm_standardize_eps)
    latent_cache.flush()

    print("Stage 1: fitting diagonal GMM with Generalized EM.")
    gmm_state = fit_diag_gmm(
        latents_std=latent_cache,
        num_modes=FLAGS.gmm_num_modes,
        em_iters=FLAGS.gmm_em_iters,
        restarts=FLAGS.gmm_em_restarts,
        seed=FLAGS.gmm_init_seed,
        chunk_size=FLAGS.gmm_em_chunk_size,
        var_floor=FLAGS.gmm_var_floor,
        use_kmeanspp=bool(FLAGS.gmm_kmeanspp_init),
        pi_uniform_beta=FLAGS.gmm_pi_uniform_beta,
        var_update=FLAGS.gmm_var_update,
        var_beta=FLAGS.gmm_var_beta,
        var_target=FLAGS.gmm_var_target,
        var_eps=FLAGS.gmm_var_eps,
        var_scale_min=FLAGS.gmm_var_scale_min,
        var_scale_max=FLAGS.gmm_var_scale_max,
        min_component_count=FLAGS.gmm_min_component_count,
    )

    print("Assigning GMM modes for angular/radius preprocessing.")
    mode_assignments, final_hard_counts, final_posterior_entropy = _assign_gmm_modes(
        latent_cache, gmm_state)
    print("Stage 2: fitting angular submodes and log-radius distributions.")
    source_code_state = _fit_angular_radius_stats(
        latent_cache, mode_assignments, gmm_state, latent_shape)

    stats_to_save = {
        'mean': mean,
        'std': std,
        'pi': gmm_state['pi'],
        'mu': gmm_state['mu'],
        'var': gmm_state['var'],
        'nll_trace': gmm_state['nll_trace'],
        'objective_trace': gmm_state['objective_trace'],
        'counts_trace': gmm_state['counts_trace'],
        'hard_counts_trace': gmm_state['hard_counts_trace'],
        'var_min_trace': gmm_state['var_min_trace'],
        'var_max_trace': gmm_state['var_max_trace'],
        'floor_frac_trace': gmm_state['floor_frac_trace'],
        'dead_count_trace': gmm_state['dead_count_trace'],
        'min_count_trace': gmm_state['min_count_trace'],
        'max_count_trace': gmm_state['max_count_trace'],
        'pi_kl_u_to_pi_trace': gmm_state['pi_kl_u_to_pi_trace'],
        'pi_entropy_trace': gmm_state['pi_entropy_trace'],
        'effective_components_trace': gmm_state['effective_components_trace'],
        'posterior_entropy_trace': gmm_state['posterior_entropy_trace'],
        'var_target_mse_trace': gmm_state['var_target_mse_trace'],
        'var_scale_min_trace': gmm_state['var_scale_min_trace'],
        'var_scale_max_trace': gmm_state['var_scale_max_trace'],
        'final_counts': gmm_state['final_counts'],
        'final_hard_counts': final_hard_counts,
        'final_posterior_entropy': np.array(final_posterior_entropy, dtype=np.float32),
        'restart_index': np.array(gmm_state['restart_index'], dtype=np.int32),
        'n_train': np.array(target_examples, dtype=np.int32),
        'standardize_eps': np.array(FLAGS.gmm_standardize_eps, dtype=np.float32),
        'local_eta': np.array(FLAGS.local_eta, dtype=np.float32),
        'gmm_pi_uniform_beta': np.array(FLAGS.gmm_pi_uniform_beta, dtype=np.float32),
        'gmm_var_beta': np.array(FLAGS.gmm_var_beta, dtype=np.float32),
        'gmm_var_target': np.array(FLAGS.gmm_var_target, dtype=np.float32),
        'gmm_var_floor': np.array(FLAGS.gmm_var_floor, dtype=np.float32),
        'gmm_var_eps': np.array(FLAGS.gmm_var_eps, dtype=np.float32),
        'gmm_var_scale_min': np.array(FLAGS.gmm_var_scale_min, dtype=np.float32),
        'gmm_var_scale_max': np.array(FLAGS.gmm_var_scale_max, dtype=np.float32),
        'angular_num_submodes': np.array(FLAGS.angular_num_submodes, dtype=np.int32),
        'angular_min_cluster_size': np.array(FLAGS.angular_min_cluster_size, dtype=np.int32),
        'radius_min_log_std': np.array(FLAGS.radius_min_log_std, dtype=np.float32),
        **source_code_state,
    }
    save_gmm_stats(FLAGS.gmm_save_path, stats_to_save)
    print(f"Saved source stats to {FLAGS.gmm_save_path}")

    if jax.process_index() == 0:
        import wandb

        for step, nll in enumerate(stats_to_save['nll_trace'], start=1):
            idx = step - 1
            wandb.log({
                'gmm/nll': float(nll),
                'gmm/objective': float(stats_to_save['objective_trace'][idx]),
                'gmm/pi_kl_u_to_pi': float(stats_to_save['pi_kl_u_to_pi_trace'][idx]),
                'gmm/pi_entropy': float(stats_to_save['pi_entropy_trace'][idx]),
                'gmm/dead_components': float(stats_to_save['dead_count_trace'][idx]),
                'gmm/min_count': float(stats_to_save['min_count_trace'][idx]),
                'gmm/max_count': float(stats_to_save['max_count_trace'][idx]),
                'gmm/effective_components': float(stats_to_save['effective_components_trace'][idx]),
                'gmm/posterior_entropy_mean': float(stats_to_save['posterior_entropy_trace'][idx]),
                'gmm/var_floor_hit_rate': float(stats_to_save['floor_frac_trace'][idx]),
                'gmm/var_target_mse': float(stats_to_save['var_target_mse_trace'][idx]),
                'gmm/var_scale_min': float(stats_to_save['var_scale_min_trace'][idx]),
                'gmm/var_scale_max': float(stats_to_save['var_scale_max_trace'][idx]),
            }, step=step)
        for idx in range(FLAGS.gmm_num_modes):
            wandb.log({
                f'gmm/pi_{idx}': float(stats_to_save['pi'][idx]),
                f'gmm/N_{idx}': float(stats_to_save['final_counts'][idx]),
                f'gmm/hard_N_{idx}': float(stats_to_save['final_hard_counts'][idx]),
            }, step=FLAGS.gmm_em_iters + 1)
        wandb.log({
            'latent/std_mean': float(np.mean(std)),
            'latent/mean_abs': float(np.mean(np.abs(mean))),
            'gmm/pi_min': float(np.min(stats_to_save['pi'])),
            'gmm/pi_max': float(np.max(stats_to_save['pi'])),
            'gmm/final_posterior_entropy': float(stats_to_save['final_posterior_entropy']),
            'angular/inactive_submodes': float(stats_to_save['angular_inactive_count']),
            'angular/min_count': float(np.min(stats_to_save['angular_counts'])),
            'radius/log_std_min': float(np.min(stats_to_save['radius_log_std'])),
            'radius/log_std_mean': float(np.mean(stats_to_save['radius_log_std'])),
            'radius/backoff_count': float(stats_to_save['radius_backoff_count']),
            'source_base/base_to_data_l2_mean_std_space': float(
                stats_to_save['source_base_l2_mean_std_space']),
        }, step=FLAGS.gmm_em_iters + 1)

    if not FLAGS.gmm_keep_latent_cache and os.path.exists(cache_path):
        os.remove(cache_path)


if __name__ == '__main__':
    app.run(main)
