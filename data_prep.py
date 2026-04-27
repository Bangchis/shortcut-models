import os
import json

from absl import app, flags
from ml_collections import config_flags
import ml_collections
import jax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gmm_utils import (
    diag_gmm_log_prob,
    fit_diag_gmm,
    flatten_latents,
    standardize_latents,
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
flags.DEFINE_string('gmm_save_path', '/kaggle/working/gmm_stats.npz', 'Output npz path for GMM stats.')
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
flags.DEFINE_integer('gmm_num_modes', 4, 'Number of diagonal GMM modes.')
flags.DEFINE_integer('gmm_em_iters', 100, 'Maximum EM iterations per restart.')
flags.DEFINE_integer('gmm_em_restarts', 3, 'Number of EM restarts.')
flags.DEFINE_integer('gmm_init_seed', 0, 'Seed for GMM initialization.')
flags.DEFINE_float('gmm_standardize_eps', 1e-6, 'Epsilon used for latent standardization.')
flags.DEFINE_float('gmm_var_floor', 1e-4, 'Minimum variance for every GMM dimension.')
flags.DEFINE_float('gmm_weight_prior', 1e-2, 'Pseudo-count added to each mixture component.')
flags.DEFINE_float(
    'gmm_var_mse_target_std',
    0.0,
    'Target sigma for per-mode mean-variance MSE penalty (sigma_target). '
    'Penalty: (1/K) sum_k (mean_j sigma_{k,j}^2 - sigma_target^2)^2. '
    'Set 0 (with weight 0) to disable.',
)
flags.DEFINE_float(
    'gmm_var_mse_weight',
    0.0,
    'beta_var in [0,1]: per-iter fraction of mean-variance correction toward target. '
    'beta_var = 2*lambda_var/(K*d). 0=off, 1=full pull each EM iter.',
)
flags.DEFINE_float(
    'gmm_pi_kl_weight',
    0.0,
    'beta_pi in [0,1]: exact reparam of DKL(U||pi) penalty as convex blend with uniform. '
    '0=pure Dirichlet+EM, 1=force pi exactly uniform.',
)
flags.DEFINE_integer('gmm_kmeanspp_init', 1, 'Whether to use kmeans++-style initialization.')
flags.DEFINE_integer('gmm_em_chunk_size', 1024, 'Chunk size for E-step/M-step accumulation.')
flags.DEFINE_integer('gmm_keep_latent_cache', 0, 'Whether to keep the latent cache file after fitting.')
flags.DEFINE_integer('gmm_valid_samples', -1, 'How many validation latents to use. -1 means full split.')
flags.DEFINE_integer('gmm_visual_subset', 2048, 'How many points to use for PCA/t-SNE visualizations.')
flags.DEFINE_string('gmm_wandb_level', 'full', 'GMM WandB logging level: "summary" or "full".')
flags.DEFINE_string('metrics_output_path', None, 'Optional JSON path for GMM metrics.')
flags.DEFINE_string('figures_dir', None, 'Optional directory to save diagnostic figures.')


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


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path, payload):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)


def _collect_latents(dataset, target_examples, vae_encode, vae_rng):
    latents = []
    count = 0
    while target_examples < 0 or count < target_examples:
        try:
            batch_images, _ = next(dataset)
        except StopIteration:
            break
        vae_rng, vae_key = jax.random.split(vae_rng)
        batch_latents = vae_encode(vae_key, batch_images)
        batch_flat = np.asarray(jax.device_get(flatten_latents(batch_latents)), dtype=np.float32)
        if target_examples >= 0:
            take = min(target_examples - count, batch_flat.shape[0])
            if take <= 0:
                break
            batch_flat = batch_flat[:take]
        latents.append(batch_flat)
        count += batch_flat.shape[0]
    if not latents:
        return np.zeros((0, 0), dtype=np.float32), vae_rng
    return np.concatenate(latents, axis=0), vae_rng


def _save_figure(fig, figures_dir, name):
    if not figures_dir:
        return None
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f'{name}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


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
                'gmm_em_iters': FLAGS.gmm_em_iters,
                'gmm_em_restarts': FLAGS.gmm_em_restarts,
                'gmm_var_floor': FLAGS.gmm_var_floor,
                'gmm_weight_prior': FLAGS.gmm_weight_prior,
                'gmm_var_mse_target_std': FLAGS.gmm_var_mse_target_std,
                'gmm_var_mse_weight': FLAGS.gmm_var_mse_weight,
                'gmm_pi_kl_weight': FLAGS.gmm_pi_kl_weight,
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
    dataset_valid = get_dataset(
        FLAGS.dataset_name,
        FLAGS.batch_size,
        False,
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

    print(f"Encoding {target_examples} train latents for GMM preprocessing.")
    while count < target_examples:
        try:
            batch_images, _ = next(dataset)
        except StopIteration:
            break

        vae_rng, vae_key = jax.random.split(vae_rng)
        latents = vae_encode(vae_key, batch_images)
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
        raise ValueError("No train latents were extracted for GMM preprocessing.")

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

    print("Fitting diagonal GMM with EM.")
    gmm_state = fit_diag_gmm(
        latents_std=latent_cache,
        num_modes=FLAGS.gmm_num_modes,
        em_iters=FLAGS.gmm_em_iters,
        restarts=FLAGS.gmm_em_restarts,
        seed=FLAGS.gmm_init_seed,
        chunk_size=FLAGS.gmm_em_chunk_size,
        var_floor=FLAGS.gmm_var_floor,
        weight_prior=FLAGS.gmm_weight_prior,
        use_kmeanspp=bool(FLAGS.gmm_kmeanspp_init),
        var_mse_target_std=FLAGS.gmm_var_mse_target_std,
        var_mse_weight=FLAGS.gmm_var_mse_weight,
        pi_kl_weight=FLAGS.gmm_pi_kl_weight,
    )

    stats_to_save = {
        'mean': mean,
        'std': std,
        'pi': gmm_state['pi'],
        'mu': gmm_state['mu'],
        'var': gmm_state['var'],
        'nll_trace': gmm_state['nll_trace'],
        'counts_trace': gmm_state['counts_trace'],
        'var_min_trace': gmm_state['var_min_trace'],
        'var_max_trace': gmm_state['var_max_trace'],
        'final_counts': gmm_state['final_counts'],
        'restart_index': np.array(gmm_state['restart_index'], dtype=np.int32),
        'n_train': np.array(target_examples, dtype=np.int32),
        'standardize_eps': np.array(FLAGS.gmm_standardize_eps, dtype=np.float32),
    }
    save_gmm_stats(FLAGS.gmm_save_path, stats_to_save)
    print(f"Saved GMM stats to {FLAGS.gmm_save_path}")

    valid_latents, vae_rng = _collect_latents(
        dataset_valid,
        FLAGS.gmm_valid_samples,
        vae_encode,
        vae_rng,
    )
    if valid_latents.shape[0] == 0:
        raise ValueError("No validation latents were extracted for GMM evaluation.")

    valid_latents_std = standardize_latents(
        valid_latents,
        stats_to_save['mean'],
        stats_to_save['std'],
        FLAGS.gmm_standardize_eps,
    )
    train_latents_std = np.asarray(latent_cache[:min(target_examples, max(FLAGS.gmm_visual_subset, 1))], dtype=np.float32)

    log_pi = jax.numpy.asarray(np.log(np.maximum(stats_to_save['pi'], 1e-8)), dtype=jax.numpy.float32)
    mu_device = jax.numpy.asarray(stats_to_save['mu'], dtype=jax.numpy.float32)
    var_device = jax.numpy.asarray(stats_to_save['var'], dtype=jax.numpy.float32)

    def _mean_nll(latents_std):
        chunk = jax.numpy.asarray(latents_std, dtype=jax.numpy.float32)
        log_prob = diag_gmm_log_prob(chunk, log_pi, mu_device, var_device)
        log_norm = jax.scipy.special.logsumexp(log_prob, axis=-1)
        return float(-np.mean(np.asarray(jax.device_get(log_norm))))

    valid_log_prob = np.asarray(jax.device_get(diag_gmm_log_prob(
        jax.numpy.asarray(valid_latents_std, dtype=jax.numpy.float32),
        log_pi,
        mu_device,
        var_device,
    )))
    valid_log_norm = np.asarray(jax.device_get(jax.scipy.special.logsumexp(valid_log_prob, axis=-1, keepdims=True)))
    valid_q = np.exp(valid_log_prob - valid_log_norm)
    train_vis_log_prob = np.asarray(jax.device_get(diag_gmm_log_prob(
        jax.numpy.asarray(train_latents_std, dtype=jax.numpy.float32),
        log_pi,
        mu_device,
        var_device,
    )))
    train_vis_log_norm = np.asarray(
        jax.device_get(jax.scipy.special.logsumexp(train_vis_log_prob, axis=-1, keepdims=True))
    )
    train_vis_q = np.exp(train_vis_log_prob - train_vis_log_norm)

    final_counts = np.asarray(stats_to_save['final_counts'], dtype=np.float64)
    occupancy = final_counts / np.maximum(np.sum(final_counts), 1e-8)
    uniform_fraction = 1.0 / max(int(FLAGS.gmm_num_modes), 1)
    occupancy_entropy = float(-np.sum(occupancy * np.log(np.maximum(occupancy, 1e-8))))
    effective_num_components = float(np.exp(occupancy_entropy))
    pi = np.asarray(stats_to_save['pi'], dtype=np.float64)
    pi_entropy = float(-np.sum(pi * np.log(np.maximum(pi, 1e-8))))
    pi_effective_num_components = float(np.exp(pi_entropy))
    posterior_entropy = -np.sum(valid_q * np.log(np.maximum(valid_q, 1e-8)), axis=-1)
    max_entropy = float(np.log(max(FLAGS.gmm_num_modes, 1))) if FLAGS.gmm_num_modes > 1 else 1.0
    posterior_entropy_normalized = posterior_entropy / max(max_entropy, 1e-8)
    posterior_top1_prob = np.max(valid_q, axis=-1)
    q_sorted = np.sort(valid_q, axis=-1)
    posterior_margin = q_sorted[:, -1] - q_sorted[:, -2] if valid_q.shape[1] > 1 else q_sorted[:, -1]
    pairwise_center_distance = np.linalg.norm(
        stats_to_save['mu'][:, None, :] - stats_to_save['mu'][None, :, :],
        axis=-1,
    )
    if pairwise_center_distance.shape[0] > 1:
        pairwise_non_diag = pairwise_center_distance[
            ~np.eye(pairwise_center_distance.shape[0], dtype=bool)
        ]
    else:
        pairwise_non_diag = np.asarray([], dtype=np.float64)
    var = np.asarray(stats_to_save['var'], dtype=np.float64)
    component_var_mean = np.mean(var, axis=1)
    component_std_mean = np.mean(np.sqrt(np.maximum(var, 0.0)), axis=1)
    floor_mask = var <= (FLAGS.gmm_var_floor * 1.0001)
    train_nll = float(stats_to_save['nll_trace'][-1])
    valid_nll = _mean_nll(valid_latents_std)
    nll_trace = np.asarray(stats_to_save['nll_trace'], dtype=np.float64)
    nll_step_delta = np.diff(nll_trace) if nll_trace.shape[0] > 1 else np.asarray([], dtype=np.float64)
    target_var = float(FLAGS.gmm_var_mse_target_std) ** 2
    var_target_abs_error = np.abs(component_var_mean - target_var)
    gmm_metrics = {
        'gmm_num_modes': int(FLAGS.gmm_num_modes),
        'train_nll': train_nll,
        'valid_nll': valid_nll,
        'train_valid_nll_gap': float(valid_nll - train_nll),
        'nll_improvement': float(nll_trace[0] - nll_trace[-1]) if nll_trace.size > 0 else 0.0,
        'nll_increase_step_count': int(np.sum(nll_step_delta > 1e-5)),
        'dead_component_count': int(np.sum(final_counts < 1.0)),
        'near_dead_component_count': int(np.sum(occupancy < (0.25 * uniform_fraction))),
        'under_half_uniform_component_count': int(np.sum(occupancy < (0.5 * uniform_fraction))),
        'min_component_count': float(np.min(final_counts)),
        'max_component_count': float(np.max(final_counts)),
        'min_component_fraction': float(np.min(occupancy)),
        'p05_component_fraction': float(np.percentile(occupancy, 5)),
        'median_component_fraction': float(np.percentile(occupancy, 50)),
        'max_component_fraction': float(np.max(occupancy)),
        'component_fraction_std': float(np.std(occupancy)),
        'occupancy_entropy': occupancy_entropy,
        'occupancy_entropy_normalized': float(occupancy_entropy / max(max_entropy, 1e-8)),
        'effective_num_components': effective_num_components,
        'effective_component_fraction': float(effective_num_components / max(int(FLAGS.gmm_num_modes), 1)),
        'pi_min': float(np.min(pi)),
        'pi_max': float(np.max(pi)),
        'pi_entropy': pi_entropy,
        'pi_entropy_normalized': float(pi_entropy / max(max_entropy, 1e-8)),
        'pi_effective_num_components': pi_effective_num_components,
        'posterior_entropy_mean': float(np.mean(posterior_entropy)),
        'posterior_entropy_std': float(np.std(posterior_entropy)),
        'posterior_entropy_min': float(np.min(posterior_entropy)),
        'posterior_entropy_max': float(np.max(posterior_entropy)),
        'posterior_entropy_p05': float(np.percentile(posterior_entropy, 5)),
        'posterior_entropy_p50': float(np.percentile(posterior_entropy, 50)),
        'posterior_entropy_p95': float(np.percentile(posterior_entropy, 95)),
        'posterior_entropy_normalized_mean': float(np.mean(posterior_entropy_normalized)),
        'posterior_top1_prob_mean': float(np.mean(posterior_top1_prob)),
        'posterior_top1_prob_p05': float(np.percentile(posterior_top1_prob, 5)),
        'posterior_top1_prob_p50': float(np.percentile(posterior_top1_prob, 50)),
        'posterior_top1_margin_mean': float(np.mean(posterior_margin)),
        'posterior_top1_margin_p05': float(np.percentile(posterior_margin, 5)),
        'var_floor_hit_rate': float(np.mean(stats_to_save['var'] <= (FLAGS.gmm_var_floor * 1.0001))),
        'var_floor_component_count': int(np.sum(np.any(floor_mask, axis=1))),
        'var_mean': float(np.mean(var)),
        'var_min': float(np.min(var)),
        'var_p05': float(np.percentile(var, 5)),
        'var_median': float(np.percentile(var, 50)),
        'var_max': float(np.max(var)),
        'component_var_mean_min': float(np.min(component_var_mean)),
        'component_var_mean_max': float(np.max(component_var_mean)),
        'component_var_mean_std': float(np.std(component_var_mean)),
        'component_std_mean_min': float(np.min(component_std_mean)),
        'component_std_mean_max': float(np.max(component_std_mean)),
        'component_std_mean_std': float(np.std(component_std_mean)),
        'var_target_abs_error_mean': float(np.mean(var_target_abs_error)),
        'var_target_abs_error_max': float(np.max(var_target_abs_error)),
        'center_distance_min': float(np.min(pairwise_non_diag)) if pairwise_non_diag.size > 0 else 0.0,
        'center_distance_mean': float(np.mean(pairwise_non_diag)) if pairwise_non_diag.size > 0 else 0.0,
        'center_distance_p05': float(np.percentile(pairwise_non_diag, 5)) if pairwise_non_diag.size > 0 else 0.0,
        'n_train_used': int(target_examples),
        'n_valid_used': int(valid_latents.shape[0]),
        'gmm_save_path': FLAGS.gmm_save_path,
    }

    figure_paths = {}
    figures_dir = FLAGS.figures_dir
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        subset_n = min(FLAGS.gmm_visual_subset, valid_latents_std.shape[0], train_latents_std.shape[0])
        if subset_n > 1:
            train_labels = np.argmax(train_vis_q[:subset_n], axis=-1).astype(np.int32)
            valid_labels = np.argmax(valid_q[:subset_n], axis=-1).astype(np.int32)
            plot_points = np.concatenate(
                [train_latents_std[:subset_n], valid_latents_std[:subset_n], stats_to_save['mu']],
                axis=0,
            )
            pca = PCA(n_components=2, random_state=0)
            coords = pca.fit_transform(plot_points)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(
                coords[:subset_n, 0],
                coords[:subset_n, 1],
                c=train_labels,
                cmap='tab20',
                s=8,
                alpha=0.20,
                marker='o',
                label='train',
            )
            valid_coords = coords[subset_n:2 * subset_n]
            ax.scatter(
                valid_coords[:, 0],
                valid_coords[:, 1],
                c=valid_labels,
                cmap='tab20',
                s=14,
                alpha=0.65,
                marker='^',
                label='valid',
            )
            center_coords = coords[2 * subset_n:]
            ax.scatter(
                center_coords[:, 0],
                center_coords[:, 1],
                c=np.arange(stats_to_save['mu'].shape[0]),
                cmap='tab20',
                s=90,
                marker='X',
                edgecolors='black',
                linewidths=0.8,
                label='centers',
            )
            ax.set_title(f'PCA GMM K={FLAGS.gmm_num_modes}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.text(
                0.02,
                0.98,
                'Color = cluster, marker = split',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'),
            )
            ax.legend(loc='best')
            figure_paths['pca'] = _save_figure(fig, figures_dir, 'pca')

            tsne_subset = min(1024, subset_n)
            train_tsne_labels = train_labels[:tsne_subset]
            valid_tsne_labels = valid_labels[:tsne_subset]
            tsne_input = np.concatenate(
                [train_latents_std[:tsne_subset], valid_latents_std[:tsne_subset], stats_to_save['mu']],
                axis=0,
            )
            tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
            tsne_coords = tsne.fit_transform(tsne_input)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(
                tsne_coords[:tsne_subset, 0],
                tsne_coords[:tsne_subset, 1],
                c=train_tsne_labels,
                cmap='tab20',
                s=8,
                alpha=0.20,
                marker='o',
                label='train',
            )
            ax.scatter(
                tsne_coords[tsne_subset:2 * tsne_subset, 0],
                tsne_coords[tsne_subset:2 * tsne_subset, 1],
                c=valid_tsne_labels,
                cmap='tab20',
                s=14,
                alpha=0.65,
                marker='^',
                label='valid',
            )
            ax.scatter(
                tsne_coords[2 * tsne_subset:, 0],
                tsne_coords[2 * tsne_subset:, 1],
                c=np.arange(stats_to_save['mu'].shape[0]),
                cmap='tab20',
                s=90,
                marker='X',
                edgecolors='black',
                linewidths=0.8,
                label='centers',
            )
            ax.set_title(f't-SNE GMM K={FLAGS.gmm_num_modes}')
            ax.text(
                0.02,
                0.98,
                'Color = cluster, marker = split',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'),
            )
            ax.legend(loc='best')
            figure_paths['tsne'] = _save_figure(fig, figures_dir, 'tsne')

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(np.arange(occupancy.shape[0]), occupancy)
        ax.set_title('Component Occupancy')
        ax.set_xlabel('Component')
        ax.set_ylabel('Fraction')
        ax.axhline(uniform_fraction, color='black', linestyle='--', linewidth=1.0, label='uniform')
        ax.axhline(0.25 * uniform_fraction, color='red', linestyle=':', linewidth=1.0, label='near-dead')
        ax.text(
            0.02,
            0.95,
            (
                f'min_frac={np.min(occupancy):.4f}\n'
                f'max_frac={np.max(occupancy):.4f}\n'
                f'eff_K={effective_num_components:.2f}/{FLAGS.gmm_num_modes}'
            ),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'),
        )
        ax.legend(loc='upper right')
        figure_paths['occupancy'] = _save_figure(fig, figures_dir, 'occupancy')

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(posterior_entropy, bins=40, range=(0.0, max_entropy))
        ax.set_title('Posterior Entropy')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Count')
        ax.set_xlim(0.0, max_entropy)
        ax.axvline(float(np.mean(posterior_entropy)), color='red', linestyle='--', linewidth=1.5, label='mean')
        ax.text(
            0.02,
            0.95,
            (
                f'min={np.min(posterior_entropy):.4f}\n'
                f'p50={np.percentile(posterior_entropy, 50):.4f}\n'
                f'max={np.max(posterior_entropy):.4f}\n'
                f'max_theory=log(K)={max_entropy:.4f}'
            ),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'),
        )
        ax.legend(loc='upper right')
        figure_paths['posterior_entropy'] = _save_figure(fig, figures_dir, 'posterior_entropy')

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(posterior_entropy_normalized, bins=40, range=(0.0, 1.0))
        ax.set_title('Normalized Posterior Entropy')
        ax.set_xlabel('Entropy / log(K)')
        ax.set_ylabel('Count')
        ax.set_xlim(0.0, 1.0)
        ax.axvline(float(np.mean(posterior_entropy_normalized)), color='red', linestyle='--', linewidth=1.5, label='mean')
        ax.legend(loc='upper right')
        figure_paths['posterior_entropy_normalized'] = _save_figure(fig, figures_dir, 'posterior_entropy_normalized')

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(posterior_margin, bins=40, range=(0.0, 1.0))
        ax.set_title('Posterior Top-1 Margin')
        ax.set_xlabel('q_max - q_second')
        ax.set_ylabel('Count')
        ax.set_xlim(0.0, 1.0)
        ax.axvline(float(np.mean(posterior_margin)), color='red', linestyle='--', linewidth=1.5, label='mean')
        ax.legend(loc='upper left')
        figure_paths['posterior_top1_margin'] = _save_figure(fig, figures_dir, 'posterior_top1_margin')

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pairwise_center_distance, cmap='viridis')
        ax.set_title('Center Distance Heatmap')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        figure_paths['center_distance'] = _save_figure(fig, figures_dir, 'center_distance')

    _write_json(
        FLAGS.metrics_output_path,
        {
            **gmm_metrics,
            'figure_paths': figure_paths,
        },
    )

    if jax.process_index() == 0:
        import wandb

        if FLAGS.gmm_wandb_level not in ('summary', 'full'):
            raise ValueError('--gmm_wandb_level must be "summary" or "full".')
        if FLAGS.gmm_wandb_level == 'full':
            for step, nll in enumerate(stats_to_save['nll_trace'], start=1):
                wandb.log({
                    'gmm/nll': float(nll),
                    'gmm/var_min': float(stats_to_save['var_min_trace'][step - 1]),
                    'gmm/var_max': float(stats_to_save['var_max_trace'][step - 1]),
                }, step=step)
            for idx in range(FLAGS.gmm_num_modes):
                wandb.log({
                    f'gmm/pi_{idx}': float(stats_to_save['pi'][idx]),
                    f'gmm/N_{idx}': float(stats_to_save['final_counts'][idx]),
                }, step=FLAGS.gmm_em_iters + 1)
            wandb.log({
                'latent/std_mean': float(np.mean(std)),
                'latent/mean_abs': float(np.mean(np.abs(mean))),
            }, step=FLAGS.gmm_em_iters + 1)
        wandb.log({f'gmm_eval/{k}': v for k, v in gmm_metrics.items() if isinstance(v, (int, float))},
                  step=FLAGS.gmm_em_iters + 2)
        if FLAGS.gmm_wandb_level == 'full':
            for fig_name, fig_path in figure_paths.items():
                if fig_path:
                    wandb.log({f'gmm_fig/{fig_name}': wandb.Image(fig_path)}, step=FLAGS.gmm_em_iters + 2)

    if not FLAGS.gmm_keep_latent_cache and os.path.exists(cache_path):
        os.remove(cache_path)


if __name__ == '__main__':
    app.run(main)
