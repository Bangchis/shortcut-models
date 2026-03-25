import os

from absl import app, flags
from ml_collections import config_flags
import ml_collections
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
flags.DEFINE_integer('gmm_kmeanspp_init', 1, 'Whether to use kmeans++-style initialization.')
flags.DEFINE_integer('gmm_em_chunk_size', 1024, 'Chunk size for E-step/M-step accumulation.')
flags.DEFINE_integer('gmm_keep_latent_cache', 0, 'Whether to keep the latent cache file after fitting.')


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

    if jax.process_index() == 0:
        import wandb

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

    if not FLAGS.gmm_keep_latent_cache and os.path.exists(cache_path):
        os.remove(cache_path)


if __name__ == '__main__':
    app.run(main)
