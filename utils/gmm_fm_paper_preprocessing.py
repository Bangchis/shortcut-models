"""
Preprocessing orchestration for GMM-FM paper implementation.

Coordinates the complete pipeline:
1. Latent cache creation
2. GMM prior fitting
3. Cluster cache creation
4. Validation and logging
"""

import os
from pathlib import Path
from typing import Callable, Tuple
import jax

from utils.latent_cache import (
    LatentCacheConfig,
    create_latent_cache,
    load_latent_cache,
    get_cache_info
)
from utils.cluster_cache import (
    ClusterCacheConfig,
    create_cluster_cache,
    load_cluster_cache,
    get_cluster_stats
)
from utils.gmm_preprocessing import GMMPreprocessConfig, preprocess_gmm_prior
from utils.gmm_prior import load_prior_npz, GMMPrior


def run_gmm_fm_paper_preprocessing(
    FLAGS,
    get_dataset_fn: Callable,
    encode_fn: Callable,
    local_batch_size: int,
    verbose: bool = True
) -> Tuple[str, str, str]:
    """
    Complete preprocessing pipeline for gmm-fm-paper.

    Steps:
    1. Determine cache paths
    2. Check if caches exist (unless force_recache)
    3. Create latent cache (or load existing) - uses encode_posterior() for (μ, logσ²)
    4. Fit GMM prior (or load existing) - uses μ only
    5. Create cluster cache (or load existing) - uses μ only
    6. Validate caches
    7. Log to W&B

    Args:
        FLAGS: Configuration flags
        get_dataset_fn: Function to get dataset iterator
        encode_fn: VAE encoding function (should return (mu, logvar) in posterior mode)
        local_batch_size: Batch size
        verbose: Show progress

    Returns:
        latent_cache_path: Path to latents.npy (mu file in posterior mode)
        prior_path: Path to prior.npz
        cluster_cache_path: Path to clusters.npz
    """
    if verbose and jax.process_index() == 0:
        print("\n" + "="*80)
        print("GMM-FM Paper Preprocessing Pipeline")
        print("="*80)

    # 1. Determine cache directory (align with LatentCacheConfig pattern)
    base_dir = FLAGS.save_dir if FLAGS.save_dir else os.getcwd()
    cache_dir = Path(base_dir) / 'caches'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(cache_dir)

    if verbose and jax.process_index() == 0:
        print(f"Cache directory: {cache_dir}")

    # 2. Get epsilon scale from FLAGS
    vae_epsilon_scale = float(FLAGS.model.get('vae_epsilon_scale', 1.0))

    # 3. Latent cache configuration (use posterior mode by default)
    latent_cfg = LatentCacheConfig(
        cache_dir=cache_dir,
        dataset_name=FLAGS.dataset_name,
        vae_epsilon_scale=vae_epsilon_scale,
        dtype=FLAGS.model.get('gmm_paper_cache_dtype', 'float16'),
        overwrite=FLAGS.model.get('gmm_paper_force_recache', False),
        cache_posterior=True  # Cache (μ, logσ²) for stochastic sampling
    )

    latent_cache_path = latent_cfg.get_cache_path()

    # 4. Create or load latent cache
    if not Path(latent_cache_path).exists() or latent_cfg.overwrite:
        if verbose and jax.process_index() == 0:
            print("\n[Step 1/3] Creating latent cache...")

        # Get dataset info
        dataset = get_dataset_fn(FLAGS.dataset_name, local_batch_size, is_train=True, debug_overfit=FLAGS.debug_overfit)

        # Determine number of examples
        if FLAGS.debug_overfit > 0:
            num_examples = FLAGS.debug_overfit
        else:
            # Estimate from dataset (TODO: get exact count from TFDS)
            # For now, use a heuristic based on dataset name
            num_examples_map = {
                'imagenet256': 1281167,
                'celebahq256': 30000,
                'ffhq256': 70000,
            }
            num_examples = num_examples_map.get(FLAGS.dataset_name, 100000)

        # Get example latent shape
        example_images, _ = next(dataset)
        dataset = get_dataset_fn(FLAGS.dataset_name, local_batch_size, is_train=True, debug_overfit=FLAGS.debug_overfit)  # Reset
        example_latent = encode_fn(jax.random.PRNGKey(0), example_images[:1])

        # Handle tuple return from encode_posterior (mu, logvar)
        if isinstance(example_latent, tuple):
            example_latent_shape = example_latent[0].shape[1:]  # Use mu shape
        else:
            example_latent_shape = example_latent.shape[1:]  # (H, W, C)

        # Create cache
        latent_cache_path = create_latent_cache(
            dataset_iterator=dataset,
            encode_fn=encode_fn,
            cfg=latent_cfg,
            num_examples=num_examples,
            local_batch_size=local_batch_size,
            example_latent_shape=example_latent_shape,
            verbose=verbose
        )
    else:
        if verbose and jax.process_index() == 0:
            print(f"\n[Step 1/3] Loading existing latent cache: {latent_cache_path}")

    # Load cache info
    cache_info = get_cache_info(latent_cache_path)
    if verbose and jax.process_index() == 0:
        print(f"  Latent cache shape: {cache_info['shape']}")
        print(f"  Cache mode: {'Posterior (μ, logσ²)' if cache_info.get('cache_posterior', False) else 'Legacy (fixed latents)'}")
        print(f"  VAE epsilon scale (τ): {cache_info['vae_epsilon_scale']}")

    # 5. GMM prior configuration
    gmm_cfg = GMMPreprocessConfig.from_flags(FLAGS)
    prior_path = gmm_cfg.get_prior_path()

    # 6. Fit or load GMM prior
    if not Path(prior_path).exists() or FLAGS.model.get('gmm_force_refit', False):
        if verbose and jax.process_index() == 0:
            print("\n[Step 2/3] Fitting GMM prior...")

        # For GMM fitting, we can use latent cache directly
        latent_cache = load_latent_cache(latent_cache_path, mode='r')

        # Use existing GMM preprocessing (will load from cache)
        # For now, re-encode subset (not ideal but works)
        dataset_for_gmm = get_dataset_fn(FLAGS.dataset_name, local_batch_size, is_train=True, debug_overfit=FLAGS.debug_overfit)

        def encode_to_latent(batch_images, key):
            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                result = encode_fn(key, batch_images)
                # If posterior mode, use μ only for GMM fitting (deterministic)
                batch_images = result[0] if isinstance(result, tuple) else result
            if 'latent' in FLAGS.dataset_name and batch_images.shape[-1] > 4:
                batch_images = batch_images[..., batch_images.shape[-1] // 2:]
            return batch_images

        gmm_prior = preprocess_gmm_prior(
            cfg=gmm_cfg,
            get_dataset_fn=get_dataset_fn,
            encode_fn=encode_to_latent,
            local_batch_size=local_batch_size,
            debug_overfit=FLAGS.debug_overfit,
            verbose=verbose
        )
    else:
        if verbose and jax.process_index() == 0:
            print(f"\n[Step 2/3] Loading existing GMM prior: {prior_path}")
        gmm_prior = load_prior_npz(prior_path)

    if verbose and jax.process_index() == 0:
        print(f"  GMM components: K={gmm_prior.K}, D={gmm_prior.D}")

    # 7. Cluster cache configuration
    cluster_cfg = ClusterCacheConfig(
        latent_cache_path=latent_cache_path,
        prior_path=prior_path,
        save_path=str(Path(cache_dir) / f"clusters_{FLAGS.dataset_name}_K{gmm_prior.K}.npz"),
        overwrite=latent_cfg.overwrite or FLAGS.model.get('gmm_force_refit', False)
    )

    cluster_cache_path = cluster_cfg.save_path

    # 8. Create or load cluster cache
    if not Path(cluster_cache_path).exists() or cluster_cfg.overwrite:
        if verbose and jax.process_index() == 0:
            print("\n[Step 3/3] Creating cluster cache...")

        # Load latent cache
        latent_cache = load_latent_cache(latent_cache_path, mode='r')

        # Create cluster cache
        cluster_cache_path = create_cluster_cache(
            latent_cache=latent_cache,
            gmm_prior=gmm_prior,
            cfg=cluster_cfg,
            batch_size=1024,
            verbose=verbose
        )
    else:
        if verbose and jax.process_index() == 0:
            print(f"\n[Step 3/3] Loading existing cluster cache: {cluster_cache_path}")

    # Load cluster cache for stats
    cluster_cache = load_cluster_cache(cluster_cache_path)
    cluster_stats = get_cluster_stats(cluster_cache)

    if verbose and jax.process_index() == 0:
        print(f"  Cluster stats:")
        print(f"    Total samples: {cluster_stats['N']}")
        print(f"    Clusters: {cluster_stats['K']}")
        print(f"    Empty clusters: {cluster_stats['empty_clusters']}")
        print(f"    Count range: [{cluster_stats['min_count']}, {cluster_stats['max_count']}]")
        print(f"    Mean count: {cluster_stats['mean_count']:.1f} ± {cluster_stats['std_count']:.1f}")

    # 9. Log to W&B (only process 0)
    if jax.process_index() == 0:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'gmm_paper/latent_cache_size_mb': Path(latent_cache_path).stat().st_size / 1024 / 1024,
                    'gmm_paper/num_examples': cluster_stats['N'],
                    'gmm_paper/num_clusters': cluster_stats['K'],
                    'gmm_paper/empty_clusters': cluster_stats['empty_clusters'],
                    'gmm_paper/min_cluster_count': cluster_stats['min_count'],
                    'gmm_paper/max_cluster_count': cluster_stats['max_count'],
                    'gmm_paper/mean_cluster_count': cluster_stats['mean_count'],
                    'gmm_paper/vae_epsilon_scale': vae_epsilon_scale,
                })

                # Log cluster distribution histogram
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(cluster_stats['K']), cluster_stats['counts'])
                ax.set_xlabel('Cluster ID')
                ax.set_ylabel('Count')
                ax.set_title(f'Cluster Distribution (K={cluster_stats["K"]})')
                wandb.log({'gmm_paper/cluster_histogram': wandb.Image(fig)})
                plt.close(fig)

        except ImportError:
            pass

    if verbose and jax.process_index() == 0:
        print("\n" + "="*80)
        print("Preprocessing complete!")
        print("="*80 + "\n")

    return latent_cache_path, prior_path, cluster_cache_path
