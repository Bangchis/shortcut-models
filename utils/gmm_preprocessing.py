"""
GMM-FM preprocessing pipeline for training.

This module handles the complete GMM prior fitting workflow:
- Cache management
- Dataset size estimation
- Init point collection
- EM fitting with streaming data
- Saving and W&B logging
- PCA visualization
"""

from __future__ import annotations

import os
from typing import Optional, Callable, Tuple, Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tqdm

from .gmm_em import EMConfig, fit_gmm_em_streaming
from .gmm_prior import GMMPrior, load_prior_npz, save_prior_npz
from .gmm_visualization import create_and_log_pca_visualization


@dataclass
class GMMPreprocessConfig:
    """Configuration for GMM preprocessing pipeline."""
    # Core GMM settings
    K: int  # number of components
    em_subset: float = 0.1  # fraction of dataset to use
    em_max_iters: int = 50
    em_tol: float = 1e-4
    em_patience: int = 3
    var_floor: float = 1e-5

    # Cache settings
    cache_path: str = ''  # if empty, auto-generated
    force_refit: bool = False

    # W&B settings
    log_artifact: bool = True
    pca_max_points: int = 2000

    # Runtime settings
    dataset_name: str = ''
    save_dir: Optional[str] = None
    seed: int = 42

    @classmethod
    def from_flags(cls, FLAGS) -> 'GMMPreprocessConfig':
        """Create config from FLAGS object."""
        return cls(
            K=int(FLAGS.model['gmm_K']),
            em_subset=float(FLAGS.model.get('gmm_em_subset', 0.1)),
            em_max_iters=int(FLAGS.model.get('gmm_em_max_iters', 50)),
            em_tol=float(FLAGS.model.get('gmm_em_tol', 1e-4)),
            em_patience=int(FLAGS.model.get('gmm_em_patience', 3)),
            var_floor=float(FLAGS.model.get('gmm_var_floor', 1e-5)),
            cache_path=str(FLAGS.model.get('gmm_cache_path', '') or ''),
            force_refit=bool(FLAGS.model.get('gmm_force_refit', False)),
            log_artifact=bool(FLAGS.model.get('gmm_log_artifact', True)),
            pca_max_points=int(FLAGS.model.get('gmm_pca_max_points', 2000)),
            dataset_name=FLAGS.dataset_name,
            save_dir=FLAGS.save_dir,
            seed=FLAGS.seed,
        )

    def get_prior_path(self) -> str:
        """
        Get path to GMM prior cache file.

        Returns:
            Absolute path to prior .npz file
        """
        if self.cache_path:
            return self.cache_path

        base_dir = self.save_dir if self.save_dir is not None else os.getcwd()
        return os.path.join(
            base_dir,
            f"gmm_prior_{self.dataset_name}_K{self.K}.npz"
        )


def get_dataset_size(dataset_name: str, is_train: bool = True) -> int:
    """
    Best-effort dataset size lookup using TFDS.

    Args:
        dataset_name: Dataset name (may use repo-specific aliases)
        is_train: Whether to look up train or validation split

    Returns:
        Number of examples, or 100000 as fallback
    """
    # Map common aliases used in this repo to TFDS builder names
    if 'imagenet' in dataset_name:
        tfds_name = 'imagenet2012'
        split = 'train' if is_train else 'validation'
    elif 'celebahq256' in dataset_name:
        tfds_name = 'celebahq256'
        split = 'train'
    else:
        tfds_name = dataset_name
        split = 'train' if is_train else 'validation'

    try:
        builder = tfds.builder(tfds_name)
        splits = builder.info.splits
        if split in splits:
            return int(splits[split].num_examples)
        # fallback: first available split
        return int(next(iter(splits.values())).num_examples)
    except Exception as e:
        if jax.process_index() == 0:
            print(
                f"[GMM-FM] WARNING: could not read TFDS num_examples for {tfds_name} ({e}). "
                f"Using 100000 as fallback.")
        return 100000


def determine_cache_path(cfg: GMMPreprocessConfig) -> str:
    """
    Determine cache path from config, auto-generating if needed.

    Args:
        cfg: GMM preprocessing config

    Returns:
        Absolute path to cache file
    """
    if cfg.cache_path:
        return cfg.cache_path

    base_dir = cfg.save_dir if cfg.save_dir is not None else os.getcwd()
    return os.path.join(
        base_dir,
        f"gmm_prior_{cfg.dataset_name}_K{cfg.K}.npz"
    )


def collect_init_points(
    dataset_iterator: Iterator,
    encode_fn: Callable,
    K: int,
    rng: jax.Array,
    verbose: bool = True
) -> Tuple[jnp.ndarray, jax.Array]:
    """
    Collect K initialization points from dataset.

    Args:
        dataset_iterator: Iterator yielding (images, labels) batches
        encode_fn: Function (batch_images, rng_key) -> latent_batch
        K: Number of points to collect
        rng: JAX random key
        verbose: Whether to show progress bar

    Returns:
        init_points: (K, D) array
        rng: Updated random key
    """
    init_buf = []
    need = K
    total_needed = K

    pbar = None
    if verbose and jax.process_index() == 0:
        print(f"[GMM-FM] Collecting {total_needed} init points...")
        pbar = tqdm.tqdm(total=total_needed, desc="GMM init points", unit="pts")

    try:
        while need > 0:
            batch_images, _ = next(dataset_iterator)
            rng, k = jax.random.split(rng)
            lat = encode_fn(batch_images, k)
            x = jnp.asarray(lat).reshape((lat.shape[0], -1)).astype(jnp.float32)
            take = min(int(x.shape[0]), need)
            init_buf.append(x[:take])
            need -= take

            if pbar is not None:
                pbar.update(take)
    finally:
        if pbar is not None:
            pbar.close()

    init_points = jnp.concatenate(init_buf, axis=0)
    return init_points, rng


def create_latent_iterator(
    dataset_iterator: Iterator,
    encode_fn: Callable,
    rng_holder: dict  # mutable container for RNG
) -> Iterator[jnp.ndarray]:
    """
    Create iterator that yields flattened latent batches.

    Args:
        dataset_iterator: Iterator yielding (images, labels)
        encode_fn: Encoding function (batch_images, rng_key) -> latents
        rng_holder: Dict with 'rng' key for mutable RNG state

    Yields:
        Flattened latent batches (B, D)
    """
    while True:
        batch_images, _ = next(dataset_iterator)
        rng_holder['rng'], k = jax.random.split(rng_holder['rng'])
        lat = encode_fn(batch_images, k)
        x = jnp.asarray(lat).reshape((lat.shape[0], -1)).astype(jnp.float32)
        yield x


def fit_gmm_prior(
    dataset_iterator: Iterator,
    encode_fn: Callable,
    cfg: GMMPreprocessConfig,
    local_batch_size: int,
    rng: jax.Array,
    verbose: bool = True
) -> Tuple[GMMPrior, dict, jax.Array]:
    """
    Fit GMM prior using EM algorithm on streaming data.

    Args:
        dataset_iterator: Dataset iterator
        encode_fn: Encoding function (batch_images, rng_key) -> latents
        cfg: GMM preprocessing config
        local_batch_size: Batch size for this process
        rng: JAX random key
        verbose: Show progress bars

    Returns:
        gmm_prior: Fitted GMM prior
        em_logs: EM training logs
        rng: Updated random key
    """
    # Calculate dataset size and target examples
    em_subset = max(0.0, min(1.0, cfg.em_subset))
    num_examples = get_dataset_size(cfg.dataset_name, is_train=True)
    target_examples = max(int(num_examples * em_subset), cfg.K * 4)
    num_batches = int(np.ceil(target_examples / local_batch_size))

    if verbose and jax.process_index() == 0:
        print(f"[GMM-FM] Train split examples: {num_examples}")
        print(f"[GMM-FM] EM_subset={em_subset} => target_examples={target_examples} "
              f"(~{num_batches} batches)")

    # Collect init points
    init_points, rng = collect_init_points(
        dataset_iterator, encode_fn, cfg.K, rng, verbose
    )
    D = int(init_points.shape[1])

    # Create latent batch iterator
    rng_holder = {'rng': rng}
    latent_iter = create_latent_iterator(dataset_iterator, encode_fn, rng_holder)

    # Setup EM config
    em_cfg = EMConfig(
        K=cfg.K,
        max_iters=cfg.em_max_iters,
        var_floor=cfg.var_floor,
        tol=cfg.em_tol,
        patience=cfg.em_patience,
    )

    # Setup W&B callback
    def on_iter_end(iter_idx, logs):
        if jax.process_index() == 0:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "gmm_em/iter": int(iter_idx),
                        "gmm_em/loglik": float(logs["loglik"]),
                        "gmm_em/min_Nk": float(logs["min_Nk"]),
                        "gmm_em/max_Nk": float(logs["max_Nk"]),
                    })
            except ImportError:
                pass

    # Fit GMM
    gmm_prior, em_logs = fit_gmm_em_streaming(
        latent_iter,
        num_batches=num_batches,
        D=D,
        cfg=em_cfg,
        rng=rng,
        init_points=init_points,
        on_iter_end=on_iter_end,
        verbose=verbose,
    )

    rng = rng_holder['rng']
    return gmm_prior, em_logs, rng


def save_and_log_gmm_prior(
    gmm_prior: GMMPrior,
    cache_path: str,
    cfg: GMMPreprocessConfig,
    verbose: bool = True
) -> None:
    """
    Save GMM prior to disk and log to W&B (process 0 only).

    Args:
        gmm_prior: Fitted GMM prior
        cache_path: Path to save NPZ file
        cfg: GMM preprocessing config
        verbose: Print save confirmation
    """
    if jax.process_index() != 0:
        return

    # Save to disk
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    save_prior_npz(cache_path, gmm_prior)

    if verbose:
        print(f"[GMM-FM] Saved GMM prior to: {cache_path}")

    # Log to W&B as artifact
    try:
        import wandb
        if wandb.run is not None and cfg.log_artifact:
            try:
                art = wandb.Artifact(
                    name=f"gmm_prior_{cfg.dataset_name}_K{cfg.K}",
                    type="gmm_prior"
                )
                art.add_file(cache_path)
                wandb.log_artifact(art)
            except Exception as e:
                print(f"[GMM-FM] W&B artifact logging failed: {e}")
    except ImportError:
        pass


def log_pca_visualization(
    dataset_iterator: Iterator,
    encode_fn: Callable,
    gmm_prior: GMMPrior,
    cache_path: str,
    cfg: GMMPreprocessConfig,
    rng: jax.Array,
) -> None:
    """
    Create and log PCA visualization to W&B (process 0 only).

    Args:
        dataset_iterator: Fresh dataset iterator for PCA sampling
        encode_fn: Encoding function
        gmm_prior: Fitted GMM prior
        cache_path: Cache path (used to determine viz save location)
        cfg: GMM preprocessing config
        rng: JAX random key
    """
    if jax.process_index() != 0:
        return

    try:
        import wandb
        if wandb.run is None:
            return
    except ImportError:
        return

    max_pts = max(200, cfg.pca_max_points)
    tmp_path = os.path.join(
        os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".",
        "gmm_pca2d.png"
    )

    create_and_log_pca_visualization(
        dataset_iterator=dataset_iterator,
        encode_fn=encode_fn,
        gmm_prior=gmm_prior,
        rng=rng,
        max_points=max_pts,
        save_path=tmp_path,
        wandb_key="gmm_em/pca2d",
        dpi=150
    )


def preprocess_gmm_prior(
    cfg: GMMPreprocessConfig,
    get_dataset_fn: Callable,
    encode_fn: Callable,
    local_batch_size: int,
    debug_overfit: int = 0,
    verbose: bool = True
) -> GMMPrior:
    """
    Complete GMM preprocessing pipeline: load cached or fit new GMM prior.

    This is the main entry point for GMM-FM preprocessing. It handles:
    - Cache path determination
    - Loading existing cached prior
    - Or fitting new prior with EM
    - Saving and logging results
    - PCA visualization

    Args:
        cfg: GMM preprocessing configuration
        get_dataset_fn: Function (dataset_name, batch_size, is_train, debug) -> iterator
        encode_fn: Function (batch_images, rng_key) -> latent_batch
        local_batch_size: Batch size for this process
        debug_overfit: Debug overfit flag
        verbose: Show progress bars and messages

    Returns:
        gmm_prior: Loaded or fitted GMM prior

    Example:
        >>> cfg = GMMPreprocessConfig.from_flags(FLAGS)
        >>> gmm_prior = preprocess_gmm_prior(
        ...     cfg=cfg,
        ...     get_dataset_fn=get_dataset,
        ...     encode_fn=_encode_to_latent,
        ...     local_batch_size=local_batch_size,
        ...     debug_overfit=FLAGS.debug_overfit
        ... )
    """
    # Determine cache path
    cache_path = determine_cache_path(cfg)

    # Try loading existing prior
    if not cfg.force_refit and os.path.exists(cache_path):
        if verbose and jax.process_index() == 0:
            print(f"[GMM-FM] Loading GMM prior from: {cache_path}")
        return load_prior_npz(cache_path)

    # Need to fit new prior
    if verbose and jax.process_index() == 0:
        print("[GMM-FM] Fitting GMM prior with EM...")
        print(f"[GMM-FM] cache_path: {cache_path}")

    # Create fresh dataset iterator for EM
    dataset_em = get_dataset_fn(
        cfg.dataset_name, local_batch_size, True, debug_overfit
    )

    # Fit GMM prior
    rng = jax.random.PRNGKey(cfg.seed + 12345)
    gmm_prior, em_logs, rng = fit_gmm_prior(
        dataset_iterator=dataset_em,
        encode_fn=encode_fn,
        cfg=cfg,
        local_batch_size=local_batch_size,
        rng=rng,
        verbose=verbose
    )

    # Save and log
    save_and_log_gmm_prior(gmm_prior, cache_path, cfg, verbose)

    # PCA visualization
    try:
        import wandb
        if wandb.run is not None:
            dataset_pca = get_dataset_fn(
                cfg.dataset_name, local_batch_size, True, debug_overfit
            )
            log_pca_visualization(
                dataset_iterator=dataset_pca,
                encode_fn=encode_fn,
                gmm_prior=gmm_prior,
                cache_path=cache_path,
                cfg=cfg,
                rng=rng
            )
    except ImportError:
        pass

    return gmm_prior
