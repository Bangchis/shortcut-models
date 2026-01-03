"""
Latent cache creation and loading for GMM-FM paper implementation.

Encodes entire dataset to latents and saves as memory-mapped numpy array
for efficient random access during training.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class LatentCacheConfig:
    """Configuration for latent cache creation."""
    cache_dir: str
    dataset_name: str
    vae_epsilon_scale: float = 1.0
    dtype: str = 'float16'  # Storage dtype
    overwrite: bool = False

    def get_cache_path(self) -> str:
        """Get path to latent cache file."""
        eps_str = f"eps{self.vae_epsilon_scale:.2f}".replace('.', 'p')
        filename = f"latents_{self.dataset_name}_{eps_str}.npy"
        return str(Path(self.cache_dir) / filename)

    def get_meta_path(self) -> str:
        """Get path to metadata file."""
        return self.get_cache_path().replace('.npy', '_meta.json')

    def get_config_hash(self) -> str:
        """Get hash of configuration for validation."""
        config_str = f"{self.dataset_name}_{self.vae_epsilon_scale}_{self.dtype}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


def create_latent_cache(
    dataset_iterator,
    encode_fn: Callable,
    cfg: LatentCacheConfig,
    num_examples: int,
    local_batch_size: int,
    example_latent_shape: Tuple[int, ...],  # (H, W, C)
    verbose: bool = True
) -> str:
    """
    Encode entire dataset to latents and save as memmap.

    Args:
        dataset_iterator: Iterator yielding (images, labels) batches
        encode_fn: Function (key, images) -> latents
        cfg: LatentCacheConfig
        num_examples: Total number of examples in dataset
        local_batch_size: Batch size
        example_latent_shape: Shape of single latent (H, W, C)
        verbose: Show progress bar

    Returns:
        path: Path to created latent cache file
    """
    cache_path = cfg.get_cache_path()
    meta_path = cfg.get_meta_path()

    # Create cache directory
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    # Multi-host coordination: only process 0 creates cache
    if jax.process_index() == 0:
        if verbose:
            print(f"[Latent Cache] Creating cache at: {cache_path}")
            print(f"[Latent Cache] Dataset: {cfg.dataset_name}, epsilon_scale: {cfg.vae_epsilon_scale}")
            print(f"[Latent Cache] Total examples: {num_examples}, dtype: {cfg.dtype}")

        # Determine storage dtype
        storage_dtype = np.dtype(cfg.dtype)

        # Create memmap file
        latent_shape = (num_examples,) + example_latent_shape
        latents_mmap = np.memmap(
            cache_path,
            dtype=storage_dtype,
            mode='w+',
            shape=latent_shape
        )

        # Encode dataset in batches
        rng = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
        idx = 0

        if verbose:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=num_examples, desc="Encoding latents", unit="imgs")
            except ImportError:
                pbar = None
        else:
            pbar = None

        try:
            while idx < num_examples:
                # Get batch
                batch_images, _ = next(dataset_iterator)
                B = batch_images.shape[0]

                # Encode
                rng, encode_key = jax.random.split(rng)
                latents = encode_fn(encode_key, batch_images)
                latents_np = np.array(jax.device_get(latents))

                # Write to memmap
                end_idx = min(idx + B, num_examples)
                actual_B = end_idx - idx
                latents_mmap[idx:end_idx] = latents_np[:actual_B].astype(storage_dtype)

                idx = end_idx

                if pbar is not None:
                    pbar.update(actual_B)

            # Flush to disk
            latents_mmap.flush()

            if pbar is not None:
                pbar.close()

            if verbose:
                print(f"[Latent Cache] Successfully created cache with shape {latent_shape}")

        finally:
            # Clean up memmap
            del latents_mmap

        # Save metadata
        metadata = {
            'dataset_name': cfg.dataset_name,
            'vae_epsilon_scale': cfg.vae_epsilon_scale,
            'dtype': cfg.dtype,
            'shape': list(latent_shape),
            'num_examples': num_examples,
            'config_hash': cfg.get_config_hash(),
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            print(f"[Latent Cache] Metadata saved to: {meta_path}")

        # Write ready flag
        ready_path = cache_path + ".ready"
        Path(ready_path).touch()

    # Multi-host barrier
    if jax.process_count() > 1:
        _wait_for_cache(cache_path + ".ready", timeout=3600)
        jax.experimental.multihost_utils.sync_global_devices("latent_cache_ready")

    # All processes validate
    _validate_cache(cache_path, meta_path, cfg, num_examples, example_latent_shape)

    return cache_path


def load_latent_cache(cache_path: str, mode: str = 'r') -> np.memmap:
    """
    Load existing latent cache as memmap.

    Args:
        cache_path: Path to .npy file
        mode: Memmap mode ('r' for read-only, 'r+' for read-write)

    Returns:
        latents_mmap: Memory-mapped array
    """
    if not Path(cache_path).exists():
        raise FileNotFoundError(f"Latent cache not found: {cache_path}")

    # Load metadata to get shape and dtype
    meta_path = cache_path.replace('.npy', '_meta.json')
    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['dtype'])

    # Load memmap
    latents_mmap = np.memmap(cache_path, dtype=dtype, mode=mode, shape=shape)

    return latents_mmap


def _wait_for_cache(ready_file: str, timeout: int):
    """Wait for ready file with timeout (multi-host coordination)."""
    import time
    start = time.time()
    while not Path(ready_file).exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"Cache creation timeout after {timeout}s")
        time.sleep(1)


def _validate_cache(
    cache_path: str,
    meta_path: str,
    cfg: LatentCacheConfig,
    expected_num_examples: int,
    expected_latent_shape: Tuple[int, ...]
):
    """Validate cache file and metadata."""
    # Check files exist
    if not Path(cache_path).exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    # Load and validate metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # Validate configuration
    if metadata['dataset_name'] != cfg.dataset_name:
        raise ValueError(f"Dataset name mismatch: {metadata['dataset_name']} != {cfg.dataset_name}")

    if abs(metadata['vae_epsilon_scale'] - cfg.vae_epsilon_scale) > 1e-6:
        raise ValueError(f"VAE epsilon scale mismatch: {metadata['vae_epsilon_scale']} != {cfg.vae_epsilon_scale}")

    if metadata['dtype'] != cfg.dtype:
        raise ValueError(f"Dtype mismatch: {metadata['dtype']} != {cfg.dtype}")

    # Validate shape
    expected_shape = (expected_num_examples,) + expected_latent_shape
    actual_shape = tuple(metadata['shape'])

    if actual_shape != expected_shape:
        raise ValueError(f"Shape mismatch: {actual_shape} != {expected_shape}")

    # Validate file size
    expected_size = np.prod(expected_shape) * np.dtype(cfg.dtype).itemsize
    actual_size = Path(cache_path).stat().st_size

    if actual_size != expected_size:
        raise ValueError(f"File size mismatch: {actual_size} != {expected_size} bytes")


def get_cache_info(cache_path: str) -> dict:
    """Get cache metadata as dict."""
    meta_path = cache_path.replace('.npy', '_meta.json')

    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    return metadata
