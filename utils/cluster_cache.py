"""
Cluster cache creation and loading for GMM-FM paper implementation.

Hard-partitions dataset into clusters using GMM prior and builds
CSR (Compressed Sparse Row) structure for efficient sampling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union
import json

import jax
import jax.numpy as jnp
import numpy as np

from utils.gmm_prior import GMMPrior, posterior_logp


@dataclass
class ClusterCache:
    """
    Cluster cache with CSR structure for efficient sampling.

    Attributes:
        cluster_id: (N,) int32 - cluster assignment for each sample
        ptr: (K+1,) int32 - CSR pointer array
        idx: (N,) int32 - sorted sample indices by cluster
        counts: (K,) int32 - number of samples per cluster
        pi_emp: (K,) float32 - empirical cluster probabilities
    """
    cluster_id: np.ndarray  # (N,) int32
    ptr: np.ndarray         # (K+1,) int32
    idx: np.ndarray         # (N,) int32
    counts: np.ndarray      # (K,) int32
    pi_emp: np.ndarray      # (K,) float32

    @property
    def K(self) -> int:
        """Number of clusters."""
        return len(self.counts)

    @property
    def N(self) -> int:
        """Total number of samples."""
        return len(self.cluster_id)

    def get_cluster_indices(self, k: int) -> np.ndarray:
        """Get all sample indices for cluster k."""
        return self.idx[self.ptr[k]:self.ptr[k+1]]


@dataclass
class ClusterCacheConfig:
    """Configuration for cluster cache creation."""
    latent_cache_path: str
    prior_path: str  # GMM prior npz
    save_path: str
    overwrite: bool = False


def create_cluster_cache(
    latent_cache: Union[np.memmap, Tuple[np.memmap, np.memmap]],
    gmm_prior: GMMPrior,
    cfg: ClusterCacheConfig,
    batch_size: int = 1024,
    verbose: bool = True
) -> str:
    """
    Hard-partition latents into clusters using GMM prior.

    Steps:
    1. Process latents in batches: cluster_id[i] = argmax_k log p(k|z_i)
    2. Build CSR structure (ptr, idx) for efficient sampling
    3. Compute empirical pi_emp (counts per cluster)
    4. Save clusters.npz

    Args:
        latent_cache: Memory-mapped latents array (N, H, W, C) OR
                     (mu_mmap, logvar_mmap) tuple for posterior mode
        gmm_prior: Fitted GMM prior
        cfg: ClusterCacheConfig
        batch_size: Batch size for processing
        verbose: Show progress

    Returns:
        save_path: Path to saved clusters.npz
    """
    save_path = cfg.save_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Handle both single memmap and tuple input
    if isinstance(latent_cache, tuple):
        # Posterior mode: use μ only for deterministic clustering
        mu_cache, logvar_cache = latent_cache
        latents_for_clustering = mu_cache
        is_posterior = True
    else:
        # Legacy mode: use fixed latents
        latents_for_clustering = latent_cache
        is_posterior = False

    # Multi-host: only process 0 creates cache
    if jax.process_index() == 0:
        if verbose:
            print(f"[Cluster Cache] Creating cluster cache at: {save_path}")
            print(f"[Cluster Cache] Mode: {'Posterior (using μ only)' if is_posterior else 'Legacy (fixed latents)'}")
            print(f"[Cluster Cache] GMM prior: K={gmm_prior.K} components")
            print(f"[Cluster Cache] Dataset size: N={latents_for_clustering.shape[0]}")

        N = latents_for_clustering.shape[0]
        K = gmm_prior.K
        D = np.prod(latents_for_clustering.shape[1:])  # Flatten dimension

        # Allocate cluster_id array
        cluster_id = np.zeros(N, dtype=np.int32)

        # Process in batches
        num_batches = (N + batch_size - 1) // batch_size

        if verbose:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=N, desc="Assigning clusters", unit="samples")
            except ImportError:
                pbar = None
        else:
            pbar = None

        try:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, N)
                B = end_idx - start_idx

                # Load batch and flatten (use latents_for_clustering)
                latents_batch = latents_for_clustering[start_idx:end_idx]  # (B, H, W, C)
                latents_flat = latents_batch.reshape(B, -1).astype(np.float32)
                latents_flat = jnp.asarray(latents_flat)

                # Compute posterior log probabilities
                log_r = posterior_logp(gmm_prior, latents_flat)  # (B, K)

                # Hard assignment: argmax
                k_batch = jnp.argmax(log_r, axis=-1).astype(jnp.int32)  # (B,)
                cluster_id[start_idx:end_idx] = np.array(jax.device_get(k_batch))

                if pbar is not None:
                    pbar.update(B)

            if pbar is not None:
                pbar.close()

        except Exception as e:
            if pbar is not None:
                pbar.close()
            raise e

        # Compute counts
        counts = np.bincount(cluster_id, minlength=K).astype(np.int32)
        pi_emp = counts.astype(np.float32) / counts.sum()

        if verbose:
            print(f"[Cluster Cache] Cluster counts: {counts}")
            print(f"[Cluster Cache] Empty clusters: {np.sum(counts == 0)}")
            if np.any(counts == 0):
                empty_clusters = np.where(counts == 0)[0]
                print(f"[Cluster Cache] WARNING: Empty cluster IDs: {empty_clusters}")

        # Build CSR structure
        # Sort indices by cluster_id
        sort_idx = np.argsort(cluster_id)
        idx = sort_idx.astype(np.int32)

        # Build ptr array
        ptr = np.zeros(K + 1, dtype=np.int32)
        ptr[1:] = np.cumsum(counts)

        if verbose:
            print(f"[Cluster Cache] CSR structure built")
            print(f"[Cluster Cache] ptr shape: {ptr.shape}, idx shape: {idx.shape}")

        # Save to npz
        np.savez(
            save_path,
            cluster_id=cluster_id,
            ptr=ptr,
            idx=idx,
            counts=counts,
            pi_emp=pi_emp,
        )

        if verbose:
            print(f"[Cluster Cache] Saved to: {save_path}")

        # Save metadata
        metadata = {
            'N': int(N),
            'K': int(K),
            'empty_clusters': int(np.sum(counts == 0)),
            'min_count': int(np.min(counts)),
            'max_count': int(np.max(counts)),
            'mean_count': float(np.mean(counts)),
        }

        meta_path = save_path.replace('.npz', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Write ready flag
        ready_path = save_path + ".ready"
        Path(ready_path).touch()

    # Multi-host barrier
    if jax.process_count() > 1:
        _wait_for_cache(save_path + ".ready", timeout=3600)
        jax.experimental.multihost_utils.sync_global_devices("cluster_cache_ready")

    return save_path


def load_cluster_cache(cache_path: str) -> ClusterCache:
    """
    Load cluster cache from .npz file.

    Args:
        cache_path: Path to clusters.npz

    Returns:
        ClusterCache object
    """
    if not Path(cache_path).exists():
        raise FileNotFoundError(f"Cluster cache not found: {cache_path}")

    # Load npz
    data = np.load(cache_path)

    cluster_cache = ClusterCache(
        cluster_id=data['cluster_id'],
        ptr=data['ptr'],
        idx=data['idx'],
        counts=data['counts'],
        pi_emp=data['pi_emp'],
    )

    return cluster_cache


def _wait_for_cache(ready_file: str, timeout: int):
    """Wait for ready file with timeout (multi-host coordination)."""
    import time
    start = time.time()
    while not Path(ready_file).exists():
        if time.time() - start > timeout:
            raise TimeoutError(f"Cluster cache creation timeout after {timeout}s")
        time.sleep(1)


def get_cluster_stats(cluster_cache: ClusterCache) -> dict:
    """Get statistics about cluster distribution."""
    stats = {
        'N': cluster_cache.N,
        'K': cluster_cache.K,
        'counts': cluster_cache.counts.tolist(),
        'pi_emp': cluster_cache.pi_emp.tolist(),
        'empty_clusters': int(np.sum(cluster_cache.counts == 0)),
        'min_count': int(np.min(cluster_cache.counts)),
        'max_count': int(np.max(cluster_cache.counts)),
        'mean_count': float(np.mean(cluster_cache.counts)),
        'std_count': float(np.std(cluster_cache.counts)),
    }
    return stats
