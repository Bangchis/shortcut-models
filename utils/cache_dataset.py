"""
Cache dataset iterator for GMM-FM paper training.

Samples (x1_latent, k_vec) from cached latents and cluster assignments.
"""

from typing import Tuple
import numpy as np

from utils.cluster_cache import ClusterCache


class CacheDatasetIterator:
    """
    Iterator that yields (x1_latent, k_vec) from cached data.

    Each iteration:
    1. Sample k_vec ~ Categorical(pi_emp)  (B,)
    2. For each k in k_vec, sample random index from cluster k
    3. Gather x1 = latents[indices]
    4. Yield (x1, k_vec)

    This implements the paper's mixed batch sampling approach:
    - Sample cluster k ~ π
    - Sample data index uniformly from D_k
    - x1 = cached latent at that index
    - x0 will be sampled from N(μ_k, σ_k²) in targets
    """

    def __init__(
        self,
        latent_cache: np.memmap,
        cluster_cache: ClusterCache,
        batch_size: int,
        rng_seed: int = 42,
    ):
        """
        Args:
            latent_cache: Memory-mapped latents (N, H, W, C)
            cluster_cache: ClusterCache with CSR structure
            batch_size: Number of samples per batch
            rng_seed: Random seed (different per process for multi-host)
        """
        self.latent_cache = latent_cache
        self.cluster_cache = cluster_cache
        self.batch_size = batch_size
        self.rng = np.random.RandomState(rng_seed)

        # Precompute valid clusters (non-empty)
        self.valid_clusters = np.where(cluster_cache.counts > 0)[0]
        if len(self.valid_clusters) < cluster_cache.K:
            print(f"[CacheDataset] WARNING: {cluster_cache.K - len(self.valid_clusters)} empty clusters")
            # Adjust pi_emp to only valid clusters
            valid_counts = cluster_cache.counts[self.valid_clusters]
            self.pi_emp_valid = valid_counts.astype(np.float32) / valid_counts.sum()
        else:
            self.pi_emp_valid = cluster_cache.pi_emp

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample next batch.

        Returns:
            x1: Latents (B, H, W, C) float32
            k_vec: Cluster IDs (B,) int32
        """
        # Sample k_vec from empirical distribution
        k_vec = self.rng.choice(
            self.valid_clusters,
            size=self.batch_size,
            p=self.pi_emp_valid
        ).astype(np.int32)

        # Sample indices from each cluster
        indices = np.zeros(self.batch_size, dtype=np.int32)

        for i, k in enumerate(k_vec):
            # Get cluster boundaries from CSR
            start = self.cluster_cache.ptr[k]
            end = self.cluster_cache.ptr[k+1]

            if end > start:
                # Sample random index within cluster k
                local_idx = self.rng.randint(0, end - start)
                idx = self.cluster_cache.idx[start + local_idx]
            else:
                # Empty cluster fallback: sample from largest cluster
                largest_k = np.argmax(self.cluster_cache.counts)
                start_fallback = self.cluster_cache.ptr[largest_k]
                end_fallback = self.cluster_cache.ptr[largest_k+1]
                local_idx = self.rng.randint(0, end_fallback - start_fallback)
                idx = self.cluster_cache.idx[start_fallback + local_idx]

            indices[i] = idx

        # Gather latents from memmap
        x1 = self.latent_cache[indices]  # (B, H, W, C)
        x1 = x1.astype(np.float32)  # Convert from float16 to float32

        return x1, k_vec


class CacheDatasetIteratorWithLabels(CacheDatasetIterator):
    """
    Extended iterator that also yields labels for class-conditional training.

    Yields (x1_latent, k_vec, labels).
    """

    def __init__(
        self,
        latent_cache: np.memmap,
        cluster_cache: ClusterCache,
        labels_array: np.ndarray,  # (N,) int32
        batch_size: int,
        rng_seed: int = 42,
    ):
        """
        Args:
            latent_cache: Memory-mapped latents (N, H, W, C)
            cluster_cache: ClusterCache with CSR structure
            labels_array: Class labels (N,) int32
            batch_size: Number of samples per batch
            rng_seed: Random seed
        """
        super().__init__(latent_cache, cluster_cache, batch_size, rng_seed)
        self.labels_array = labels_array

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample next batch with labels.

        Returns:
            x1: Latents (B, H, W, C) float32
            k_vec: Cluster IDs (B,) int32
            labels: Class labels (B,) int32
        """
        # Get base samples
        x1, k_vec = super().__next__()

        # Get corresponding labels
        # We need to track which indices were sampled
        # For now, we'll re-implement the sampling to get indices

        # Re-sample to get indices (TODO: optimize by modifying parent class)
        k_vec_resample = self.rng.choice(
            self.valid_clusters,
            size=self.batch_size,
            p=self.pi_emp_valid
        ).astype(np.int32)

        indices = np.zeros(self.batch_size, dtype=np.int32)
        for i, k in enumerate(k_vec_resample):
            start = self.cluster_cache.ptr[k]
            end = self.cluster_cache.ptr[k+1]
            if end > start:
                local_idx = self.rng.randint(0, end - start)
                idx = self.cluster_cache.idx[start + local_idx]
            else:
                largest_k = np.argmax(self.cluster_cache.counts)
                start_fallback = self.cluster_cache.ptr[largest_k]
                end_fallback = self.cluster_cache.ptr[largest_k+1]
                local_idx = self.rng.randint(0, end_fallback - start_fallback)
                idx = self.cluster_cache.idx[start_fallback + local_idx]
            indices[i] = idx

        # Gather latents and labels
        x1 = self.latent_cache[indices].astype(np.float32)
        labels = self.labels_array[indices]

        return x1, k_vec_resample, labels
