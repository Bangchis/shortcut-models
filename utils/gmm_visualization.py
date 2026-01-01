"""
GMM Visualization utilities for debugging and analysis.

This module provides PCA-based visualization tools for understanding
the structure of GMM priors in latent space.
"""

from __future__ import annotations

from typing import Optional, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from .gmm_prior import GMMPrior, posterior_logp


def compute_pca_2d(
    X: np.ndarray,
    return_components: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute 2D PCA projection using Gram matrix method.

    Efficient for cases where N << D (few samples, high dimension).

    Args:
        X: Data matrix (N, D)
        return_components: If True, also return the PCA basis vectors V2

    Returns:
        Z: Projected data (N, 2)
        mean: Data mean (1, D)
        evals2: Top 2 eigenvalues (2,)
        V2: PCA basis vectors (D, 2), only if return_components=True
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    N = Xc.shape[0]

    # Gram matrix PCA (efficient when N << D)
    G = (Xc @ Xc.T) / max(N - 1, 1)  # (N, N)
    evals, evecs = np.linalg.eigh(G)

    # Select top 2 components
    idx2 = np.argsort(evals)[-2:]
    evals2 = evals[idx2]
    U2 = evecs[:, idx2]  # (N, 2)

    # Scale factor to correct for normalized Gram matrix
    scale = np.sqrt(max(N - 1, 1))

    # Project data: Z = Xc @ V = U @ sqrt((N-1) * evals)
    Z = U2 * np.sqrt(np.maximum(evals2, 1e-12)) * scale  # (N, 2)

    if return_components:
        # Compute basis vectors in feature space
        # V = Xc.T @ U / sqrt((N-1) * evals)
        V2 = (Xc.T @ U2) / (np.sqrt(np.maximum(evals2, 1e-12))[None, :] * scale)  # (D, 2)
        return Z, mean, evals2, V2
    else:
        return Z, mean, evals2, None


def project_to_pca_space(
    points: np.ndarray,
    mean: np.ndarray,
    V2: np.ndarray
) -> np.ndarray:
    """
    Project new points to existing PCA space.

    Args:
        points: Points to project (K, D)
        mean: PCA mean (1, D)
        V2: PCA basis vectors (D, 2)

    Returns:
        projected: Points in 2D PCA space (K, 2)
    """
    points_c = points - mean
    projected = points_c @ V2
    return projected


def visualize_gmm_pca2d(
    X: np.ndarray,
    gmm_prior: GMMPrior,
    title: str = "GMM-FM: PCA 2D of latents + centers",
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Create PCA 2D visualization of latent space with GMM components.

    Args:
        X: Latent vectors (N, D) - must be numpy array
        gmm_prior: Fitted GMM prior
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Compute PCA
    Z, mean, evals2, V2 = compute_pca_2d(X, return_components=True)

    # Project GMM centers
    mu = np.array(jax.device_get(gmm_prior.mu))  # (K, D)
    centers_2d = project_to_pca_space(mu, mean, V2)

    # Color points by hard assignment
    log_r = np.array(jax.device_get(posterior_logp(gmm_prior, jnp.asarray(X))))
    mode = log_r.argmax(axis=-1)

    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Scatter data points colored by assignment
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=mode, s=6, alpha=0.6)

    # Plot GMM centers
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               marker='x', s=80, linewidths=2)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()

    return fig


def sample_latents_for_pca(
    dataset_iterator,
    encode_fn: Callable,
    max_points: int,
    rng: jax.Array,
    verbose: bool = True
) -> Tuple[np.ndarray, jax.Array]:
    """
    Sample and encode latents from dataset for PCA analysis.

    Args:
        dataset_iterator: Iterator yielding (images, labels) batches
        encode_fn: Function to encode images to latent space
        max_points: Maximum number of points to collect
        rng: JAX random key
        verbose: If True, show progress bar

    Returns:
        X: Latent vectors (N, D) as numpy array
        rng: Updated JAX random key
    """
    pts = []

    # Create progress bar
    pbar = None
    if verbose:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_points, desc="Sampling for PCA", unit="pts")
        except ImportError:
            pass

    try:
        while sum([p.shape[0] for p in pts]) < max_points:
            batch_images, _ = next(dataset_iterator)
            rng, k = jax.random.split(rng)
            lat = encode_fn(batch_images, k)
            x = np.array(jax.device_get(
                jnp.asarray(lat).reshape((lat.shape[0], -1)).astype(jnp.float32)
            ))
            pts.append(x)

            # Update progress
            if pbar is not None:
                current_total = sum([p.shape[0] for p in pts])
                pbar.n = min(current_total, max_points)
                pbar.refresh()
    finally:
        if pbar is not None:
            pbar.close()

    X = np.concatenate(pts, axis=0)[:max_points]
    return X, rng


def create_and_log_pca_visualization(
    dataset_iterator,
    encode_fn: Callable,
    gmm_prior: GMMPrior,
    rng: jax.Array,
    max_points: int = 2000,
    save_path: Optional[str] = None,
    wandb_key: str = "gmm_em/pca2d",
    dpi: int = 150
) -> Optional[str]:
    """
    Complete pipeline: sample data, compute PCA, create plot, save/log.

    Args:
        dataset_iterator: Dataset iterator
        encode_fn: Encoding function
        gmm_prior: Fitted GMM prior
        rng: JAX random key
        max_points: Max points for PCA
        save_path: Path to save plot (if None, uses temp)
        wandb_key: W&B log key
        dpi: Plot DPI

    Returns:
        Path to saved plot, or None if failed
    """
    try:
        import wandb

        # Sample latents
        X, _ = sample_latents_for_pca(
            dataset_iterator, encode_fn, max_points, rng
        )

        # Create visualization
        fig = visualize_gmm_pca2d(X, gmm_prior)

        # Save plot
        if save_path is None:
            import tempfile
            save_path = tempfile.mktemp(suffix='.png')

        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)

        # Log to wandb
        if wandb.run is not None:
            wandb.log({wandb_key: wandb.Image(save_path)})

        return save_path

    except Exception as e:
        print(f"[GMM-FM] PCA viz failed: {e}")
        return None
