"""
GMM Prior Utilities for Flow Matching

Provides sampling, log-probability computation, and visualization tools
for GMM-based flow matching.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from jaxtyping import Array, Float, PRNGKeyArray
import os

from utils.gmm_em import GMMParams, log_joint_diag


def save_gmm_prior(params: GMMParams, path: str, verbose: bool = True):
    """
    Save GMM parameters to npz file

    Args:
        params: GMMParams to save
        path: file path (*.npz)
        verbose: print confirmation
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert to numpy and save
    np.savez(
        path,
        pi=np.array(params.pi),
        mu=np.array(params.mu),
        var=np.array(params.var)
    )

    if verbose:
        K, D = params.mu.shape
        print(f"[GMM Prior] Saved to {path}")
        print(f"  K={K}, D={D}")
        print(f"  File size: {os.path.getsize(path) / 1024:.2f} KB")


def load_gmm_prior(path: str, verbose: bool = True) -> GMMParams:
    """
    Load GMM parameters from npz file

    Args:
        path: file path (*.npz)
        verbose: print confirmation

    Returns:
        GMMParams loaded from file
    """
    data = np.load(path)

    params = GMMParams(
        pi=jnp.array(data['pi']),
        mu=jnp.array(data['mu']),
        var=jnp.array(data['var'])
    )

    if verbose:
        K, D = params.mu.shape
        print(f"[GMM Prior] Loaded from {path}")
        print(f"  K={K}, D={D}")
        print(f"  Pi entropy: {-np.sum(params.pi * np.log(params.pi + 1e-10)):.4f}")
        print(f"  Var range: [{np.min(params.var):.6f}, {np.max(params.var):.6f}]")

    return params


def gmm_logp(
    params: GMMParams,
    x1_latent: Float[Array, "B ..."]
) -> Float[Array, "B K"]:
    """
    Compute log p(x1) under GMM for each component

    Args:
        params: GMMParams
        x1_latent: [B, H, W, C] or [B, D] latents

    Returns:
        logp: [B, K] log probabilities for each component
    """
    # Flatten if needed
    original_shape = x1_latent.shape
    if len(original_shape) > 2:
        B = original_shape[0]
        D = int(np.prod(original_shape[1:]))
        x1_flat = x1_latent.reshape(B, D)
    else:
        x1_flat = x1_latent

    # Compute log joint probabilities
    logp = log_joint_diag(x1_flat, params.pi, params.mu, params.var)

    return logp


def sample_gmm_x0(
    params: GMMParams,
    key: PRNGKeyArray,
    x1_latent: Float[Array, "B H W C"],
    assign_mode: str = 'soft_sample',
    temperature: float = 1.0,
    debug: bool = False
) -> Float[Array, "B H W C"]:
    """
    Sample x0 from GMM prior conditioned on x1 (via responsibilities)

    Implements Algorithm 2 Step 3 from paper with 3 assignment modes.

    Args:
        params: GMMParams (fitted GMM prior)
        key: JAX random key
        x1_latent: [B, H, W, C] data latents (for computing responsibilities)
        assign_mode: 'soft_sample' | 'moment' | 'hard'
            - soft_sample: Sample k~Cat(r), then x0~N(μ_k, σ²_k)
            - moment: Compute moment-matched Gaussian, then x0~N(μ̄, σ̄²)
            - hard: k=argmax(r), then x0~N(μ_k, σ²_k)
        temperature: temperature for softmax (lower = sharper assignment)
        debug: print debug info

    Returns:
        x0: [B, H, W, C] sampled noise from GMM
    """
    B, H, W, C = x1_latent.shape
    D = H * W * C

    # Compute responsibilities r[b, k] = softmax(logp[b, k] / T)
    logp = gmm_logp(params, x1_latent)  # [B, K]
    r = jax.nn.softmax(logp / temperature, axis=-1)  # [B, K]

    if debug:
        print(f"\n[GMM Sampling Debug]")
        print(f"  Mode: {assign_mode}, Temperature: {temperature}")
        print(f"  Batch size: {B}, Latent dim: {D}")
        print(f"  Responsibilities shape: {r.shape}")
        print(f"  Resp entropy (mean): {-np.mean(np.sum(r * np.log(r + 1e-10), axis=1)):.4f}")

    if assign_mode == 'soft_sample':
        # Sample component k ~ Categorical(r[b])
        key_k, key_x = jax.random.split(key)
        k = jax.random.categorical(key_k, jnp.log(r + 1e-10), axis=-1)  # [B]

        # Sample x0 ~ N(mu[k], var[k])
        mu_selected = params.mu[k]  # [B, D]
        var_selected = params.var[k]  # [B, D]
        x0_flat = mu_selected + jnp.sqrt(var_selected) * jax.random.normal(key_x, (B, D))

        if debug:
            unique_k, counts = np.unique(np.array(k), return_counts=True)
            print(f"  Sampled components: {len(unique_k)} unique out of {params.pi.shape[0]}")
            print(f"  Top 5 components: {unique_k[:5]} with counts {counts[:5]}")

    elif assign_mode == 'moment':
        # Compute moment-matched mean: μ̄ = Σ_k r[b,k] μ_k
        mu_bar = jnp.sum(r[:, :, None] * params.mu[None, :, :], axis=1)  # [B, D]

        # Compute moment-matched variance: σ̄² = E[σ²] + Var[μ]
        # E[σ²] = Σ_k r[b,k] σ²_k
        var_within = jnp.sum(r[:, :, None] * params.var[None, :, :], axis=1)  # [B, D]

        # Var[μ] = Σ_k r[b,k] (μ_k - μ̄)²
        var_between = jnp.sum(
            r[:, :, None] * (params.mu[None, :, :] - mu_bar[:, None, :])**2,
            axis=1
        )  # [B, D]

        var_bar = var_within + var_between  # [B, D]

        # Sample from N(μ̄, σ̄²)
        x0_flat = mu_bar + jnp.sqrt(var_bar) * jax.random.normal(key, (B, D))

        if debug:
            print(f"  Moment-matched mean norm (avg): {np.mean(np.linalg.norm(mu_bar, axis=1)):.4f}")
            print(f"  Moment-matched var (mean): {np.mean(var_bar):.6f}")
            print(f"  Var within/between ratio: {np.mean(var_within) / (np.mean(var_between) + 1e-10):.4f}")

    elif assign_mode == 'hard':
        # Hard assignment: k = argmax_k r[b,k]
        k = jnp.argmax(r, axis=-1)  # [B]

        # Sample x0 ~ N(mu[k], var[k])
        mu_selected = params.mu[k]  # [B, D]
        var_selected = params.var[k]  # [B, D]
        x0_flat = mu_selected + jnp.sqrt(var_selected) * jax.random.normal(key, (B, D))

        if debug:
            unique_k, counts = np.unique(np.array(k), return_counts=True)
            print(f"  Hard-assigned components: {len(unique_k)} unique")
            print(f"  Top 5 components: {unique_k[:5]} with counts {counts[:5]}")
            print(f"  Max responsibility (mean): {np.mean(np.max(r, axis=1)):.4f}")

    else:
        raise ValueError(f"Unknown assign_mode: {assign_mode}. "
                       f"Must be 'soft_sample', 'moment', or 'hard'")

    # Reshape back to [B, H, W, C]
    x0 = x0_flat.reshape(B, H, W, C)

    # Debug statistics
    if debug:
        print(f"  x0 norm (mean): {np.mean(np.linalg.norm(x0_flat, axis=1)):.4f}")
        print(f"  x0 std per dim (mean): {np.mean(np.std(x0_flat, axis=0)):.6f}")

    # NaN check
    assert not jnp.any(jnp.isnan(x0)), f"NaN detected in sampled x0 (mode={assign_mode})"

    return x0


def sample_gmm_unconditional(
    params: GMMParams,
    key: PRNGKeyArray,
    batch_size: int,
    shape: Tuple[int, int, int],  # (H, W, C)
    debug: bool = False
) -> Float[Array, "B H W C"]:
    """
    Sample from GMM without conditioning (for inference initialization)

    Samples component k ~ Cat(π) then x0 ~ N(μ_k, σ²_k)

    Args:
        params: GMMParams
        key: JAX random key
        batch_size: number of samples
        shape: latent shape (H, W, C) e.g. (32, 32, 4)
        debug: print debug info

    Returns:
        x0: [B, H, W, C] sampled from GMM
    """
    H, W, C = shape
    D = H * W * C
    K = params.pi.shape[0]

    # Sample component k ~ Categorical(π)
    key_k, key_x = jax.random.split(key)
    k = jax.random.categorical(key_k, jnp.log(params.pi + 1e-10), shape=(batch_size,))  # [B]

    # Sample x0 ~ N(mu[k], var[k])
    mu_selected = params.mu[k]  # [B, D]
    var_selected = params.var[k]  # [B, D]
    x0_flat = mu_selected + jnp.sqrt(var_selected) * jax.random.normal(key_x, (batch_size, D))

    # Reshape to [B, H, W, C]
    x0 = x0_flat.reshape(batch_size, H, W, C)

    if debug:
        unique_k, counts = np.unique(np.array(k), return_counts=True)
        print(f"\n[GMM Unconditional Sampling Debug]")
        print(f"  Batch size: {batch_size}, Shape: {shape}, K: {K}")
        print(f"  Sampled components: {len(unique_k)} unique")
        print(f"  Component usage: min={np.min(counts)}, max={np.max(counts)}, mean={np.mean(counts):.2f}")
        print(f"  x0 norm (mean): {np.mean(np.linalg.norm(x0_flat, axis=1)):.4f}")

    # NaN check
    assert not jnp.any(jnp.isnan(x0)), "NaN detected in unconditional GMM sampling"

    return x0


def plot_gmm_pca_2d(
    latents: np.ndarray,  # [N, D]
    params: GMMParams,
    max_points: int = 10000,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create PCA 2D scatter plot with GMM centers overlay

    Args:
        latents: [N, D] latent vectors
        params: GMMParams with fitted GMM
        max_points: maximum points to plot (for speed)
        figsize: matplotlib figure size

    Returns:
        matplotlib Figure
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        print("[Warning] matplotlib or sklearn not available, skipping PCA plot")
        return None

    # Subsample if needed
    if latents.shape[0] > max_points:
        indices = np.random.choice(latents.shape[0], max_points, replace=False)
        latents_sub = latents[indices]
    else:
        latents_sub = latents

    N, D = latents_sub.shape
    K = params.pi.shape[0]

    print(f"\n[PCA Visualization] Plotting {N} points, K={K} components...")

    # Fit PCA
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_sub)  # [N, 2]

    # Project GMM centers
    centers_2d = pca.transform(np.array(params.mu))  # [K, 2]

    # Assign each point to nearest component (for coloring)
    logp = log_joint_diag(
        jnp.array(latents_sub),
        params.pi,
        params.mu,
        params.var
    )
    assignments = np.argmax(np.array(logp), axis=-1)  # [N]

    # Compute responsibilities entropy for each point
    r = jax.nn.softmax(logp, axis=-1)
    resp_entropy = -np.sum(np.array(r) * np.log(np.array(r) + 1e-10), axis=1)  # [N]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Points colored by component assignment
    scatter1 = axes[0].scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=assignments,
        cmap='tab20',
        alpha=0.5,
        s=10,
        rasterized=True
    )

    # Overlay GMM centers
    axes[0].scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        linewidths=2,
        label='GMM Centers',
        zorder=10
    )

    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
    axes[0].set_title(f'GMM Prior - Component Assignment (K={K})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Component ID', fontsize=10)

    # Plot 2: Points colored by responsibility entropy
    scatter2 = axes[1].scatter(
        latents_2d[:, 0],
        latents_2d[:, 1],
        c=resp_entropy,
        cmap='viridis',
        alpha=0.6,
        s=10,
        rasterized=True
    )

    # Overlay GMM centers
    axes[1].scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='red',
        marker='X',
        s=200,
        edgecolors='white',
        linewidths=2,
        label='GMM Centers',
        zorder=10
    )

    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
    axes[1].set_title('GMM Prior - Responsibility Entropy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Entropy (higher = more uncertain)', fontsize=10)

    # Add summary text
    fig.text(0.5, 0.02,
             f'N={N} samples | K={K} components | '
             f'PC1+PC2 explained var: {(pca.explained_variance_ratio_[:2].sum())*100:.1f}% | '
             f'Mean entropy: {np.mean(resp_entropy):.3f}',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    print(f"  PCA explained variance: {pca.explained_variance_ratio_[:2]}")
    print(f"  Assignment distribution: {np.bincount(assignments, minlength=K)[:10]}... (first 10)")
    print(f"  Responsibility entropy: mean={np.mean(resp_entropy):.4f}, std={np.std(resp_entropy):.4f}")

    return fig


def compare_assignment_modes(
    params: GMMParams,
    x1_batch: Float[Array, "B H W C"],
    key: PRNGKeyArray,
    temperature: float = 1.0
):
    """
    Debug utility: Compare x0 samples from different assignment modes

    Args:
        params: GMMParams
        x1_batch: [B, H, W, C] sample batch
        key: JAX random key
        temperature: softmax temperature

    Returns:
        dict with statistics for each mode
    """
    print("\n" + "="*60)
    print("Comparing Assignment Modes")
    print("="*60)

    results = {}

    for mode in ['soft_sample', 'moment', 'hard']:
        key, subkey = jax.random.split(key)

        x0 = sample_gmm_x0(
            params,
            subkey,
            x1_batch,
            assign_mode=mode,
            temperature=temperature,
            debug=True
        )

        x0_flat = x0.reshape(x0.shape[0], -1)

        results[mode] = {
            'mean_norm': float(np.mean(np.linalg.norm(x0_flat, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(x0_flat, axis=1))),
            'mean_value': float(np.mean(x0_flat)),
            'std_value': float(np.std(x0_flat)),
        }

    print("\n" + "="*60)
    print("Summary Statistics by Mode:")
    print("="*60)
    for mode, stats in results.items():
        print(f"{mode:12s}: norm={stats['mean_norm']:.4f}±{stats['std_norm']:.4f}, "
              f"value={stats['mean_value']:.4f}±{stats['std_value']:.4f}")

    return results


if __name__ == "__main__":
    """Test GMM prior utilities"""
    print("Testing GMM Prior utilities...")

    # Create dummy GMM for testing
    K, D = 10, 4096  # 10 components, 32x32x4 latent
    H, W, C = 32, 32, 4

    dummy_params = GMMParams(
        pi=jnp.ones(K) / K,
        mu=jax.random.normal(jax.random.PRNGKey(0), (K, D)),
        var=jnp.ones((K, D)) * 0.5
    )

    # Test save/load
    print("\n1. Testing save/load...")
    save_gmm_prior(dummy_params, '/tmp/test_gmm.npz')
    loaded_params = load_gmm_prior('/tmp/test_gmm.npz')
    assert jnp.allclose(dummy_params.pi, loaded_params.pi), "Save/load failed!"
    print("   ✓ Save/load successful")

    # Test unconditional sampling
    print("\n2. Testing unconditional sampling...")
    key = jax.random.PRNGKey(42)
    x0_uncond = sample_gmm_unconditional(loaded_params, key, batch_size=8, shape=(H, W, C), debug=True)
    print(f"   ✓ Unconditional samples shape: {x0_uncond.shape}")

    # Test conditional sampling
    print("\n3. Testing conditional sampling (3 modes)...")
    x1_dummy = jax.random.normal(jax.random.PRNGKey(1), (4, H, W, C))

    for mode in ['soft_sample', 'moment', 'hard']:
        key, subkey = jax.random.split(key)
        x0 = sample_gmm_x0(loaded_params, subkey, x1_dummy, assign_mode=mode, debug=True)
        print(f"   ✓ Mode '{mode}': x0 shape={x0.shape}")

    # Test mode comparison
    print("\n4. Testing mode comparison...")
    stats = compare_assignment_modes(loaded_params, x1_dummy, key)

    print("\n✓ All tests passed!")
