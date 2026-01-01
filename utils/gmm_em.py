"""
Gaussian Mixture Model - Expectation Maximization (JAX Implementation)

Pure JAX implementation of diagonal-covariance GMM fitting for flow matching.
Optimized for numerical stability and efficiency.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional, Callable, Tuple, Dict
from jaxtyping import Array, Float, PRNGKeyArray


@dataclass
class GMMParams:
    """GMM parameters for diagonal covariance model"""
    pi: Float[Array, "K"]        # Component weights [K]
    mu: Float[Array, "K D"]      # Means [K, D]
    var: Float[Array, "K D"]     # Diagonal variances [K, D]

    def __post_init__(self):
        """Validate shapes and values"""
        K_pi = self.pi.shape[0]
        K_mu, D_mu = self.mu.shape
        K_var, D_var = self.var.shape

        assert K_pi == K_mu == K_var, f"K mismatch: pi={K_pi}, mu={K_mu}, var={K_var}"
        assert D_mu == D_var, f"D mismatch: mu={D_mu}, var={D_var}"
        assert jnp.allclose(jnp.sum(self.pi), 1.0), f"pi sum = {jnp.sum(self.pi)}, should be 1.0"
        assert jnp.all(self.pi >= 0), "pi should be non-negative"
        assert jnp.all(self.var > 0), "var should be positive"


def log_joint_diag(
    x: Float[Array, "B D"],
    pi: Float[Array, "K"],
    mu: Float[Array, "K D"],
    var: Float[Array, "K D"]
) -> Float[Array, "B K"]:
    """
    Compute log p(x, k) = log p(k) + log p(x | k) for diagonal covariance GMM

    Optimized implementation using broadcasting instead of explicit loops.

    Args:
        x: [B, D] data points
        pi: [K] component weights
        mu: [K, D] component means
        var: [K, D] diagonal variances

    Returns:
        logp: [B, K] log joint probabilities

    Formula:
        log p(x, k) = log π_k - D/2 log(2π) - 1/2 Σ_d log(σ²_{kd})
                      - 1/2 Σ_d (x_d - μ_{kd})² / σ²_{kd}
    """
    B, D = x.shape
    K = pi.shape[0]

    # Shape checks (debug)
    assert x.shape == (B, D), f"x shape {x.shape} != ({B}, {D})"
    assert pi.shape == (K,), f"pi shape {pi.shape} != ({K},)"
    assert mu.shape == (K, D), f"mu shape {mu.shape} != ({K}, {D})"
    assert var.shape == (K, D), f"var shape {var.shape} != ({K}, {D})"

    # log π_k: [K]
    log_pi = jnp.log(pi + 1e-10)

    # Constant term: -D/2 * log(2π)
    const_term = -0.5 * D * jnp.log(2 * jnp.pi)

    # log determinant term: -1/2 Σ_d log(σ²_{kd}) -> [K]
    log_det_term = -0.5 * jnp.sum(jnp.log(var + 1e-10), axis=1)  # [K]

    # Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ)
    # Compute per component k and per sample b
    # diff[b, k, d] = x[b, d] - mu[k, d]
    diff = x[:, None, :] - mu[None, :, :]  # [B, K, D]

    # Weighted squared distance: Σ_d (x_d - μ_{kd})² / σ²_{kd}
    mahal_dist = jnp.sum(diff**2 / (var[None, :, :] + 1e-10), axis=2)  # [B, K]

    # Combine all terms
    logp = log_pi[None, :] + const_term + log_det_term[None, :] - 0.5 * mahal_dist  # [B, K]

    # NaN check (debug)
    assert not jnp.any(jnp.isnan(logp)), "NaN detected in log_joint_diag"
    assert not jnp.any(jnp.isinf(logp)), "Inf detected in log_joint_diag"

    return logp


def e_step(
    logp: Float[Array, "B K"]
) -> Tuple[Float[Array, "B K"], float]:
    """
    E-step: Compute responsibilities using log-sum-exp for stability

    Args:
        logp: [B, K] log joint probabilities p(x, k)

    Returns:
        r: [B, K] responsibilities (normalized, sum to 1 over K)
        loglik: scalar mean log-likelihood

    Formula:
        r_{bk} = p(k | x_b) = p(x_b, k) / Σ_{k'} p(x_b, k')
        log-likelihood = mean_b log Σ_k p(x_b, k)
    """
    B, K = logp.shape

    # Log-sum-exp for numerical stability
    # log Σ_k exp(logp[b, k]) = log_sum_exp_k(logp[b, k])
    log_sum_exp = jax.scipy.special.logsumexp(logp, axis=1, keepdims=True)  # [B, 1]

    # Responsibilities: r[b, k] = exp(logp[b, k] - log_sum_exp[b])
    log_r = logp - log_sum_exp  # [B, K]
    r = jnp.exp(log_r)  # [B, K]

    # Log-likelihood: mean over samples
    loglik = float(jnp.mean(log_sum_exp))

    # Sanity checks (debug)
    assert r.shape == (B, K), f"r shape {r.shape} != ({B}, {K})"
    assert jnp.allclose(jnp.sum(r, axis=1), 1.0, atol=1e-5), "r should sum to 1 over K"
    assert not jnp.any(jnp.isnan(r)), "NaN in responsibilities"

    return r, loglik


def m_step(
    x: Float[Array, "B D"],
    r: Float[Array, "B K"],
    var_floor: float = 1e-6
) -> GMMParams:
    """
    M-step: Update GMM parameters

    Args:
        x: [B, D] data points
        r: [B, K] responsibilities
        var_floor: minimum variance (for numerical stability)

    Returns:
        Updated GMMParams

    Formula:
        N_k = Σ_b r_{bk}
        π_k = N_k / B
        μ_k = Σ_b r_{bk} x_b / N_k
        σ²_k = Σ_b r_{bk} (x_b - μ_k)² / N_k
    """
    B, D = x.shape
    K = r.shape[1]

    # Effective sample count per component: N_k = Σ_b r_{bk}
    Nk = jnp.sum(r, axis=0)  # [K]

    # Updated weights: π_k = N_k / B
    pi = Nk / B  # [K]

    # Updated means: μ_k = Σ_b r_{bk} x_b / N_k
    # r.T @ x: [K, B] @ [B, D] = [K, D]
    mu = (r.T @ x) / (Nk[:, None] + 1e-10)  # [K, D]

    # Updated variances: σ²_k = Σ_b r_{bk} (x_b - μ_k)² / N_k
    # diff[b, k, d] = x[b, d] - mu[k, d]
    diff = x[:, None, :] - mu[None, :, :]  # [B, K, D]

    # Weighted squared difference: r[b, k] * diff[b, k, d]²
    weighted_sq_diff = r[:, :, None] * diff**2  # [B, K, D]

    # Sum over samples: Σ_b r_{bk} (x_b - μ_k)²
    var = jnp.sum(weighted_sq_diff, axis=0) / (Nk[:, None] + 1e-10)  # [K, D]

    # Apply variance floor
    var = jnp.maximum(var, var_floor)

    # Normalize pi (should already be normalized, but for safety)
    pi = pi / (jnp.sum(pi) + 1e-10)

    params = GMMParams(pi=pi, mu=mu, var=var)

    # Debug checks
    assert not jnp.any(jnp.isnan(pi)), "NaN in updated pi"
    assert not jnp.any(jnp.isnan(mu)), "NaN in updated mu"
    assert not jnp.any(jnp.isnan(var)), "NaN in updated var"

    return params


def init_gmm_kmeans_plus(
    key: PRNGKeyArray,
    x: Float[Array, "N D"],
    K: int,
    var_floor: float = 1e-6
) -> GMMParams:
    """
    Initialize GMM using k-means++ strategy

    Args:
        key: JAX random key
        x: [N, D] data points
        K: number of components
        var_floor: minimum variance

    Returns:
        Initial GMMParams

    K-means++ initialization:
        - First center: random sample
        - Subsequent centers: sample proportional to squared distance to nearest center
    """
    N, D = x.shape

    centers = []

    # First center: random sample
    key, subkey = jax.random.split(key)
    first_idx = jax.random.randint(subkey, (), 0, N)
    centers.append(x[first_idx])

    # Subsequent centers
    for k in range(1, K):
        # Compute distances to nearest center
        centers_array = jnp.stack(centers, axis=0)  # [k, D]
        dists = jnp.sum((x[:, None, :] - centers_array[None, :, :])**2, axis=2)  # [N, k]
        min_dists = jnp.min(dists, axis=1)  # [N]

        # Sample next center proportional to squared distance
        key, subkey = jax.random.split(key)
        probs = min_dists / (jnp.sum(min_dists) + 1e-10)
        next_idx = jax.random.categorical(subkey, jnp.log(probs + 1e-10))
        centers.append(x[next_idx])

    # Initialize parameters
    mu = jnp.stack(centers, axis=0)  # [K, D]
    pi = jnp.ones(K) / K  # Uniform weights

    # Initialize variance as global variance
    global_var = jnp.var(x, axis=0)  # [D]
    var = jnp.tile(global_var[None, :], (K, 1))  # [K, D]
    var = jnp.maximum(var, var_floor)

    params = GMMParams(pi=pi, mu=mu, var=var)

    return params


def fit_gmm_em(
    x: Float[Array, "N D"],
    K: int,
    iters_max: int = 100,
    var_floor: float = 1e-6,
    early_stop: bool = True,
    patience: int = 5,
    tol: float = 1e-4,
    log_callback: Optional[Callable] = None,
    verbose: bool = True
) -> Tuple[GMMParams, Dict]:
    """
    Fit GMM using Expectation-Maximization algorithm

    Args:
        x: [N, D] data points
        K: number of components
        iters_max: maximum EM iterations
        var_floor: minimum variance for numerical stability
        early_stop: enable early stopping
        patience: stop if no improvement for N consecutive iterations
        tol: log-likelihood improvement threshold (absolute)
        log_callback: optional callback function(iter, metrics_dict)
        verbose: print progress

    Returns:
        params: Fitted GMMParams
        logs: dict with training history
    """
    N, D = x.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"Fitting GMM: N={N}, D={D}, K={K}")
        print(f"Early stop: {early_stop}, Patience: {patience}, Tol: {tol}")
        print(f"{'='*60}\n")

    # Initialize using k-means++
    key = jax.random.PRNGKey(42)
    params = init_gmm_kmeans_plus(key, x, K, var_floor)

    if verbose:
        print(f"Initialization complete (k-means++)")
        print(f"  Initial pi entropy: {-np.sum(params.pi * np.log(params.pi + 1e-10)):.4f}")
        print(f"  Initial var range: [{np.min(params.var):.6f}, {np.max(params.var):.6f}]\n")

    # EM loop
    loglik_history = []
    best_loglik = -np.inf
    patience_counter = 0

    for em_iter in range(iters_max):
        # E-step
        logp = log_joint_diag(x, params.pi, params.mu, params.var)
        r, loglik = e_step(logp)

        loglik_history.append(loglik)

        # Compute improvement
        improvement = loglik - loglik_history[-2] if em_iter > 0 else 0.0

        # M-step
        params = m_step(x, r, var_floor)

        # Compute additional metrics
        Nk = np.sum(r, axis=0)  # Component occupancy
        pi_entropy = -np.sum(params.pi * np.log(params.pi + 1e-10))
        resp_entropy = -np.sum(r * np.log(r + 1e-10), axis=1)  # [N]

        metrics = {
            'em_iter': em_iter,
            'loglik': loglik,
            'loglik_improve': improvement,
            'pi_entropy': float(pi_entropy),
            'min_Nk': float(np.min(Nk)),
            'max_Nk': float(np.max(Nk)),
            'mean_Nk': float(np.mean(Nk)),
            'empty_components': int(np.sum(Nk < 1.0)),
            'var_min': float(np.min(params.var)),
            'var_median': float(np.median(params.var)),
            'var_max': float(np.max(params.var)),
            'resp_entropy_mean': float(np.mean(resp_entropy)),
            'resp_entropy_std': float(np.std(resp_entropy)),
        }

        # Verbose logging
        if verbose and em_iter % 10 == 0:
            print(f"Iter {em_iter:3d}: LogLik={loglik:12.4f}, Improve={improvement:10.6f}, "
                  f"Empty={metrics['empty_components']}, Pi_H={pi_entropy:.4f}")

        # Callback for W&B logging
        if log_callback is not None:
            log_callback(em_iter, metrics)

        # Early stopping check
        if early_stop and em_iter > 0:
            if improvement < tol:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n[Early Stop] at iter {em_iter}")
                        print(f"  Final LogLik: {loglik:.4f}")
                        print(f"  Improvement: {improvement:.6f} < Tol: {tol}")
                    break
            else:
                patience_counter = 0
                best_loglik = loglik

    if verbose:
        print(f"\n{'='*60}")
        print(f"EM Finished: {len(loglik_history)} iterations")
        print(f"  Final LogLik: {loglik_history[-1]:.4f}")
        print(f"  Best LogLik: {best_loglik:.4f}")
        print(f"  Empty components: {metrics['empty_components']}/{K}")
        print(f"{'='*60}\n")

    # Return logs
    logs = {
        'loglik_history': loglik_history,
        'em_iters': len(loglik_history),
        'final_loglik': loglik_history[-1],
        'best_loglik': best_loglik,
        'final_metrics': metrics,
    }

    return params, logs


if __name__ == "__main__":
    """Test GMM EM on synthetic data"""
    print("Testing GMM EM implementation...")

    # Generate synthetic 2D data from 3 Gaussians
    np.random.seed(42)
    N_per_component = 1000

    # Component 1: centered at (0, 0)
    x1 = np.random.randn(N_per_component, 2) * 0.5

    # Component 2: centered at (5, 0)
    x2 = np.random.randn(N_per_component, 2) * 0.7 + np.array([5, 0])

    # Component 3: centered at (2.5, 4)
    x3 = np.random.randn(N_per_component, 2) * 0.6 + np.array([2.5, 4])

    x = np.concatenate([x1, x2, x3], axis=0)
    x = jnp.array(x)

    print(f"Data shape: {x.shape}")
    print(f"Data mean: {np.mean(x, axis=0)}")
    print(f"Data std: {np.std(x, axis=0)}\n")

    # Fit GMM
    params, logs = fit_gmm_em(
        x=x,
        K=3,
        iters_max=50,
        var_floor=1e-6,
        early_stop=True,
        patience=5,
        tol=1e-4,
        verbose=True
    )

    print("\nFitted GMM Parameters:")
    print(f"  Weights (pi): {params.pi}")
    print(f"  Means (mu):\n{params.mu}")
    print(f"  Variances (var):\n{params.var}")

    # Plot convergence
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Data and centers
    axes[0].scatter(x[:, 0], x[:, 1], alpha=0.3, s=10, label='Data')
    axes[0].scatter(params.mu[:, 0], params.mu[:, 1],
                   c='red', marker='X', s=200, edgecolors='black',
                   linewidths=2, label='GMM Centers')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    axes[0].set_title('Data and Fitted GMM Centers')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Log-likelihood convergence
    axes[1].plot(logs['loglik_history'], marker='o', markersize=4)
    axes[1].set_xlabel('EM Iteration')
    axes[1].set_ylabel('Log-Likelihood')
    axes[1].set_title('EM Convergence')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/mnt/c/Users/Admin/Desktop/code python/shortcut-models/gmm_em_test.png', dpi=150)
    print("\nTest plot saved to: gmm_em_test.png")
