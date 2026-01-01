\
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


AssignMode = Literal["soft_sample", "moment", "hard"]


@dataclass(frozen=True)
class GMMPrior:
    """
    Diagonal-covariance GMM in a flattened latent space of dimension D.

    pi: (K,) mixture weights (sum to 1)
    mu: (K, D)
    var: (K, D)  (diagonal variances, strictly positive)
    """
    pi: jnp.ndarray
    mu: jnp.ndarray
    var: jnp.ndarray

    @property
    def K(self) -> int:
        return int(self.pi.shape[0])

    @property
    def D(self) -> int:
        return int(self.mu.shape[1])


def _log_joint_diag(x: jnp.ndarray, pi: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    """
    Compute log p(k, x) for diagonal Gaussian components.

    x: (B, D)
    pi: (K,)
    mu: (K, D)
    var: (K, D)
    returns: (B, K)
    """
    # Stabilize
    var = jnp.maximum(var, 1e-12)
    log_pi = jnp.log(jnp.maximum(pi, 1e-30))

    log_var = jnp.log(var)
    inv_var = jnp.exp(-log_var)  # 1/var

    # quad = sum_d ( (x - mu)^2 / var )
    # Expand efficiently:
    # sum (x^2 * inv_var) - 2 sum (x * mu * inv_var) + sum (mu^2 * inv_var)
    x2 = x * x  # (B,D)
    term1 = x2 @ inv_var.T  # (B,K)
    term2 = x @ (mu * inv_var).T  # (B,K)
    mu2 = jnp.sum(mu * mu * inv_var, axis=-1)  # (K,)
    quad = term1 - 2.0 * term2 + mu2[None, :]

    logdet = jnp.sum(log_var, axis=-1)  # (K,)
    D = x.shape[-1]
    log_norm = -0.5 * (logdet[None, :] + D * jnp.log(2.0 * jnp.pi))
    log_prob = log_norm - 0.5 * quad  # (B,K)

    return log_pi[None, :] + log_prob


@jax.jit
def posterior_logp(prior: GMMPrior, x: jnp.ndarray) -> jnp.ndarray:
    """
    log p(k | x) up to normalization (actually returns log p(k,x) - log p(x)).
    x: (B,D)
    returns: (B,K) log responsibilities.
    """
    log_joint = _log_joint_diag(x, prior.pi, prior.mu, prior.var)  # (B,K)
    log_px = logsumexp(log_joint, axis=-1, keepdims=True)          # (B,1)
    return log_joint - log_px


@jax.jit
def responsibilities(prior: GMMPrior, x: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """
    r = softmax(log p(k|x) / T)
    """
    log_r = posterior_logp(prior, x)
    if temperature != 1.0:
        log_r = log_r / temperature
    # softmax on log_r
    log_r = log_r - logsumexp(log_r, axis=-1, keepdims=True)
    return jnp.exp(log_r)


@jax.jit
def sample_x0_conditional(
    prior: GMMPrior,
    key: jax.Array,
    x1: jnp.ndarray,
    assign_mode: AssignMode = "soft_sample",
    temperature: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample x0 given x1 by computing posterior responsibilities p(k|x1)
    and then sampling from the chosen component (or moment-matching).

    x1: (B,D)
    returns: (x0: (B,D), k: (B,) int32 chosen component (for logging/debug))
    """
    B = x1.shape[0]
    r = responsibilities(prior, x1, temperature=temperature)  # (B,K)

    if assign_mode == "moment":
        # Moment-matched Gaussian:
        # mu_bar = sum_k r_k mu_k
        # var_bar = sum_k r_k (var_k + mu_k^2) - mu_bar^2
        mu_bar = r @ prior.mu  # (B,D)
        second = r @ (prior.var + prior.mu * prior.mu)  # (B,D)
        var_bar = jnp.maximum(second - mu_bar * mu_bar, 1e-12)
        key_eps, = jax.random.split(key, 1)
        eps = jax.random.normal(key_eps, (B, prior.D), dtype=mu_bar.dtype)
        x0 = mu_bar + jnp.sqrt(var_bar) * eps
        k = jnp.argmax(r, axis=-1).astype(jnp.int32)
        return x0, k

    if assign_mode == "hard":
        k = jnp.argmax(r, axis=-1).astype(jnp.int32)
    else:
        # soft_sample (default): categorical sample from r
        key_cat, key_eps = jax.random.split(key, 2)
        k = jax.random.categorical(key_cat, jnp.log(jnp.maximum(r, 1e-30)), axis=-1).astype(jnp.int32)

    # Gather component params
    mu_k = jnp.take(prior.mu, k, axis=0)    # (B,D)
    var_k = jnp.take(prior.var, k, axis=0)  # (B,D)
    key_eps, = jax.random.split(key, 1)
    eps = jax.random.normal(key_eps, (B, prior.D), dtype=mu_k.dtype)
    x0 = mu_k + jnp.sqrt(jnp.maximum(var_k, 1e-12)) * eps
    return x0, k


@jax.jit
def sample_x0_uncond(prior: GMMPrior, key: jax.Array, B: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Unconditional x0 sampling: k ~ Cat(pi), x0 ~ N(mu_k, var_k)
    returns x0: (B,D) and k: (B,)
    """
    key_cat, key_eps = jax.random.split(key, 2)
    k = jax.random.categorical(key_cat, jnp.log(jnp.maximum(prior.pi, 1e-30)), shape=(B,)).astype(jnp.int32)
    mu_k = jnp.take(prior.mu, k, axis=0)
    var_k = jnp.take(prior.var, k, axis=0)
    eps = jax.random.normal(key_eps, (B, prior.D), dtype=mu_k.dtype)
    x0 = mu_k + jnp.sqrt(jnp.maximum(var_k, 1e-12)) * eps
    return x0, k


def save_prior_npz(path: str, prior: GMMPrior) -> None:
    import numpy as np
    np.savez(path,
             pi=np.array(prior.pi),
             mu=np.array(prior.mu),
             var=np.array(prior.var))


def load_prior_npz(path: str) -> GMMPrior:
    import numpy as np
    data = np.load(path)
    pi = jnp.asarray(data["pi"])
    mu = jnp.asarray(data["mu"])
    var = jnp.asarray(data["var"])
    return GMMPrior(pi=pi, mu=mu, var=var)
