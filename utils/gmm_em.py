\
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .gmm_prior import GMMPrior


@dataclass
class EMConfig:
    K: int
    max_iters: int = 50
    var_floor: float = 1e-5
    tol: float = 1e-4
    patience: int = 3
    min_weight: float = 1e-4  # avoid zero-weight components
    reinit_min_Nk: float = 1.0  # if Nk < this, re-init component


def _log_joint_diag(x: jnp.ndarray, pi: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    var = jnp.maximum(var, 1e-12)
    log_pi = jnp.log(jnp.maximum(pi, 1e-30))
    log_var = jnp.log(var)
    inv_var = jnp.exp(-log_var)

    x2 = x * x
    term1 = x2 @ inv_var.T
    term2 = x @ (mu * inv_var).T
    mu2 = jnp.sum(mu * mu * inv_var, axis=-1)
    quad = term1 - 2.0 * term2 + mu2[None, :]

    logdet = jnp.sum(log_var, axis=-1)
    D = x.shape[-1]
    log_norm = -0.5 * (logdet[None, :] + D * jnp.log(2.0 * jnp.pi))
    log_prob = log_norm - 0.5 * quad
    return log_pi[None, :] + log_prob


@jax.jit
def em_batch_stats(
    x: jnp.ndarray,  # (B,D)
    pi: jnp.ndarray, # (K,)
    mu: jnp.ndarray, # (K,D)
    var: jnp.ndarray,# (K,D)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (Nk, sum_x, sum_x2, loglik_sum) for one batch.
    Nk: (K,)
    sum_x: (K,D)
    sum_x2: (K,D)
    loglik_sum: scalar
    """
    log_joint = _log_joint_diag(x, pi, mu, var)                 # (B,K)
    log_px = logsumexp(log_joint, axis=-1, keepdims=True)       # (B,1)
    r = jnp.exp(log_joint - log_px)                             # (B,K)
    Nk = jnp.sum(r, axis=0)                                     # (K,)
    sum_x = r.T @ x                                             # (K,D)
    sum_x2 = r.T @ (x * x)                                      # (K,D)
    loglik_sum = jnp.sum(log_px)                                # scalar
    return Nk, sum_x, sum_x2, loglik_sum


def init_params_from_points(points: jnp.ndarray, K: int, var_floor: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    points: (N,D) (N >= K)
    """
    # deterministic init: take first K points as means
    mu = points[:K]
    # global variance init
    v = jnp.var(points, axis=0, keepdims=True)
    var = jnp.tile(jnp.maximum(v, var_floor), (K, 1))
    pi = jnp.ones((K,), dtype=points.dtype) / K
    return pi, mu, var


def fit_gmm_em_streaming(
    batch_iterator,
    num_batches: int,
    D: int,
    cfg: EMConfig,
    rng: jax.Array,
    init_points: Optional[jnp.ndarray] = None,
    on_iter_end=None,
    verbose: bool = True,
) -> Tuple[GMMPrior, Dict[str, jnp.ndarray]]:
    """
    Streaming EM over an iterator that yields (B,D) jnp arrays.

    - Only uses num_batches from batch_iterator per EM iteration.
    - Uses diagonal covariance.
    - Supports early stopping.
    - on_iter_end: optional callback(iter_idx, logs_dict) executed on host.
    - verbose: if True, show tqdm progress bars (only on process 0)

    Returns: (prior, logs)
    logs contains arrays of per-iter values (loglik, etc.)
    """
    K = cfg.K
    logs = {"loglik": [], "min_Nk": [], "max_Nk": []}

    # Collect init points if not provided
    if init_points is None:
        # Grab enough points for initialization
        buf = []
        need = K
        while need > 0:
            x = next(batch_iterator)  # (B,D)
            take = min(int(x.shape[0]), need)
            buf.append(x[:take])
            need -= take
        init_points = jnp.concatenate(buf, axis=0)
    else:
        init_points = init_points[:max(K, 1)]

    pi, mu, var = init_params_from_points(init_points, K, cfg.var_floor)
    prev_loglik = None
    bad_iters = 0

    # Setup progress bar for EM iterations
    pbar_outer = None
    if verbose and jax.process_index() == 0:
        try:
            import tqdm
            pbar_outer = tqdm.tqdm(total=cfg.max_iters, desc="EM iterations", position=0)
        except ImportError:
            pass  # tqdm not available, continue without progress bar

    try:
        for it in range(cfg.max_iters):
            Nk_acc = jnp.zeros((K,), dtype=jnp.float32)
            sum_acc = jnp.zeros((K, D), dtype=jnp.float32)
            sum2_acc = jnp.zeros((K, D), dtype=jnp.float32)
            loglik_acc = jnp.array(0.0, dtype=jnp.float32)
            n_seen = 0

            # Setup progress bar for E-step batches
            pbar_inner = None
            if verbose and jax.process_index() == 0 and pbar_outer is not None:
                try:
                    import tqdm
                    pbar_inner = tqdm.tqdm(total=num_batches, desc=f"  E-step",
                                          position=1, leave=False)
                except ImportError:
                    pass

            try:
                for _ in range(num_batches):
                    x = next(batch_iterator)  # (B,D)
                    Nk_b, sum_b, sum2_b, ll_b = em_batch_stats(x, pi, mu, var)
                    Nk_acc = Nk_acc + Nk_b
                    sum_acc = sum_acc + sum_b
                    sum2_acc = sum2_acc + sum2_b
                    loglik_acc = loglik_acc + ll_b
                    n_seen += int(x.shape[0])

                    if pbar_inner is not None:
                        pbar_inner.update(1)
            finally:
                if pbar_inner is not None:
                    pbar_inner.close()

            # M-step
            Nk = jnp.maximum(Nk_acc, cfg.reinit_min_Nk)
            pi_new = Nk / jnp.sum(Nk)
            # Avoid degenerate components
            pi_new = jnp.maximum(pi_new, cfg.min_weight)
            pi_new = pi_new / jnp.sum(pi_new)

            mu_new = sum_acc / Nk[:, None]
            ex2 = sum2_acc / Nk[:, None]
            var_new = jnp.maximum(ex2 - mu_new * mu_new, cfg.var_floor)

            # Re-init components that were effectively empty BEFORE clamping
            empty = Nk_acc < cfg.reinit_min_Nk
            if bool(jnp.any(empty)):
                # Use init_points to re-seed means deterministically
                repl = init_points
                repl = jnp.pad(repl, ((0, max(0, K - repl.shape[0])), (0, 0)), mode="wrap")
                mu_new = jnp.where(empty[:, None], repl[:K], mu_new)
                var_new = jnp.where(empty[:, None], jnp.ones_like(var_new) * cfg.var_floor, var_new)
                # keep pi_new as-is (already normalized)

            pi, mu, var = pi_new.astype(jnp.float32), mu_new.astype(jnp.float32), var_new.astype(jnp.float32)

            loglik_mean = loglik_acc / max(n_seen, 1)
            logs["loglik"].append(loglik_mean)
            logs["min_Nk"].append(jnp.min(Nk_acc))
            logs["max_Nk"].append(jnp.max(Nk_acc))

            # Update outer progress bar with metrics
            if pbar_outer is not None:
                pbar_outer.set_postfix({
                    'loglik': f'{float(loglik_mean):.4f}',
                    'min_Nk': f'{float(logs["min_Nk"][-1]):.1f}',
                    'max_Nk': f'{float(logs["max_Nk"][-1]):.1f}'
                })
                pbar_outer.update(1)

            logs_iter = {
                "loglik": loglik_mean,
                "min_Nk": logs["min_Nk"][-1],
                "max_Nk": logs["max_Nk"][-1],
            }
            if on_iter_end is not None:
                on_iter_end(it, logs_iter)

            # Early stop
            if prev_loglik is not None:
                improve = loglik_mean - prev_loglik
                if float(improve) < cfg.tol:
                    bad_iters += 1
                else:
                    bad_iters = 0
                if bad_iters >= cfg.patience:
                    break
            prev_loglik = loglik_mean
    finally:
        if pbar_outer is not None:
            pbar_outer.close()

    # Stack logs
    logs = {k: jnp.stack(v) if len(v) else jnp.array([]) for k, v in logs.items()}
    prior = GMMPrior(pi=pi, mu=mu, var=var)
    return prior, logs
