"""
Loss functions for training.
"""

import jax
import jax.numpy as jnp


def normalized_residual_mse(v_pred, v_true, eps=1e-4, s_min=0.3):
    """
    Normalized residual MSE loss: ||（v_pred - v_true) / s||^2

    where s = max(sqrt(mean(v_true^2) + eps), s_min) with stopgrad

    This loss normalizes the residual by the per-sample RMS of v_true,
    which helps when v_true has varying magnitudes across samples.
    The clamp s_min prevents extreme weights when v_true is very small.

    Args:
        v_pred: Predicted velocity, shape (B, H, W, C) or (B, D)
        v_true: Ground truth velocity, same shape as v_pred
        eps: Small constant for numerical stability (default: 1e-4)
        s_min: Minimum scale clamp to prevent extreme weights (default: 0.3)
               With s_min=0.3, max weight ~ 1/0.3^2 ≈ 11.1

    Returns:
        loss: Scalar loss value
        info: Dict with debug metrics:
            - loss/v_rms_mean: Mean RMS of v_true across batch
            - loss/scale_mean: Mean scale factor s across batch
            - loss/scale_min: Minimum scale factor s in batch
    """
    # Compute per-sample RMS: (1,2,3) for (B,H,W,C) or (1,) for (B,D)
    axes = tuple(range(1, v_true.ndim))

    # s_i = sqrt(mean(v_true_i^2) + eps) for each sample i
    s = jnp.sqrt(jnp.mean(jnp.square(v_true), axis=axes) + eps)  # (B,)

    # Clamp to minimum scale
    s = jnp.maximum(s, s_min)

    # Stop gradient to prevent backprop through normalization
    s = jax.lax.stop_gradient(s)

    # Broadcast s to v shape: (B,) -> (B,1,1,1) or (B,1)
    while s.ndim < v_true.ndim:
        s = s[..., None]

    # Normalized residual
    err = (v_pred - v_true) / s

    # MSE of normalized residual
    loss = jnp.mean(jnp.square(err))

    # Debug metrics
    s_squeezed = s.reshape(-1)  # Back to (B,) for statistics
    v_rms = jnp.sqrt(jnp.mean(jnp.square(v_true), axis=axes) + eps)  # (B,)

    info = {
        "loss/v_rms_mean": jnp.mean(v_rms),
        "loss/scale_mean": jnp.mean(s_squeezed),
        "loss/scale_min": jnp.min(s_squeezed),
    }

    return loss, info
