from jax._src.nn.initializers import _compute_fans
from jax._src import dtypes
from jax._src import core
from math_utils import get_2d_sincos_pos_embed, modulate
import math
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any


# class ConditionalInstanceNorm2dNHWC(nn.Module):
#     num_channels: int
#     special_t: Sequence[float]      # ví dụ [0.0, 0.25, 0.5, 0.75, 1.0]
#     eps: float = 1e-5
#     use_affine: bool = True

#     # thêm cấu hình nhỏ cho t-embedding
#     time_embed_dim: int = 256       # giống frequency_embedding_size
#     time_max_period: float = 10000.0
#     mlp_hidden_dim: int = 64
#     t_scale: float = 0.5

#     @nn.compact
#     def __call__(self, x, t):
#         # x: [B,H,W,C], t: [B] (giống DiT.__call__)
#         # 1) Instance norm cơ bản

#         # ========= 0) T-EMBED GIỐNG KIỂU TimestepEmbedder =========
#         def timestep_embedding(t_scalar, dim, max_period):
#             # t_scalar: [B]
#             t_scalar = jax.lax.convert_element_type(t_scalar, jnp.float32)
#             half = dim // 2
#             freqs = jnp.exp(
#                 -math.log(max_period)
#                 * jnp.arange(half, dtype=jnp.float32)
#                 / half
#             )  # [half]
#             args = t_scalar[:, None] * freqs[None, :]  # [B, half]
#             emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
#             return emb  # [B, dim]

#         mean = jnp.mean(x, axis=(1, 2), keepdims=True)
#         var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
#         x_norm = (x - mean) / jnp.sqrt(var + self.eps)

#         # 2) Affine phụ thuộc t: γ_b = 1 + α Δγ_b, β_b = α Δβ_b
#         x_tilde = x_norm
#         if self.use_affine:
#             B = x.shape[0]

#             # Embed t → [B, time_embed_dim]
#             t_emb = timestep_embedding(
#                 t, self.time_embed_dim, self.time_max_period)

#             # MLP nhỏ để lấy Δγ, Δβ
#             h = nn.Dense(
#                 features=self.mlp_hidden_dim,
#                 kernel_init=nn.initializers.normal(0.02),
#             )(t_emb)
#             h = nn.silu(h)
#             h = nn.Dense(
#                 features=2 * self.num_channels,
#                 kernel_init=nn.initializers.normal(0.02),
#             )(h)                          # [B, 2C]

#             delta_gamma, delta_beta = jnp.split(h, 2, axis=-1)  # [B,C] mỗi bên

#             # scale nhỏ để ổn định (α)
#             delta_gamma = self.t_scale * delta_gamma
#             delta_beta = self.t_scale * delta_beta

#             # [B,1,1,C]
#             gamma = 1.0 + delta_gamma.reshape(B, 1, 1, self.num_channels)
#             beta = delta_beta.reshape(B, 1, 1, self.num_channels)  # [B,1,1,C]

#             x_tilde = x_norm * gamma + beta
#         # nếu use_affine = False thì x_tilde = x_norm

#         # 3) Gating theo t đặc biệt (giữ y nguyên idea cũ)
#         special_t = jnp.asarray(self.special_t, dtype=t.dtype)       # [K]
#         mask = (t[:, None] == special_t[None, :]).any(axis=-1)       # [B]
#         # [B,1,1,1]
#         mask = mask[:, None, None, None]

#         # 4) Compute masked MSE difference (so với x_tilde – tức norm + affine)
#         sq_err = (x - x_tilde) ** 2                          # [B,H,W,C]
#         mse_per_sample = sq_err.mean(axis=(1, 2, 3))         # [B]

#         mask_f = mask.astype(jnp.float32)                    # [B,1,1,1]
#         denom = jnp.maximum(mask_f.sum(), 1.0)

#         masked_avg_mse = (mse_per_sample * mask_f.reshape(-1)).sum() / denom
#         avg_mse = jnp.mean(mse_per_sample)
#         norm_percentage = jnp.mean(mask_f)

#         # 5) Chỉ norm tại các t thuộc special_t
#         y = jnp.where(mask, x_tilde, x)
#         return y, masked_avg_mse, avg_mse, norm_percentage


class ConditionalInstanceNorm2dNHWC(nn.Module):
    num_channels: int
    special_t: Sequence[float]      # ví dụ [0.0, 0.25, 0.5, 0.75, 1.0]
    eps: float = 1e-5
    use_affine: bool = True

    @nn.compact
    def __call__(self, x, t):
        # x: [B,H,W,C], t: [B] (giống DiT.__call__)
        # 1) Instance norm cơ bản
        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        # 2) Affine learnable (gamma, beta)
        if self.use_affine:
            gamma = self.param(
                "gamma",
                nn.initializers.ones,
                (1, 1, 1, self.num_channels),
            )
            beta = self.param(
                "beta",
                nn.initializers.zeros,
                (1, 1, 1, self.num_channels),
            )
            x_norm = x_norm * gamma + beta

        # 3) Gating theo t đặc biệt
        special_t = jnp.asarray(self.special_t, dtype=t.dtype)       # [K]
        # diff = jnp.abs(t[:, None] - special_t[None, :])              # [B,K]
        # mask = (diff < 1e-6).any(axis=-1)                            # [B]
        mask = (t[:, None] == special_t[None, :]).any(axis=-1)
        # [B,1,1,1]
        mask = mask[:, None, None, None]

        # 4) Compute masked MSE difference
        sq_err = (x - x_norm) ** 2                            # [B,H,W,C]
        mse_per_sample = sq_err.mean(axis=(1, 2, 3))            # [B]
        mask_f = mask.astype(jnp.float32)              # [B]
        denom = jnp.maximum(mask_f.sum(), 1.0)         # tránh chia 0

        masked_avg_mse = (mse_per_sample * mask_f).sum() / denom
        avg_mse = jnp.mean(mse_per_sample)
        norm_percentage = jnp.mean(mask_f)

        # Chỉ norm nếu t thuộc special_t, còn lại giữ nguyên x
        return jnp.where(mask, x_norm, x), masked_avg_mse, avg_mse, norm_percentage



class ConditionalBatchNormSpecialT(nn.Module):
    """
    BatchNorm trên NHWC dành cho diffusion shortcut:

    - x: [B,H,W,C], t: [B]
    - Chỉ chuẩn hoá sample có t ∈ special_t.
    - Mỗi τ_k có EMA riêng: mean[k,1,1,C], var[k,1,1,C].
    - Train: dùng batch stats để update EMA, nhưng NORM luôn dùng EMA.
    - Eval: không update, chỉ dùng EMA.
    """
    num_channels: int
    special_t: Sequence[float]
    eps: float = 1e-5
    momentum: float = 0.1
    use_affine: bool = True
    num_channels: int

    @nn.compact
    def __call__(self, x: Array, t: Array):
        """
        x: [B,H,W,C]
        t: [B]
        Trả về: (x_out, masked_avg_mse, avg_mse, norm_percentage)
        """
        B, H, W, C = x.shape
        assert C == self.num_channels, f"num_channels mismatch: {C} vs {self.num_channels}"

        # 1) special_t & mask
        special_t = jnp.asarray(self.special_t, dtype=t.dtype)  # [K]
        K = special_t.shape[0]

        # mask_bk[b,k] = True nếu t[b] == τ_k (xấp xỉ)
        diff = jnp.abs(t[:, None] - special_t[None, :])         # [B,K]
        mask_bk = diff < 1e-6                                   # [B,K] bool
        mask_fk = mask_bk.astype(x.dtype)                       # [B,K] float

        # 2) EMA stats: [K,1,1,C]
        mu_ema = self.variable(
            "batch_stats", "mean",
            lambda: jnp.zeros((K, 1, 1, C), dtype=x.dtype),
        )
        var_ema = self.variable(
            "batch_stats", "var",
            lambda: jnp.ones((K, 1, 1, C), dtype=x.dtype),
        )

        is_training = self.is_mutable_collection("batch_stats")

        # 3) Nếu train: tính batch stats để update EMA
        if is_training:
            # broadcast mask ra [B,H,W,C,K]
            mask_full = mask_fk[:, None, None, None, :]                 # [B,1,1,1,K]
            x_exp = x[..., None]                                        # [B,H,W,C,1]
            mask_full = mask_full * jnp.ones_like(x_exp)                # [B,H,W,C,K]

            # sum_x, count per (C,K)
            sum_x = (x_exp * mask_full).sum(axis=(0, 1, 2))             # [C,K]
            count = mask_full.sum(axis=(0, 1, 2))                       # [C,K]
            count_safe = jnp.maximum(count, 1.0)

            mean_ck = sum_x / count_safe                                # [C,K]

            # var: E[(x - mean)^2]
            mean_exp = mean_ck[None, None, None, :, :]                  # [1,1,1,C,K]
            diff_sq = (x_exp - mean_exp) ** 2                           # [B,H,W,C,K]
            sum_diff2 = (diff_sq * mask_full).sum(axis=(0, 1, 2))       # [C,K]
            var_ck = sum_diff2 / count_safe                             # [C,K]

            # chuyển [C,K] -> [K,1,1,C]
            mean_batch = jnp.transpose(mean_ck, (1, 0))[:, None, None, :]  # [K,1,1,C]
            var_batch = jnp.transpose(var_ck, (1, 0))[:, None, None, :]    # [K,1,1,C]

            # τ_k nào thực sự có sample?
            has_data_k = (mask_fk.sum(axis=0) > 0).astype(x.dtype)      # [K]
            has_data = has_data_k[:, None, None, None]                  # [K,1,1,1]

            m = jnp.asarray(self.momentum, dtype=x.dtype)

            mu_new = (1.0 - m * has_data) * mu_ema.value + m * has_data * mean_batch
            var_new = (1.0 - m * has_data) * var_ema.value + m * has_data * var_batch

            mu_ema.value = mu_new
            var_ema.value = var_new

        # 4) NORM LUÔN BẰNG EMA stats
        mu = mu_ema.value   # [K,1,1,C]
        var = var_ema.value # [K,1,1,C]

        # tạo x_norm_k cho từng τ_k: [K,B,H,W,C]
        x_b = x[None, ...]                                            # [1,B,H,W,C]
        x_centered = x_b - mu[:, None, :, :, :]                       # [K,B,H,W,C]
        x_hat_k = x_centered / jnp.sqrt(var[:, None, :, :, :] + self.eps)  # [K,B,H,W,C]

        # combine theo mask: mỗi sample chỉ thuộc tối đa 1 τ_k
        mask_kb = mask_bk.T[:, None, :, None, None]                   # [K,1,B,1,1]
        mask_kb = mask_kb * jnp.ones_like(x_hat_k)                    # [K,B,H,W,C]
        x_norm = (mask_kb * x_hat_k).sum(axis=0)                      # [B,H,W,C]

        # sample không thuộc special_t: giữ nguyên x
        mask_any = mask_bk.any(axis=1)                                # [B]
        mask_any_bc = mask_any[:, None, None, None]                   # [B,1,1,1]
        x_after = jnp.where(mask_any_bc, x_norm, x)                   # [B,H,W,C]

        # 5) Affine γ, β
        if self.use_affine:
            gamma = self.param("gamma", nn.initializers.ones, (1, 1, 1, C))
            beta = self.param("beta", nn.initializers.zeros, (1, 1, 1, C))
            y = x_after * gamma + beta
        else:
            y = x_after

        # 6) Logging các thống kê (giữ API như InstanceNorm cũ)
        sq_err = (x - x_after) ** 2                                   # [B,H,W,C]
        mse_per_sample = sq_err.mean(axis=(1, 2, 3))                  # [B]
        mask_f_any = mask_any.astype(x.dtype)
        denom = jnp.maximum(mask_f_any.sum(), 1.0)

        masked_avg_mse = (mse_per_sample * mask_f_any).sum() / denom
        avg_mse = mse_per_sample.mean()
        norm_percentage = mask_f_any.mean()

        return y, masked_avg_mse, avg_mse, norm_percentage