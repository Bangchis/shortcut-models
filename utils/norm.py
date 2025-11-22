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
from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn

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
    # số kênh C của latent x (NHWC)
    num_channels: int
    # list các timestep đặc biệt, ví dụ (0.25, 0.5, 0.75)
    special_t: Sequence[float]
    eps: float = 1e-5
    momentum: float = 0.1
    use_affine: bool = True

    @nn.compact
    def __call__(self, x, t, use_running_average: bool = False):
        """
        x: (B, H, W, C)
        t: (B,) timesteps liên tục [0,1]
        use_running_average=False: train → dùng batch stats + update EMA
        use_running_average=True:  eval  → chỉ dùng EMA
        """

        B, H, W, C = x.shape
        assert C == self.num_channels

        # Tham số gamma, beta giống BatchNorm thường
        if self.use_affine:
            gamma = self.param("gamma", nn.initializers.ones, (C,))
            beta = self.param("beta", nn.initializers.zeros, (C,))
        else:
            gamma = None
            beta = None

        K = len(self.special_t)
        special_t = jnp.asarray(self.special_t, dtype=x.dtype)  # (K,)

        # EMA stats cho từng τ_k: shape (K, C)
        running_mean = self.variable(
            "batch_stats",
            "mean",
            lambda: jnp.zeros((K, C), dtype=x.dtype),
        )
        running_var = self.variable(
            "batch_stats",
            "var",
            lambda: jnp.ones((K, C), dtype=x.dtype),
        )

        # Diagnostic: Check batch_stats for NaN/Inf/invalid values
        if jax.process_index() == 0:
            import numpy as np
            rm_cpu = np.array(running_mean.value)
            rv_cpu = np.array(running_var.value)
            print(f"[BN-DEBUG] BatchStats check: K={K}, C={C}, mode={'EVAL' if use_running_average else 'TRAIN'}", flush=True)
            print(f"[BN-DEBUG] running_mean: shape={rm_cpu.shape}, has_nan={np.isnan(rm_cpu).any()}, has_inf={np.isinf(rm_cpu).any()}", flush=True)
            print(f"[BN-DEBUG] running_var: shape={rv_cpu.shape}, has_nan={np.isnan(rv_cpu).any()}, has_inf={np.isinf(rv_cpu).any()}", flush=True)

            # Check per special_t
            for k in range(K):
                mean_k = rm_cpu[k]
                var_k = rv_cpu[k]
                print(f"[BN-DEBUG] special_t[{k}]={self.special_t[k]:.2f}: mean min={mean_k.min():.4e}, max={mean_k.max():.4e}, mean={mean_k.mean():.4e}", flush=True)
                print(f"[BN-DEBUG] special_t[{k}]={self.special_t[k]:.2f}: var min={var_k.min():.4e}, max={var_k.max():.4e}, mean={var_k.mean():.4e}", flush=True)

                num_negative_var = np.sum(var_k < 0)
                num_zero_var = np.sum(np.abs(var_k) < 1e-10)
                num_large_var = np.sum(var_k > 1e6)
                print(f"[BN-DEBUG] special_t[{k}]: num_negative_var={num_negative_var}, num_zero_var={num_zero_var}, num_large_var={num_large_var}", flush=True)

                if var_k.max() > 0:
                    var_cond = var_k.max() / max(var_k.min(), 1e-10)
                    print(f"[BN-DEBUG] special_t[{k}]: variance condition_number={var_cond:.4e}", flush=True)

        # Xác định sample nào rơi vào timestep đặc biệt nào
        t = t.astype(x.dtype)                      # (B,)
        diff = jnp.abs(t[:, None] - special_t[None, :])  # (B,K)
        is_special = diff < 1e-6                  # (B,K) bool

        x_in = x
        x_out = x

        # lặp qua từng τ_k
        for k in range(K):
            mask_b = is_special[:, k].astype(
                x.dtype).reshape(B, 1, 1, 1)  # (B,1,1,1)

            # tổng / đếm với mask
            denom = jnp.sum(mask_b)                             # scalar
            sum_x = jnp.sum(mask_b * x_in, axis=(0, 1, 2))      # (C,)
            sum_x2 = jnp.sum(mask_b * (x_in ** 2), axis=(0, 1, 2))

            count = jnp.maximum(denom, 1.0)
            mean_batch = sum_x / count                          # (C,)
            var_batch = sum_x2 / count - mean_batch ** 2        # (C,)

            has_sample = denom > 0.0

            # stats dùng để chuẩn hóa
            mean_used_train = jnp.where(
                has_sample, mean_batch, running_mean.value[k])
            var_used_train = jnp.where(
                has_sample, var_batch, running_var.value[k])

            mean_used = jnp.where(
                use_running_average,
                running_mean.value[k],
                mean_used_train,
            )
            var_used = jnp.where(
                use_running_average,
                running_var.value[k],
                var_used_train,
            )

            # update EMA chỉ khi train + có sample
            if not use_running_average:
                new_mean_k = jnp.where(
                    has_sample,
                    (1.0 - self.momentum) *
                    running_mean.value[k] + self.momentum * mean_batch,
                    running_mean.value[k],
                )
                new_var_k = jnp.where(
                    has_sample,
                    (1.0 - self.momentum) *
                    running_var.value[k] + self.momentum * var_batch,
                    running_var.value[k],
                )

                # Diagnostic: Log EMA updates
                if jax.process_index() == 0 and has_sample:
                    import numpy as np
                    old_mean = np.array(running_mean.value[k])
                    old_var = np.array(running_var.value[k])
                    new_mean_cpu = np.array(new_mean_k)
                    new_var_cpu = np.array(new_var_k)
                    batch_mean_cpu = np.array(mean_batch)
                    batch_var_cpu = np.array(var_batch)

                    mean_change = np.abs(new_mean_cpu - old_mean).max()
                    var_change = np.abs(new_var_cpu - old_var).max()

                    print(f"[BN-DEBUG] EMA update k={k}, denom={float(denom):.1f}", flush=True)
                    print(f"[BN-DEBUG]   batch_mean: min={batch_mean_cpu.min():.4e}, max={batch_mean_cpu.max():.4e}, has_nan={np.isnan(batch_mean_cpu).any()}", flush=True)
                    print(f"[BN-DEBUG]   batch_var: min={batch_var_cpu.min():.4e}, max={batch_var_cpu.max():.4e}, has_nan={np.isnan(batch_var_cpu).any()}", flush=True)
                    print(f"[BN-DEBUG]   mean_change={mean_change:.4e}, var_change={var_change:.4e}", flush=True)

                    if np.isnan(new_mean_cpu).any() or np.isnan(new_var_cpu).any():
                        print(f"[BN-DEBUG]   WARNING: NaN detected in new EMA stats!", flush=True)

                running_mean.value = running_mean.value.at[k].set(new_mean_k)
                running_var.value = running_var.value.at[k].set(new_var_k)

            # chuẩn hóa tất cả, rồi chỉ ghi đè lên sample thuộc τ_k
            mean_broadcast = mean_used[None, None, None, :]
            var_broadcast = var_used[None, None, None, :]

            # Diagnostic: Check normalization stats being used
            if jax.process_index() == 0 and jnp.sum(mask_b) > 0:
                import numpy as np
                mean_used_cpu = np.array(mean_used)
                var_used_cpu = np.array(var_used)
                print(f"[BN-DEBUG] Normalizing k={k}, num_samples={float(jnp.sum(mask_b)):.1f}", flush=True)
                print(f"[BN-DEBUG]   mean_used: min={mean_used_cpu.min():.4e}, max={mean_used_cpu.max():.4e}, has_nan={np.isnan(mean_used_cpu).any()}", flush=True)
                print(f"[BN-DEBUG]   var_used: min={var_used_cpu.min():.4e}, max={var_used_cpu.max():.4e}, has_nan={np.isnan(var_used_cpu).any()}", flush=True)

                # Check if variance is too small (will cause issues with 1/sqrt(var))
                num_small_var = np.sum(var_used_cpu < 1e-8)
                if num_small_var > 0:
                    print(f"[BN-DEBUG]   WARNING: {num_small_var} channels have var < 1e-8", flush=True)

            x_norm = (x_in - mean_broadcast) / \
                jnp.sqrt(var_broadcast + self.eps)

            if self.use_affine:
                x_norm = x_norm * gamma + beta

            # chỉ những sample có t ∈ special_t[k] mới bị thay đổi
            x_out = jnp.where(mask_b > 0, x_norm, x_out)

        # logging: độ lệch norm
        diff_all = x_out - x_in
        norm_diff = jnp.mean(diff_all ** 2)

        mask_any = jnp.any(is_special, axis=1).astype(x.dtype)   # (B,)
        mask_any_b = mask_any.reshape(B, 1, 1, 1)
        diff_masked = diff_all * mask_any_b
        masked_norm_diff = jnp.mean(diff_masked ** 2)

        norm_percentage = jnp.mean(mask_any)

        # Diagnostic: Check output for NaN/Inf
        if jax.process_index() == 0:
            import numpy as np
            x_out_cpu = np.array(x_out)
            print(f"[BN-DEBUG] Output check: has_nan={np.isnan(x_out_cpu).any()}, has_inf={np.isinf(x_out_cpu).any()}", flush=True)
            print(f"[BN-DEBUG] Output stats: min={x_out_cpu.min():.4e}, max={x_out_cpu.max():.4e}, mean={x_out_cpu.mean():.4e}, std={x_out_cpu.std():.4e}", flush=True)

            if np.isnan(x_out_cpu).any() or np.isinf(x_out_cpu).any():
                print(f"[BN-DEBUG] WARNING: NaN/Inf in BatchNorm output!", flush=True)
                # Find which samples have NaN
                has_nan_per_sample = np.isnan(x_out_cpu).any(axis=(1,2,3))
                print(f"[BN-DEBUG]   Samples with NaN: {np.where(has_nan_per_sample)[0]}", flush=True)

        return x_out, masked_norm_diff, norm_diff, norm_percentage
