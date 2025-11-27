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


class ConditionalLayerNorm2dNHWC(nn.Module):
    num_channels: int
    special_t: Sequence[float]      # ví dụ [0.0, 0.25, 0.5, 0.75, 1.0]
    eps: float = 1e-5
    use_affine: bool = True

    @nn.compact
    def __call__(self, x, t):
        B, H, W, C = x.shape

        # 2. QUAN TRỌNG: Sửa axis từ (1, 2) thành (1, 2, 3)
        # Để tính Mean/Var trên toàn bộ bức ảnh (H, W) VÀ cả kênh màu (C)
        mean = jnp.mean(x, axis=(1, 2, 3), keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)

        # 2) Affine phụ thuộc t sử dụng nn.Embed
        # Thay vì dùng MLP, ta map t sang index của bảng embedding
        x_tilde = x_norm

        # Chuyển special_t sang jnp array để so sánh
        special_t_arr = jnp.asarray(self.special_t, dtype=t.dtype)  # [K]
        K = len(self.special_t)

        # Tìm index k tương ứng với mỗi t trong batch
        # Tính khoảng cách giữa t (batch) và special_t (list)
        # diff: [B, K]
        diff = jnp.abs(t[:, None] - special_t_arr[None, :])

        # Tạo mask những t nào khớp với special_t (sai số nhỏ)
        is_match = diff < 1e-5  # [B, K] (Bool)

        # Lấy index của timestep trùng khớp.
        # Lưu ý: Nếu t không nằm trong special_t, argmax sẽ trả về 0 (hoặc index đầu tiên),
        # nhưng điều này không sao vì ở bước (3) ta có mask chặn lại rồi.
        indices = jnp.argmax(is_match, axis=-1)  # [B]

        if self.use_affine:
            # Tạo Embedding Table giống như ví dụ ConditionalBatchNormSpecialT
            gamma_embed = nn.Embed(
                num_embeddings=K,
                features=C,
                embedding_init=nn.initializers.ones,
                name='gamma_embed'
            )
            beta_embed = nn.Embed(
                num_embeddings=K,
                features=C,
                embedding_init=nn.initializers.zeros,
                name='beta_embed'
            )

            # Lookup gamma, beta theo indices [B] -> [B, C]
            gamma_val = gamma_embed(indices)
            beta_val = beta_embed(indices)

            # Reshape để broadcast với x [B, H, W, C]
            gamma = gamma_val.reshape(B, 1, 1, C)
            beta = beta_val.reshape(B, 1, 1, C)

            # Áp dụng affine
            x_tilde = x_norm * gamma + beta

        # 3) Gating: Chỉ áp dụng Norm nếu t thực sự thuộc special_t
        # mask ở đây dùng để quyết định output cuối cùng
        # [B] -> True nếu t khớp bất kỳ giá trị nào trong special_t
        mask = jnp.any(is_match, axis=-1)
        mask_expanded = mask[:, None, None, None]  # [B, 1, 1, 1]

        # 4) Compute metrics (Giữ nguyên logic cũ của bạn)
        sq_err = (x - x_tilde) ** 2                          # [B,H,W,C]
        mse_per_sample = sq_err.mean(axis=(1, 2, 3))         # [B]

        mask_f = mask_expanded.astype(jnp.float32)           # [B,1,1,1]
        denom = jnp.maximum(mask_f.sum(), 1.0)

        masked_avg_mse = (mse_per_sample * mask_f.reshape(-1)).sum() / denom
        avg_mse = jnp.mean(mse_per_sample)
        norm_percentage = jnp.mean(mask_f)

        # 5) Output cuối cùng
        # Nếu t thuộc special_t -> dùng x_tilde (đã norm + affine embedding)
        # Nếu t KHÔNG thuộc special_t -> giữ nguyên x gốc
        y = jnp.where(mask_expanded, x_tilde, x)

        return y, masked_avg_mse, avg_mse, norm_percentage


# class ConditionalInstanceNorm2dNHWC(nn.Module):
#     num_channels: int
#     special_t: Sequence[float]      # ví dụ [0.0, 0.25, 0.5, 0.75, 1.0]
#     eps: float = 1e-5
#     use_affine: bool = True

#     @nn.compact
#     def __call__(self, x, t):
#         # x: [B,H,W,C], t: [B] (giống DiT.__call__)
#         # 1) Instance norm cơ bản
#         mean = jnp.mean(x, axis=(1, 2), keepdims=True)
#         var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
#         x_norm = (x - mean) / jnp.sqrt(var + self.eps)

#         # 2) Affine learnable (gamma, beta)
#         if self.use_affine:
#             gamma = self.param(
#                 "gamma",
#                 nn.initializers.ones,
#                 (1, 1, 1, self.num_channels),
#             )
#             beta = self.param(
#                 "beta",
#                 nn.initializers.zeros,
#                 (1, 1, 1, self.num_channels),
#             )
#             x_norm = x_norm * gamma + beta

#         # 3) Gating theo t đặc biệt
#         special_t = jnp.asarray(self.special_t, dtype=t.dtype)       # [K]
#         # diff = jnp.abs(t[:, None] - special_t[None, :])              # [B,K]
#         # mask = (diff < 1e-6).any(axis=-1)                            # [B]
#         mask = (t[:, None] == special_t[None, :]).any(axis=-1)
#         # [B,1,1,1]
#         mask = mask[:, None, None, None]

#         # 4) Compute masked MSE difference
#         sq_err = (x - x_norm) ** 2                            # [B,H,W,C]
#         mse_per_sample = sq_err.mean(axis=(1, 2, 3))            # [B]
#         mask_f = mask.astype(jnp.float32)              # [B]
#         denom = jnp.maximum(mask_f.sum(), 1.0)         # tránh chia 0

#         masked_avg_mse = (mse_per_sample * mask_f).sum() / denom
#         avg_mse = jnp.mean(mse_per_sample)
#         norm_percentage = jnp.mean(mask_f)

#         # Chỉ norm nếu t thuộc special_t, còn lại giữ nguyên x
#         return jnp.where(mask, x_norm, x), masked_avg_mse, avg_mse, norm_percentage
