import jax
import jax.numpy as jnp
import jax.scipy.fft as fft


def dct_2d(x):
    """Tính DCT trên 2 chiều không gian (H, W)."""
    return fft.dct(fft.dct(x, axis=-2, norm='ortho'), axis=-3, norm='ortho')


def idct_2d(x):
    """Tính Inverse DCT."""
    return fft.idct(fft.idct(x, axis=-3, norm='ortho'), axis=-2, norm='ortho')


def dct_reduce(latents, keep_size=8):
    """
    Nén latent từ [B, 32, 32, 4] -> [B, 256] (nếu keep_size=8).
    Giữ lại tần số thấp (góc trên-trái).
    """
    # 1. Chuyển sang miền tần số
    freqs = dct_2d(latents)

    # 2. Cắt lấy góc phần tư chứa thông tin cấu trúc
    freqs_low = freqs[:, :keep_size, :keep_size, :]

    # 3. Flatten
    return freqs_low.reshape(freqs_low.shape[0], -1)


def idct_power_law(key, flat_low, original_h=32, original_w=32, keep_size=8, alpha=1.0, noise_scale=1.0):
    """
    Khôi phục ảnh từ vector nén.
    Vùng tần số cao (bị cắt bỏ) được điền bằng NHIỄU HỒNG (Pink Noise/Power Law).
    """
    B = flat_low.shape[0]
    C = 4  # Latent channels

    # 1. Chuẩn bị lưới tần số để tính suy giảm năng lượng 1/f^alpha
    u = jnp.arange(original_h)
    v = jnp.arange(original_w)
    uu, vv = jnp.meshgrid(u, v, indexing='ij')
    freq_dist = jnp.sqrt(uu**2 + vv**2)
    freq_dist = jnp.maximum(freq_dist, 1.0)  # Tránh chia 0

    # Mask tỉ lệ nghịch với tần số (Power Law)
    scale_mask = 1.0 / (freq_dist ** alpha)
    scale_mask = scale_mask[None, :, :, None]  # [1, H, W, 1]

    # 2. Sinh nhiễu nền (High frequency texture)
    white_noise = jax.random.normal(key, (B, original_h, original_w, C))
    colored_noise = white_noise * scale_mask * noise_scale

    # 3. Điền thông tin cấu trúc (Low freq) vào nền nhiễu
    freqs_low = flat_low.reshape(B, keep_size, keep_size, C)
    freqs_full = colored_noise.at[:, :keep_size, :keep_size, :].set(freqs_low)

    # 4. Inverse DCT về không gian pixel
    return idct_2d(freqs_full)
