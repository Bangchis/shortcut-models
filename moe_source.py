import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def sample_source_gaussian(key, mu, log_sigma):
    eps = jax.random.normal(key, mu.shape)
    return mu + eps * jnp.exp(log_sigma)


def source_sigma_floor_loss(log_sigma, sigma_min):
    log_sigma_min = jnp.log(jnp.asarray(sigma_min, dtype=log_sigma.dtype))
    return jnp.mean(jnp.maximum(0.0, log_sigma_min - log_sigma) ** 2)


def _valid_num_groups(channels, preferred):
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FiLMResBlock(nn.Module):
    channels: int
    cond_dim: int
    kernel_size: int = 3
    num_groups: int = 8

    @nn.compact
    def __call__(self, h, cond):
        residual = h
        groups = _valid_num_groups(self.channels, self.num_groups)

        x = nn.GroupNorm(num_groups=groups)(h)
        film = nn.Dense(
            self.channels * 2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='film_0',
        )(cond)
        scale, bias = jnp.split(film, 2, axis=-1)
        x = x * (1.0 + scale[:, None, None, :]) + bias[:, None, None, :]
        x = nn.silu(x)
        x = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            padding='SAME',
            name='conv_0',
        )(x)

        x = nn.GroupNorm(num_groups=groups)(x)
        film = nn.Dense(
            self.channels * 2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='film_1',
        )(cond)
        scale, bias = jnp.split(film, 2, axis=-1)
        x = x * (1.0 + scale[:, None, None, :]) + bias[:, None, None, :]
        x = nn.silu(x)
        x = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            padding='SAME',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='conv_1',
        )(x)
        return residual + x


class SourceBaseNet(nn.Module):
    num_modes: int
    angular_num_submodes: int
    condition_dim: int
    channels: int
    num_blocks: int
    out_channels: int
    sigma_init: float
    kernel_size: int = 3
    num_groups: int = 8

    @nn.compact
    def __call__(self, z, x_base, modes, angular_codes, log_radius):
        mode_embed = nn.Embed(
            num_embeddings=self.num_modes,
            features=self.condition_dim,
            name='mode_embed',
        )(modes)
        angular_embed = nn.Embed(
            num_embeddings=self.angular_num_submodes,
            features=self.condition_dim,
            name='angular_embed',
        )(angular_codes)
        cond = jnp.concatenate(
            [mode_embed, angular_embed, log_radius[:, None]], axis=-1)
        cond = nn.Dense(self.condition_dim * 4, name='cond_dense_0')(cond)
        cond = nn.silu(cond)
        cond = nn.Dense(self.condition_dim * 4, name='cond_dense_1')(cond)
        cond = nn.silu(cond)

        h = jnp.concatenate([z, x_base], axis=-1)
        h = nn.Conv(
            self.channels,
            (self.kernel_size, self.kernel_size),
            padding='SAME',
            name='input_conv',
        )(h)
        for idx in range(self.num_blocks):
            h = FiLMResBlock(
                channels=self.channels,
                cond_dim=self.condition_dim * 4,
                kernel_size=self.kernel_size,
                num_groups=self.num_groups,
                name=f'resblock_{idx}',
            )(h, cond)

        groups = _valid_num_groups(self.channels, self.num_groups)
        h = nn.GroupNorm(num_groups=groups, name='out_norm')(h)
        h = nn.silu(h)
        delta_mu = nn.Conv(
            self.out_channels,
            (3, 3),
            padding='SAME',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='delta_mu_head',
        )(h)
        log_sigma = nn.Conv(
            self.out_channels,
            (3, 3),
            padding='SAME',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(math.log(self.sigma_init)),
            name='log_sigma_head',
        )(h)
        mu_x0 = x_base + delta_mu
        return mu_x0, log_sigma, delta_mu
