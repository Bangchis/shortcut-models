from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


def sample_diag_gaussian(key, mu, logvar):
    eps = jax.random.normal(key, mu.shape)
    return mu + eps * jnp.exp(0.5 * logvar)


def var_only_kld_loss(var, logvar, target_std=1.0, eps=1e-6):
    target_var = jnp.maximum(jnp.asarray(target_std, dtype=var.dtype) ** 2, eps)
    var_star = var / target_var
    logvar_star = logvar - jnp.log(target_var)
    return -0.5 * jnp.mean(1.0 + logvar_star - var_star)


class ConditionProjector(nn.Module):
    condition_dim: int
    hidden_channels: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, condition_embedding):
        condition_embedding = condition_embedding.astype(self.dtype)
        x = nn.Dense(self.hidden_channels, dtype=self.dtype)(condition_embedding)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_channels, dtype=self.dtype)(x)
        return x


class SharedTrunk(nn.Module):
    hidden_channels: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z):
        z = z.astype(self.dtype)
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME', dtype=self.dtype)(z)
        x = nn.silu(x)
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME', dtype=self.dtype)(x)
        x = nn.silu(x)
        return x


class Router(nn.Module):
    hidden_channels: int
    num_modes: int
    tau: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        features = features.astype(self.dtype)
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME', dtype=self.dtype)(features)
        x = nn.silu(x)
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_modes, dtype=self.dtype)(x).astype(jnp.float32)
        alpha = nn.softmax(logits / self.tau, axis=-1)
        return alpha, logits


class SourceExpert(nn.Module):
    hidden_channels: int
    out_channels: int
    zero_init: bool
    logvar_min: float
    logvar_max: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        features = features.astype(self.dtype)
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME', dtype=self.dtype)(features)
        x = nn.silu(x)
        if self.zero_init:
            kernel_init = nn.initializers.zeros
            mu_bias_init = nn.initializers.zeros
            logvar_bias_init = nn.initializers.zeros
        else:
            kernel_init = nn.initializers.lecun_normal()
            mu_bias_init = nn.initializers.zeros
            logvar_bias_init = nn.initializers.zeros
        mu = nn.Conv(
            self.out_channels,
            (3, 3),
            padding='SAME',
            kernel_init=kernel_init,
            bias_init=mu_bias_init,
            dtype=self.dtype,
            name='mu_head',
        )(x).astype(jnp.float32)
        logvar = nn.Conv(
            self.out_channels,
            (3, 3),
            padding='SAME',
            kernel_init=kernel_init,
            bias_init=logvar_bias_init,
            dtype=self.dtype,
            name='logvar_head',
        )(x).astype(jnp.float32)
        logvar = jnp.clip(logvar, self.logvar_min, self.logvar_max)
        return mu, logvar


class SourceMoE(nn.Module):
    num_modes: int
    condition_dim: int
    hidden_channels: int
    out_channels: int
    tau: float
    soft_moe: bool
    var_eps: float
    logvar_min: float
    logvar_max: float
    zero_init: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z, condition_weights, return_experts=False):
        mode_embeddings = self.param(
            'mode_embeddings',
            nn.initializers.normal(stddev=0.02),
            (self.num_modes, self.condition_dim),
        )
        condition_embedding = jnp.matmul(condition_weights, mode_embeddings)
        condition_bias = ConditionProjector(
            condition_dim=self.condition_dim,
            hidden_channels=self.hidden_channels,
            dtype=self.dtype,
            name='condition_projector',
        )(condition_embedding)
        condition_bias = condition_bias[:, None, None, :]

        features = SharedTrunk(
            hidden_channels=self.hidden_channels,
            dtype=self.dtype,
            name='shared_trunk',
        )(z)
        conditioned_features = features + condition_bias

        alpha, logits = Router(
            hidden_channels=self.hidden_channels,
            num_modes=self.num_modes,
            tau=self.tau,
            dtype=self.dtype,
            name='router',
        )(conditioned_features)

        expert_mu = []
        expert_logvar = []
        for idx in range(self.num_modes):
            mu_j, logvar_j = SourceExpert(
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                zero_init=self.zero_init,
                logvar_min=self.logvar_min,
                logvar_max=self.logvar_max,
                dtype=self.dtype,
                name=f'expert_{idx}',
            )(conditioned_features)
            expert_mu.append(mu_j)
            expert_logvar.append(logvar_j)

        expert_mu = jnp.stack(expert_mu, axis=1)
        expert_logvar = jnp.stack(expert_logvar, axis=1)
        expert_var = jnp.exp(expert_logvar)

        if self.soft_moe:
            alpha_mix = alpha
        else:
            alpha_mix = jax.nn.one_hot(
                jnp.argmax(alpha, axis=-1),
                self.num_modes,
                dtype=alpha.dtype,
            )

        alpha_full = alpha_mix[:, :, None, None, None]
        mu_x0 = jnp.sum(expert_mu * alpha_full, axis=1)
        second_moment = jnp.sum(alpha_full * (expert_var + expert_mu ** 2), axis=1)
        var_x0 = jnp.maximum(second_moment - mu_x0 ** 2, self.var_eps)
        logvar_x0 = jnp.log(var_x0 + self.var_eps)

        if return_experts:
            return (
                mu_x0,
                logvar_x0,
                var_x0,
                alpha,
                expert_mu,
                expert_logvar,
                logits,
            )
        return mu_x0, logvar_x0, var_x0, alpha
