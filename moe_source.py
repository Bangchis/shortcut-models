import flax.linen as nn
import jax.numpy as jnp


class ConditionProjector(nn.Module):
    num_modes: int
    hidden_channels: int

    @nn.compact
    def __call__(self, condition):
        x = nn.Dense(self.hidden_channels)(condition)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_channels)(x)
        return x


class SharedTrunk(nn.Module):
    hidden_channels: int

    @nn.compact
    def __call__(self, z):
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME')(z)
        x = nn.silu(x)
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME')(x)
        x = nn.silu(x)
        return x


class Router(nn.Module):
    hidden_channels: int
    num_modes: int
    tau: float

    @nn.compact
    def __call__(self, features):
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME')(features)
        x = nn.silu(x)
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_modes)(x)
        alpha = nn.softmax(logits / self.tau, axis=-1)
        return alpha, logits


class SourceExpert(nn.Module):
    hidden_channels: int
    out_channels: int
    zero_init: bool

    @nn.compact
    def __call__(self, features):
        x = nn.Conv(self.hidden_channels, (3, 3), padding='SAME')(features)
        x = nn.silu(x)
        kernel_init = nn.initializers.zeros if self.zero_init else nn.initializers.lecun_normal()
        bias_init = nn.initializers.zeros
        x = nn.Conv(
            self.out_channels,
            (3, 3),
            padding='SAME',
            kernel_init=kernel_init,
            bias_init=bias_init,
        )(x)
        return x


class SourceMoE(nn.Module):
    num_modes: int
    hidden_channels: int
    out_channels: int
    tau: float
    zero_init: bool = True

    @nn.compact
    def __call__(self, z, condition, return_experts=False):
        condition_bias = ConditionProjector(
            num_modes=self.num_modes,
            hidden_channels=self.hidden_channels,
            name='condition_projector',
        )(condition)
        condition_bias = condition_bias[:, None, None, :]

        features = SharedTrunk(
            hidden_channels=self.hidden_channels,
            name='shared_trunk',
        )(z)
        conditioned_features = features + condition_bias

        alpha, logits = Router(
            hidden_channels=self.hidden_channels,
            num_modes=self.num_modes,
            tau=self.tau,
            name='router',
        )(conditioned_features)

        expert_outputs = []
        for idx in range(self.num_modes):
            expert_outputs.append(
                SourceExpert(
                    hidden_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    zero_init=self.zero_init,
                    name=f'expert_{idx}',
                )(conditioned_features)
            )
        expert_outputs = jnp.stack(expert_outputs, axis=1)
        delta = jnp.sum(
            expert_outputs * alpha[:, :, None, None, None],
            axis=1,
        )
        x0 = z + delta

        if return_experts:
            return x0, alpha, expert_outputs, logits
        return x0, alpha
