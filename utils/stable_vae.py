
from functools import partial, cached_property

import jax
from diffusers import FlaxAutoencoderKL
from einops import rearrange
from flax import struct

from jaxtyping import Array, PyTree, Key, Float, Shaped, Int, UInt8, jaxtyped
from typeguard import typechecked
from functools import partial
typecheck = partial(jaxtyped, typechecker=typechecked)

@struct.dataclass
class StableVAE:
    params: PyTree[Float[Array, "..."]]
    module: FlaxAutoencoderKL = struct.field(pytree_node=False)

    @classmethod
    def create(cls) -> "VAE":
        # module, params = FlaxAutoencoderKL.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae"
        # )
        module, params = FlaxAutoencoderKL.from_pretrained(
            "pcuenq/sd-vae-ft-mse-flax"
        )
        params = jax.device_get(params)
        return cls(
            params=params,
            module=module,
        )

    @partial(jax.jit, static_argnames=("scale", "epsilon_scale"))
    def encode(
        self, key: Key[Array, ""], images: Float[Array, "b h w 3"], scale: bool = True, epsilon_scale: float = 1.0
    ) -> Float[Array, "b lh lw 4"]:
        images = rearrange(images, "b h w c -> b c h w")
        latent_dist = self.module.apply(
            {"params": self.params}, images, method=self.module.encode
        ).latent_dist

        # Apply epsilon scaling
        if epsilon_scale == 0.0:
            latents = latent_dist.mean  # Deterministic mode
        else:
            eps = jax.random.normal(key, latent_dist.mean.shape)
            latents = latent_dist.mean + epsilon_scale * latent_dist.std * eps

        if scale:
            latents *= self.module.config.scaling_factor
        return latents

    @partial(jax.jit, static_argnames="scale")
    def encode_posterior(
        self, images: Float[Array, "b h w 3"], scale: bool = True
    ) -> tuple[Float[Array, "b lh lw 4"], Float[Array, "b lh lw 4"]]:
        """
        Get VAE posterior parameters (μ, logσ²) without sampling.

        This is used for caching posterior parameters to enable stochastic
        sampling during training without running VAE encoder each iteration.

        Args:
            images: Input images (b, h, w, 3)
            scale: Whether to apply VAE scaling_factor

        Returns:
            mu: Mean of posterior (b, lh, lw, 4)
            logvar: Log variance of posterior (b, lh, lw, 4)
        """
        images = rearrange(images, "b h w c -> b c h w")
        latent_dist = self.module.apply(
            {"params": self.params}, images, method=self.module.encode
        ).latent_dist

        mu = latent_dist.mean
        std = latent_dist.std
        logvar = 2.0 * jax.numpy.log(std)  # logvar = log(σ²) = 2*log(σ)

        if scale:
            # Scale mean
            mu *= self.module.config.scaling_factor
            # Scale variance: var' = var * scale²
            # logvar' = log(var * scale²) = logvar + 2*log(scale)
            logvar += 2.0 * jax.numpy.log(self.module.config.scaling_factor)

        return mu, logvar

    @partial(jax.jit, static_argnames="scale")
    def decode(
        self, latents: Float[Array, "b lh lw 4"], scale: bool = True
    ) -> Float[Array, "b h w 3"]:
        if scale:
            latents /= self.module.config.scaling_factor
        images = self.module.apply(
            {"params": self.params}, latents, method=self.module.decode
        ).sample
        # convert to channels-last
        images = rearrange(images, "b c h w -> b h w c")
        return images

    @cached_property
    def downscale_factor(self) -> int:
        return 2 ** (len(self.module.block_out_channels) - 1)