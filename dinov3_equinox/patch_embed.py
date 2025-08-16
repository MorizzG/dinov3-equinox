from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr
from chex import (
    assert_axis_dimension,
    assert_rank,
)


class PatchEmbed(eqx.Module):
    in_channels: int = eqx.field(static=True)

    patch_size: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)

    proj: nn.Conv2d

    def __init__(
        self,
        # image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, key=key
        )

        self._reset_parameters(key=key)

    def _reset_parameters(self, *, key: PRNGKeyArray):
        weight_key, bias_key = jr.split(key)

        k = 1 / (self.in_channels * self.patch_size**2)

        weight = jr.uniform(
            weight_key, self.proj.weight.shape, minval=-jnp.sqrt(k), maxval=jnp.sqrt(k)
        )

        assert self.proj.bias is not None
        bias = jr.uniform(bias_key, self.proj.bias.shape, minval=-jnp.sqrt(k), maxval=jnp.sqrt(k))

        self.proj = eqx.tree_at(lambda proj: (proj.weight, proj.bias), self.proj, (weight, bias))

    def __call__(self, img: Float[Array, "3 h w"]) -> Float[Array, "h_out w_out c"]:
        assert_rank(img, 3)
        assert_axis_dimension(img, 0, 3)

        assert (
            img.shape[1] % self.patch_size == 0 and img.shape[2] % self.patch_size == 0
        ), f"image shape must be divisible by patch size, but {img.shape} was given"

        x = self.proj(img)

        x = x.transpose(1, 2, 0)

        return x
