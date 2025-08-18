from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import assert_axis_dimension, assert_rank, assert_shape

from dinov3_equinox.layers.block import ConvNextBlock
from dinov3_equinox.layers.norm import ConvLayerNorm


class DinoConvNeXt(eqx.Module):
    embed_dim: int = eqx.field(static=True)

    downsample_layers: list[nn.Sequential]
    stages: list[nn.Sequential]

    norm: nn.LayerNorm

    def __init__(
        self,
        *,
        in_chans: int = 3,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        layer_scale_init_value: float = 1e-6,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.embed_dim = dims[-1]

        def consume():
            nonlocal key

            key, consume = jr.split(key)

            return consume

        self.downsample_layers = []

        stem = nn.Sequential(
            [
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, key=consume()),
                ConvLayerNorm(dims[0], eps=1e-6),
            ]
        )

        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                [
                    ConvLayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, key=consume()),
                ]
            )

            self.downsample_layers.append(downsample_layer)

        self.stages = []

        for i in range(4):
            stage = nn.Sequential(
                [
                    ConvNextBlock(
                        dim=dims[i], layer_scale_init_value=layer_scale_init_value, key=consume()
                    )
                    for _ in range(depths[i])
                ]
            )

            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def __call__(self, img: Float[Array, "c h w"]) -> dict[str, Array]:
        _, h, w = img.shape

        x = img

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x_pool = x.mean(axis=(1, 2))

        assert_shape(x_pool, (self.embed_dim,))

        x = x.reshape(x.shape[0], -1).T

        assert_rank(x, 2)
        assert_axis_dimension(x, 1, self.embed_dim)

        x_norm = jnp.concat([x_pool[None, :], x], axis=0)

        x_norm = jax.vmap(self.norm)(x)

        return {
            "cls_token": x_norm[0, :],
            "storage_tokens": jnp.zeros((0, self.embed_dim)),
            "patch_tokens": x_norm[1:, :],
        }
