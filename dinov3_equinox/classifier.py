from jaxtyping import Array, Float

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from chex import assert_axis_dimension, assert_rank, assert_shape

from dinov3_equinox.vit import DinoVisionTransformer


class DinoClassifier(eqx.Module):
    backbone: DinoVisionTransformer

    head: nn.Linear

    def __init__(self, *, backbone: DinoVisionTransformer, head: nn.Linear):
        super().__init__()

        assert head.in_features == 2 * backbone.embed_dim, (
            f"head doesn't fit backbone; backbone has embed_dim {backbone.embed_dim}, "
            f"but head has in_features {head.in_features} instead of 2 * embed_dim"
        )

        self.backbone = backbone
        self.head = head

    def __call__(self, img: Float[Array, "3 h w"]) -> Float[Array, " n"]:
        features = self.backbone(img)

        cls_token = features["cls_token"]
        patch_tokens = features["patch_tokens"]

        embed_dim = self.backbone.embed_dim

        assert_shape(cls_token, (1, embed_dim))
        assert_rank(patch_tokens, 2)
        assert_axis_dimension(patch_tokens, 1, embed_dim)

        x = jnp.concat([cls_token[0], patch_tokens.mean(axis=0)])

        assert_shape(x, (2 * embed_dim,))

        x = self.head(x)

        return x
