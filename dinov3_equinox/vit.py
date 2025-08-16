from jaxtyping import Array, Float, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import (
    assert_axis_dimension,
    assert_rank,
)

from dinov3_equinox.attention import SelfAttentionBlock
from dinov3_equinox.misc import STR_TO_DTYPE, make_norm_layer
from dinov3_equinox.patch_embed import PatchEmbed
from dinov3_equinox.rope import RopePositionEmbedding
from dinov3_equinox.types import (
    FFNLayer,
    NormLayer,
)


class DinoVisionTransformer(eqx.Module):
    embed_dim: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    n_storage_tokens: int = eqx.field(static=True)

    untie_cls_and_patch_norms: bool = eqx.field(static=True)
    untie_global_and_local_cls_norm: bool = eqx.field(static=True)

    patch_embed: PatchEmbed

    cls_token: Array
    storage_tokens: Array | None
    mask_token: Array

    rope_embed: RopePositionEmbedding

    blocks: list[SelfAttentionBlock]

    norm: nn.LayerNorm

    cls_norm: nn.LayerNorm | None
    local_cls_norm: nn.LayerNorm | None

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: NormLayer = "layernorm",
        ffn_layer: FFNLayer = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.n_storage_tokens = n_storage_tokens

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm

        def consume() -> PRNGKeyArray:
            nonlocal key

            key, consume = jr.split(key)

            return consume

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
            key=consume(),
        )

        self.cls_token = 0.02 * jr.normal(consume(), (1, embed_dim))

        if n_storage_tokens > 0:
            self.storage_tokens = 0.02 * jr.normal(consume(), (n_storage_tokens, embed_dim))
        else:
            self.storage_tokens = None

        self.mask_token = jnp.zeros((1, embed_dim))

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=STR_TO_DTYPE[pos_embed_rope_dtype],
        )

        self.blocks = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer="gelu",
                ffn_layer=ffn_layer,
                init_value=layerscale_init,
                mask_k_bias=mask_k_bias,
                key=consume(),
            )
            for i in range(depth)
        ]

        self.norm = make_norm_layer(norm_layer, embed_dim)

        if untie_cls_and_patch_norms:
            self.cls_norm = make_norm_layer(norm_layer, embed_dim)
        else:
            self.cls_norm = None

        if untie_global_and_local_cls_norm:
            self.local_cls_norm = make_norm_layer(norm_layer, embed_dim)
        else:
            self.local_cls_norm = None

    def __call__(
        self, img: Float[Array, "c h w"], *, return_hidden: bool = False
    ) -> dict[str, Array]:
        x = self.patch_embed(img)

        assert_rank(x, 3)
        assert_axis_dimension(x, 2, self.embed_dim)

        H, W = x.shape[:2]

        # flatten out H and W
        x = x.reshape(-1, self.embed_dim)

        cls_token = self.cls_token

        if self.storage_tokens is not None:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = jnp.empty((0, self.embed_dim))

        x = jnp.concat([cls_token, storage_tokens, x], axis=0)

        rope = self.rope_embed(H=H, W=W)

        hiddens = [x]

        for block in self.blocks:
            x = block(x, rope=rope)

            hiddens.append(x)

        hiddens = jnp.stack(hiddens, axis=0)

        if self.untie_cls_and_patch_norms:
            assert self.cls_norm is not None

            x_cls_local = jax.vmap(self.cls_norm)(x[: self.n_storage_tokens + 1, :])
            x_patch = jax.vmap(self.norm)(x[self.n_storage_tokens + 1 :, :])
        elif self.untie_global_and_local_cls_norm:
            x_cls_local = jax.vmap(self.norm)(x[: self.n_storage_tokens + 1, :])
            x_patch = jax.vmap(self.norm)(x[self.n_storage_tokens + 1 :, :])
        else:
            x = jax.vmap(self.norm)(x)

            x_cls_local = x[: self.n_storage_tokens + 1, :]
            x_patch = x[self.n_storage_tokens + 1 :, :]

        cls_token = x_cls_local[0:1, :]
        storage_tokens = x_cls_local[1:, :]

        out = {
            "cls_token": cls_token,
            "storage_tokens": storage_tokens,
            "patch_tokens": x_patch,
        }

        if return_hidden:
            out["hidden"] = hiddens

        return out
