from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr

from .act import GeLU
from .attention import SelfAttention
from .ff import Mlp, SwiGLUFFN
from .misc import (
    LayerScale,
    make_ffn,
    make_norm_layer,
)
from .types import (
    ActLayer,
    AttentionClass,
    FFNLayer,
    NormLayer,
)


class SelfAttentionBlock(eqx.Module):
    norm1: nn.LayerNorm
    attn: SelfAttention
    ls1: LayerScale | nn.Identity

    norm2: nn.LayerNorm
    mlp: Mlp | SwiGLUFFN
    ls2: LayerScale | nn.Identity

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_value: float | None = None,
        act_layer: ActLayer = "gelu",
        norm_layer: NormLayer = "layernorm",
        attn_class: AttentionClass = "self_attention",
        ffn_layer: FFNLayer = "mlp",
        mask_k_bias: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        assert attn_class == "self_attention"

        attn_key, mlp_key = jr.split(key)

        self.norm1 = make_norm_layer(norm_layer, dim)

        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            mask_k_bias=mask_k_bias,
            key=attn_key,
        )

        self.ls1 = LayerScale(dim, init_value=init_value) if init_value else nn.Identity()

        self.norm2 = make_norm_layer(norm_layer, dim)

        mlp_hidden_dim = int(dim * ffn_ratio)

        self.mlp = make_ffn(
            ffn_layer,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
            key=mlp_key,
        )
        self.ls2 = LayerScale(dim, init_value=init_value) if init_value else nn.Identity()

    def __call__(
        self, x: Float[Array, "n_seq d"], rope: Float[Array, "2 hw d_head"] | None
    ) -> Float[Array, "n_seq d"]:
        res = x

        x = jax.vmap(self.norm1)(x)
        x = self.attn(x, rope=rope)
        x = self.ls1(x)

        x += res

        res = x

        x = jax.vmap(self.norm2)(x)
        x = self.mlp(x)
        x = self.ls2(x)

        x += res

        return x


class ConvNextBlock(eqx.Module):
    dwconv: nn.Conv2d
    norm: nn.LayerNorm
    pwconv1: nn.Linear
    act: GeLU
    pwconv2: nn.Linear
    gamma: Array | None

    def __init__(self, dim: int, *, layer_scale_init_value: float = 1e-6, key: PRNGKeyArray):
        super().__init__()

        dw_key, pw1_key, pw2_key, gamma_key = jr.split(key, 4)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, key=dw_key)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim, key=pw1_key)
        self.act = GeLU()
        self.pwconv2 = nn.Linear(4 * dim, dim, key=pw2_key)

        if layer_scale_init_value > 0:
            self.gamma = jnp.full(dim, layer_scale_init_value)
        else:
            self.gamma = None

    def __call__(
        self, x: Float[Array, "c h w"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "c h w"]:
        def c_vmap(f):
            # apply over c, vmap over h and w
            return jax.vmap(jax.vmap(f, in_axes=1, out_axes=1), in_axes=2, out_axes=2)

        res = x

        x = self.dwconv(x)
        x = c_vmap(self.norm)(x)
        x = c_vmap(self.pwconv1)(x)
        x = self.act(x)
        x = c_vmap(self.pwconv2)(x)

        if self.gamma is not None:
            x *= self.gamma[:, None, None]

        x += res

        return x
