from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp

from dinov3_equinox.act import GeLU
from dinov3_equinox.ff import Mlp, SwiGLUFFN
from dinov3_equinox.types import (
    ActLayer,
    FFNLayer,
    NormLayer,
)

STR_TO_DTYPE = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}


class LayerScale(eqx.Module):
    gamma: Array

    def __init__(self, dim: int, init_value: float):
        super().__init__()

        self.gamma = jnp.full((dim,), init_value)

    def __call__(self, x: Float[Array, "n_seq d"]) -> Float[Array, "n_seq d"]:
        return x * self.gamma


def make_act_layer(act: ActLayer) -> GeLU:
    match act:
        case "gelu":
            return GeLU()
        case _:
            raise ValueError(f"Invalid act {act}")


def make_norm_layer(norm: NormLayer, shape: int | tuple[int, ...]) -> nn.LayerNorm:
    match norm:
        case "layernorm":
            return nn.LayerNorm(shape, eps=1e-6)
        case "layernormbf16":
            return nn.LayerNorm(shape, eps=1e-5)


def make_ffn(
    ffn: FFNLayer,
    in_features: int,
    hidden_features: int,
    act_layer: ActLayer,
    bias: bool,
    *,
    key: PRNGKeyArray,
) -> Mlp | SwiGLUFFN:
    match ffn:
        case "mlp":
            return Mlp(
                in_features,
                hidden_features,
                out_features=None,
                act_layer=act_layer,
                bias=bias,
                key=key,
            )
        case "swiglu":
            return SwiGLUFFN(
                in_features,
                hidden_features,
                out_features=None,
                act_layer=act_layer,
                bias=bias,
                key=key,
            )
        case "swiglu32":
            return SwiGLUFFN(
                in_features,
                hidden_features,
                out_features=None,
                act_layer=act_layer,
                bias=bias,
                align_to=32,
                key=key,
            )
        case "swiglu64":
            return SwiGLUFFN(
                in_features,
                hidden_features,
                out_features=None,
                act_layer=act_layer,
                bias=bias,
                align_to=64,
                key=key,
            )
        case "swiglu128":
            return SwiGLUFFN(
                in_features,
                hidden_features,
                out_features=None,
                act_layer=act_layer,
                bias=bias,
                align_to=128,
                key=key,
            )
        case _:
            raise ValueError(f"invalid ffn {ffn}")
