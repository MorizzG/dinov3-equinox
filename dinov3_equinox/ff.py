from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr

from dinov3_equinox.act import GeLU
from dinov3_equinox.types import (
    ActLayer,
)


class Mlp(eqx.Module):
    fc1: nn.Linear
    act: GeLU
    fc2: nn.Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        *,
        act_layer: ActLayer = "gelu",
        drop: float = 0.0,
        bias: bool = True,
        key: PRNGKeyArray,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        fc1_key, fc2_key = jr.split(key)

        match act_layer:
            case "gelu":
                self.act = GeLU()

        self.fc1 = nn.Linear(in_features, hidden_features, use_bias=bias, key=fc1_key)

        self.fc2 = nn.Linear(hidden_features, out_features, use_bias=bias, key=fc1_key)

    def __call__(self, x: Float[Array, "n_seq d"]) -> Float[Array, "n_seq d"]:
        x = jax.vmap(self.fc1)(x)
        x = self.act(x)
        x = jax.vmap(self.fc2)(x)

        return x


class SwiGLUFFN(eqx.Module):
    w1: nn.Linear
    w2: nn.Linear
    w3: nn.Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        *,
        act_layer: ActLayer = "gelu",
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        key: PRNGKeyArray,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        d = int(hidden_features * 2 / 3)

        swiglu_hidden_features = d + (-d % align_to)

        key1, key2, key3 = jr.split(key, 3)

        self.w1 = nn.Linear(in_features, swiglu_hidden_features, use_bias=bias, key=key1)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, use_bias=bias, key=key2)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, use_bias=bias, key=key3)

    def __call__(self, x: Float[Array, "n_seq d"]) -> Float[Array, "n_seq d"]:
        gate = jax.vmap(self.w1)(x)
        gate = jax.nn.silu(gate)

        x = jax.vmap(self.w2)(x)

        x *= gate

        x = jax.vmap(self.w3)(x)

        return x
