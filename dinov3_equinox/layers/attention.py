from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import (
    assert_axis_dimension,
    assert_rank,
    assert_shape,
)

from .rope import rope_apply


class LinearKMaskedBias(eqx.Module):
    weight: Array
    bias: Array | None
    bias_mask: Array | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        key: PRNGKeyArray,
    ):
        super().__init__()

        weight_key, bias_key = jr.split(key)

        self.weight = jr.normal(weight_key, (out_features, in_features))

        if use_bias:
            self.bias = jr.normal(bias_key, (out_features,))
            self.bias_mask = jnp.full_like(self.bias, jnp.nan)
        else:
            self.bias = None
            self.bias_mask = None

    def __call__(self, x: Float[Array, " in_features"]) -> Float[Array, " out_features"]:
        x = self.weight @ x

        if self.bias is not None:
            x += self.bias * jax.lax.stop_gradient(self.bias_mask)

        return x


class SelfAttention(eqx.Module):
    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    qkv: nn.Linear | LinearKMaskedBias
    proj: nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        mask_k_bias: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        qkv_key, proj_key = jr.split(key)

        linear_cls = LinearKMaskedBias if mask_k_bias else nn.Linear

        self.qkv = linear_cls(dim, dim * 3, use_bias=qkv_bias, key=qkv_key)

        self.proj = nn.Linear(dim, dim, use_bias=proj_bias, key=proj_key)

    def apply_rope(
        self,
        q: Float[Array, "n_seq n_head d_head"],
        k: Float[Array, "n_seq n_head d_head"],
        rope: Float[Array, "2 hw d_head"],
    ) -> tuple[Array, Array]:
        q_dtype = q.dtype
        k_dtype = k.dtype

        sin, cos = rope

        q = q.astype(rope.dtype)
        k = k.astype(rope.dtype)

        n_seq = q.shape[0]

        prefix = n_seq - sin.shape[-2]

        q_prefix = q[:prefix, ...]
        q = rope_apply(q[prefix:, ...], sin, cos)
        q = jnp.concat([q_prefix, q], axis=0)

        k_prefix = k[:prefix, ...]
        k = rope_apply(k[prefix:, ...], sin, cos)
        k = jnp.concat([k_prefix, k], axis=0)

        q = q.astype(q_dtype)
        k = k.astype(k_dtype)

        return q, k

    def __call__(
        self, x: Float[Array, "n_seq d"], rope: Float[Array, "2 hw d_head"] | None
    ) -> Float[Array, "n_seq d"]:
        assert_rank(x, 2)
        assert_axis_dimension(x, 1, self.dim)

        n_seq = x.shape[0]

        qkv = jax.vmap(self.qkv)(x)

        qkv = qkv.reshape(n_seq, 3, self.num_heads, self.dim // self.num_heads)

        q, k, v = qkv.swapaxes(0, 1)

        assert_shape([q, k, v], (n_seq, self.num_heads, self.dim // self.num_heads))

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        x = jax.nn.dot_product_attention(q, k, v)

        x = x.reshape(n_seq, self.dim)

        x = jax.vmap(self.proj)(x)

        return x
