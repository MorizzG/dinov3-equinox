from jaxtyping import Array, Float, PRNGKeyArray
from typing import overload

import equinox.nn as nn
import jax
from chex import assert_rank
from equinox._custom_types import sentinel
from equinox.nn._stateful import State


class ConvLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype=None,
    ):
        super().__init__(dim, eps, use_weight, use_bias, dtype=dtype)

    @overload
    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array: ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: PRNGKeyArray | None = None
    ) -> tuple[Array, State]: ...

    def __call__(
        self, x: Float[Array, "c h w"], state: State = sentinel, *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "c h w"] | tuple[Float[Array, "c h w"], State]:
        assert_rank(x, 3)

        # [c h w] -> [h w c]
        x = x.transpose(1, 2, 0)

        x = jax.vmap(jax.vmap(super().__call__))(x)

        # [h w c] -> [c h w]
        x = x.transpose(2, 0, 1)

        if state is not sentinel:
            return x, state

        return x
