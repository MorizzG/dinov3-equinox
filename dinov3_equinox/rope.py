from jaxtyping import Array, Float
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import (
    assert_equal_shape_prefix,
    assert_equal_shape_suffix,
    assert_rank,
    assert_shape,
)


def rope_rotate_half(
    x: Float[Array, "n_seq n_head d_head"],
) -> Float[Array, "n_seq n_head d_head"]:
    x_shape = x.shape

    assert x_shape[-1] % 2 == 0

    # x1, x2 = jnp.moveaxis(x.reshape(x_shape[:-1] + (x_shape[-1] // 2,) + (2,)), -1, 0)
    x1, x2 = jnp.moveaxis(x.reshape(x_shape[:-1] + (2, x_shape[-1] // 2)), -2, 0)

    x = jnp.concat([-x2, x1], axis=-1)

    assert_shape(x, x_shape)

    return x


def rope_apply(
    x: Float[Array, "hw n_head d_head"],
    sin: Float[Array, "hw d_head"],
    cos: Float[Array, "hw d_head"],
) -> Float[Array, "n_seq n_head d_head"]:
    assert_rank(x, 3)
    assert_rank([sin, cos], 2)

    assert_equal_shape_prefix([x, sin, cos], 1)
    assert_equal_shape_suffix([x, sin, cos], 1)

    return x * cos[:, None, :] + rope_rotate_half(x) * sin[:, None, :]


class RopePositionEmbedding(eqx.Module):
    base: float | None = eqx.field(static=True)
    min_period: float | None = eqx.field(static=True)
    max_period: float | None = eqx.field(static=True)

    d_head: int = eqx.field(static=True)

    normalize_coords: Literal["min", "max", "separate"] = eqx.field(static=True)
    shift_coords: float | None = eqx.field(static=True)
    jitter_coords: float | None = eqx.field(static=True)
    rescale_coords: float | None = eqx.field(static=True)

    dtype: jnp.dtype | None = eqx.field(static=True)

    periods: Array

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: jnp.dtype | None = None,
    ):
        super().__init__()

        assert embed_dim % (4 * num_heads) == 0

        both_periods = min_period is not None and max_period is not None

        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        d_head = embed_dim // num_heads

        self.base = base
        self.min_period = min_period
        self.max_period = max_period

        self.d_head = d_head

        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype

        # self.periods = jnp.empty((d_head // 4), dtype=dtype)

        if base is not None:
            self.periods = base ** (2 * jnp.arange(d_head // 4, dtype=dtype) / (d_head // 2))
        else:
            assert max_period is not None and min_period is not None

            base = max_period / min_period

            exponents = jnp.linspace(0, 1, d_head // 4, dtype=dtype)

            periods = base**exponents  # [1, max_period / min_period]

            periods /= base  # [min_period / max_period, 1]

            periods *= self.max_period  # [min_period, max_period]

            self.periods = periods

        assert_shape(self.periods, (d_head // 4,))
        if dtype is not None:
            assert self.periods.dtype == dtype

    def __call__(self, *, H: int, W: int) -> Float[Array, "2 hw d_head"]:
        match self.normalize_coords:
            case "max":
                max_HW = max(H, W)
                coords_h = jnp.arange(0.5, H, dtype=self.dtype) / max_HW
                coords_w = jnp.arange(0.5, W, dtype=self.dtype) / max_HW
            case "min":
                min_HW = min(H, W)
                coords_h = jnp.arange(0.5, H, dtype=self.dtype) / min_HW
                coords_w = jnp.arange(0.5, W, dtype=self.dtype) / min_HW
            case "separate":
                coords_h = jnp.arange(0.5, H, dtype=self.dtype) / H
                coords_w = jnp.arange(0.5, W, dtype=self.dtype) / W
            case _:
                assert False, f"invalid normalize_coords {self.normalize_coords}"

        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)

        assert_shape(coords, (H, W, 2))

        coords = coords.reshape(-1, 2)

        coords = 2.0 * coords - 1.0  # [0, 1]  ->  [-1, 1]

        periods = jax.lax.stop_gradient(self.periods)[None, None, :]

        angles = 2 * jnp.pi * coords[:, :, None] / periods  # [HW, 2, D//4]

        assert_shape(angles, (H * W, 2, self.d_head // 4))

        angles = angles.reshape(H * W, self.d_head // 2)

        angles = jnp.tile(angles, 2)

        assert_shape(angles, (H * W, self.d_head))

        cos = jnp.cos(angles)
        sin = jnp.sin(angles)

        rope = jnp.stack([sin, cos], axis=0)

        return rope
