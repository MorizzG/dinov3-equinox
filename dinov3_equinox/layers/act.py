from jaxtyping import Array

import equinox as eqx
import jax


class GeLU(eqx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Array) -> Array:
        return jax.nn.gelu(x, approximate=False)
