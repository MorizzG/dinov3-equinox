# DINOv3 in equinox

[DINOv3](https://github.com/facebookresearch/dinov3) implementation in JAX, using the [equinox](https://github.com/patrick-kidger/equinox) library

## Usage

1. Download the DINOv3 weights from [https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). This repository only distributes the model implementation, not the weights.
2. Convert the weights from pth to safetensor using the module: `python -m dinov3_equinox <args>`
3. Create a model, either manually or using the provided `dinov3_*` functions, which create the same model as their corresponding `torch.hub.dinov3_*` function in the original repo.
4. Load the weights using the `load_weights` method, giving the path of the safetensor file as an argument. Note that this will not replace the weights of the model (this is impossible as equinox models are read-only), but rather return a new model with the weights substituted in.

## Limitations

Currently only inference is supported, because the dropouts are missing. This is because DropPath operates on batches, but batches are usually vmapped in JAX (they are here), meaning to support it the vmap would have to be pulled inside the model. Will possibly do it at some point.

Also, if the size of the model to be loaded is larger than half the available VRAM, loading the weights might (will) return an out-of-memory error, as both the old and the new weights will exist at the same time. This can be avoided by putting the model creation and weight loading in a `jax.default_device(jax.devices("cpu")[0])` context and afterwards moving the weights to the GPU/TPU using `model = jax.tree.map(lambda x: jax.device_put(x, device) if eqx.is_array(x) else x, model)`, where `device` is the device the model should be placed on.
