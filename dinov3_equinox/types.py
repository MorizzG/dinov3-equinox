from typing import Literal

type ActLayer = Literal["gelu"]
type NormLayer = Literal["layernorm", "layernormbf16"]
type AttentionClass = Literal["self_attention"]
type FFNLayer = Literal["mlp", "swiglu", "swiglu32", "swiglu64", "swiglu128"]
