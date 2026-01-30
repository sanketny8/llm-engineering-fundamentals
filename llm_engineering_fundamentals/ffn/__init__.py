"""Feed-Forward Networks and Activation Functions."""

from llm_engineering_fundamentals.ffn.activations import (
    relu,
    gelu,
    gelu_approx,
    swish,
    silu,
    swiglu,
    geglu,
)

from llm_engineering_fundamentals.ffn.networks import (
    FeedForwardNetwork,
    GatedFFN,
    SwiGLUFFN,
    GeGLUFFN,
)

__all__ = [
    # Activation functions
    "relu",
    "gelu",
    "gelu_approx",
    "swish",
    "silu",
    "swiglu",
    "geglu",
    # FFN architectures
    "FeedForwardNetwork",
    "GatedFFN",
    "SwiGLUFFN",
    "GeGLUFFN",
]

