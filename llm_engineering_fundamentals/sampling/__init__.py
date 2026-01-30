"""Sampling strategies for text generation."""

from llm_engineering_fundamentals.sampling.methods import (
    sample_with_temperature,
    top_k_sampling,
    top_p_sampling,
    sample_next_token,
)

__all__ = [
    "sample_with_temperature",
    "top_k_sampling",
    "top_p_sampling",
    "sample_next_token",
]

