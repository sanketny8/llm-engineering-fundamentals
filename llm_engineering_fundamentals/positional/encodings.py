"""
Positional Encoding Implementations

This module implements 4 different positional encoding strategies:
1. Sinusoidal (classic Transformer)
2. Learned (BERT-style)
3. RoPE (Rotary Position Embedding - Llama 2)
4. ALiBi (Attention with Linear Biases - BLOOM)
"""
import math
from typing import Optional

import numpy as np


# =============================================================================
# 1. SINUSOIDAL POSITIONAL ENCODING (Vaswani et al., 2017)
# =============================================================================


def sinusoidal_positional_encoding(
    max_seq_len: int,
    d_model: int,
    base: float = 10000.0,
) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    PE(pos, 2i)   = sin(pos / base^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / base^(2i/d_model))

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimension (must be even)
        base: Base for the geometric progression (default: 10000)

    Returns:
        Positional encodings of shape (max_seq_len, d_model)
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    position = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(base) / d_model))  # (d_model/2,)

    pe = np.zeros((max_seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = np.cos(position * div_term)  # Odd dimensions

    return pe


class SinusoidalPositionEncoding:
    """Sinusoidal position encoding with caching."""

    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self._cache = sinusoidal_positional_encoding(max_len, d_model, base)

    def __call__(self, seq_len: int) -> np.ndarray:
        """Get positional encodings for sequence length."""
        if seq_len > self.max_len:
            # Generate on the fly for longer sequences
            return sinusoidal_positional_encoding(seq_len, self.d_model, self.base)
        return self._cache[:seq_len]


# =============================================================================
# 2. LEARNED POSITIONAL EMBEDDINGS (BERT-style)
# =============================================================================


def learned_positional_embedding(
    max_seq_len: int,
    d_model: int,
    init_std: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate learned positional embeddings (randomly initialized).

    In real models, these would be trained. Here we simulate with random init.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Model dimension
        init_std: Standard deviation for initialization
        seed: Random seed for reproducibility

    Returns:
        Position embeddings of shape (max_seq_len, d_model)
    """
    if seed is not None:
        np.random.seed(seed)

    # Normal initialization (common in transformers)
    embeddings = np.random.randn(max_seq_len, d_model) * init_std
    return embeddings


class LearnedPositionEmbedding:
    """Learned position embeddings (simulated as fixed after 'training')."""

    def __init__(self, max_len: int, d_model: int, seed: Optional[int] = 42):
        self.max_len = max_len
        self.d_model = d_model
        self.embeddings = learned_positional_embedding(max_len, d_model, seed=seed)

    def __call__(self, seq_len: int) -> np.ndarray:
        """Get position embeddings for sequence length."""
        if seq_len > self.max_len:
            raise ValueError(
                f"Cannot extrapolate learned embeddings beyond max_len={self.max_len}. "
                f"Got seq_len={seq_len}"
            )
        return self.embeddings[:seq_len]


# =============================================================================
# 3. RoPE (ROTARY POSITION EMBEDDING) - Llama 2, GPT-NeoX
# =============================================================================


def get_rotary_matrix(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rotation matrices for RoPE.

    Args:
        seq_len: Sequence length
        dim: Dimension (should be even)
        base: Base for frequency computation

    Returns:
        Tuple of (cos, sin) matrices, each of shape (seq_len, dim)
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    # Compute frequencies: theta_i = base^(-2i/dim) for i in [0, dim/2)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))

    # Compute position * frequency for all positions
    position = np.arange(seq_len).astype(np.float32)
    freqs = np.outer(position, inv_freq)  # (seq_len, dim/2)

    # Duplicate for pairs: [freq_0, freq_0, freq_1, freq_1, ...]
    freqs = np.repeat(freqs, 2, axis=1)  # (seq_len, dim)

    cos = np.cos(freqs)
    sin = np.sin(freqs)

    return cos, sin


def apply_rotary_pos_emb(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """
    Apply rotary position embeddings to input tensor.

    For each pair of dimensions (x_i, x_{i+1}), rotate by position-dependent angle:
    [x_i']     [cos θ  -sin θ] [x_i]
    [x_{i+1}'] [sin θ   cos θ] [x_{i+1}]

    Args:
        x: Input tensor of shape (..., seq_len, dim)
        cos: Cosine matrix of shape (seq_len, dim)
        sin: Sine matrix of shape (seq_len, dim)

    Returns:
        Rotated tensor of same shape as x
    """
    # Rotate pairs: x_rotated = x * cos + rotate_half(x) * sin
    def rotate_half(x: np.ndarray) -> np.ndarray:
        """Rotate half the hidden dims of the input (for pairs)."""
        x1, x2 = np.split(x, 2, axis=-1)
        return np.concatenate([-x2, x1], axis=-1)

    return x * cos + rotate_half(x) * sin


class RotaryPositionEmbedding:
    """RoPE: Rotary Position Embedding used in Llama 2."""

    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0):
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.cos_cached, self.sin_cached = get_rotary_matrix(max_len, dim, base)

    def __call__(self, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Get cos/sin matrices for applying RoPE."""
        if seq_len > self.max_len:
            # Extend on the fly
            cos, sin = get_rotary_matrix(seq_len, self.dim, self.base)
            return cos, sin
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def rotate(self, x: np.ndarray, seq_len: Optional[int] = None) -> np.ndarray:
        """Apply RoPE to input tensor."""
        if seq_len is None:
            seq_len = x.shape[-2] if x.ndim >= 2 else x.shape[-1]
        cos, sin = self(seq_len)
        return apply_rotary_pos_emb(x, cos, sin)


# =============================================================================
# 4. ALiBi (ATTENTION WITH LINEAR BIASES) - BLOOM, MPT
# =============================================================================


def get_alibi_slopes(num_heads: int) -> np.ndarray:
    """
    Compute ALiBi slopes for each attention head.

    Slopes form a geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
    where n = num_heads.

    Args:
        num_heads: Number of attention heads

    Returns:
        Array of slopes, shape (num_heads,)
    """

    def get_slopes_power_of_2(n: int) -> np.ndarray:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return np.array([start * (ratio**i) for i in range(n)])

    # Handle non-power-of-2 num_heads
    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        # Closest power of 2
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)

        # Interpolate for remaining heads
        extra = num_heads - closest_power_of_2
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[::2][:extra]

        return np.concatenate([slopes, extra_slopes])


def get_alibi_bias(seq_len: int, num_heads: int) -> np.ndarray:
    """
    Compute ALiBi attention bias matrix.

    For each head h and positions i,j:
        bias[h, i, j] = -slope[h] * |i - j|

    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads

    Returns:
        Bias tensor of shape (num_heads, seq_len, seq_len)
    """
    slopes = get_alibi_slopes(num_heads)  # (num_heads,)

    # Create position distance matrix: |i - j|
    positions = np.arange(seq_len)
    distance_matrix = np.abs(positions[:, None] - positions[None, :])  # (seq_len, seq_len)

    # Apply slopes: bias[h] = -slope[h] * distance
    bias = -slopes[:, None, None] * distance_matrix[None, :, :]  # (num_heads, seq_len, seq_len)

    return bias


class ALiBiAttentionBias:
    """ALiBi: Attention with Linear Biases (no positional embeddings needed!)."""

    def __init__(self, num_heads: int, max_len: int = 5000):
        self.num_heads = num_heads
        self.max_len = max_len
        self.slopes = get_alibi_slopes(num_heads)
        self._cache = get_alibi_bias(max_len, num_heads)

    def __call__(self, seq_len: int) -> np.ndarray:
        """Get attention bias matrix for sequence length."""
        if seq_len > self.max_len:
            return get_alibi_bias(seq_len, self.num_heads)
        return self._cache[:, :seq_len, :seq_len]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_position_encoding(
    encoding_type: str,
    seq_len: int,
    d_model: int,
    **kwargs,
) -> np.ndarray:
    """
    Unified interface to get any positional encoding.

    Args:
        encoding_type: One of "sinusoidal", "learned", "rope", "alibi"
        seq_len: Sequence length
        d_model: Model dimension
        **kwargs: Additional arguments for specific encodings

    Returns:
        Position encodings or attention bias
    """
    if encoding_type == "sinusoidal":
        encoder = SinusoidalPositionEncoding(d_model, max_len=seq_len)
        return encoder(seq_len)
    elif encoding_type == "learned":
        encoder = LearnedPositionEmbedding(max_len=seq_len, d_model=d_model)
        return encoder(seq_len)
    elif encoding_type == "rope":
        encoder = RotaryPositionEmbedding(dim=d_model, max_len=seq_len)
        cos, sin = encoder(seq_len)
        # Return dummy input rotated for visualization
        x = np.random.randn(seq_len, d_model) * 0.02
        return encoder.rotate(x, seq_len)
    elif encoding_type == "alibi":
        num_heads = kwargs.get("num_heads", 8)
        encoder = ALiBiAttentionBias(num_heads=num_heads, max_len=seq_len)
        # Return first head's bias for visualization
        return encoder(seq_len)[0]
    else:
        raise ValueError(
            f"Unknown encoding_type: {encoding_type}. "
            f"Choose from: sinusoidal, learned, rope, alibi"
        )



