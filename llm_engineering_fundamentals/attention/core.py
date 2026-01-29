"""
Core Attention Mechanisms

Implements:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Causal Masking
"""
import math
from typing import Optional, Tuple

import numpy as np


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional mask of shape (..., seq_len_q, seq_len_k)
              True/1 positions are KEPT, False/0 are MASKED
        dropout_p: Dropout probability (not implemented in numpy version)

    Returns:
        output: Attention output of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
    """
    d_k = query.shape[-1]

    # Compute attention scores: QK^T / √d_k
    # Transpose last two dimensions regardless of number of leading dimensions
    key_transposed = np.swapaxes(key, -2, -1)
    scores = np.matmul(query, key_transposed)
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        # Convert boolean mask to float: True/1 → 0, False/0 → -inf
        mask_float = np.where(mask, 0.0, -1e9)
        scores = scores + mask_float

    # Apply softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)

    # Apply dropout (not implemented in numpy, but signature matches PyTorch)
    if dropout_p > 0.0:
        # In production, would apply dropout here
        pass

    # Compute weighted sum of values
    output = np.matmul(attention_weights, value)

    return output, attention_weights


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (lower triangular) mask for autoregressive models.

    Each position can only attend to itself and earlier positions.

    Args:
        seq_len: Sequence length

    Returns:
        Mask of shape (seq_len, seq_len) where mask[i,j] is True if position i
        can attend to position j.

    Example:
        >>> mask = create_causal_mask(4)
        >>> mask
        array([[ True, False, False, False],
               [ True,  True, False, False],
               [ True,  True,  True, False],
               [ True,  True,  True,  True]])
    """
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    return mask


def apply_attention_mask(
    attention_scores: np.ndarray,
    mask: np.ndarray,
    mask_value: float = -1e9,
) -> np.ndarray:
    """
    Apply attention mask to scores (before softmax).

    Args:
        attention_scores: Scores of shape (..., seq_len_q, seq_len_k)
        mask: Boolean mask of shape (..., seq_len_q, seq_len_k)
              True positions are KEPT, False are MASKED
        mask_value: Value to use for masked positions (large negative number)

    Returns:
        Masked scores of same shape as attention_scores
    """
    mask_float = np.where(mask, 0.0, mask_value)
    return attention_scores + mask_float


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism.

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
            seed: Random seed for weight initialization
        """
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.dropout = dropout

        # Initialize weights (in real implementation, these would be trained)
        if seed is not None:
            np.random.seed(seed)

        scale = 1.0 / math.sqrt(d_model)

        # Linear projections: W^Q, W^K, W^V, W^O
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        if bias:
            self.b_q = np.zeros(d_model)
            self.b_k = np.zeros(d_model)
            self.b_v = np.zeros(d_model)
            self.b_o = np.zeros(d_model)
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_k, d_model)
            mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k)
            return_attention_weights: Whether to return attention weights

        Returns:
            output: Output tensor of shape (batch_size, seq_len_q, d_model)
            attention_weights: Optional, shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]

        # Linear projections
        Q = np.matmul(query, self.W_q)  # (batch, seq_len_q, d_model)
        K = np.matmul(key, self.W_k)    # (batch, seq_len_k, d_model)
        V = np.matmul(value, self.W_v)  # (batch, seq_len_k, d_model)

        if self.b_q is not None:
            Q = Q + self.b_q
            K = K + self.b_k
            V = V + self.b_v

        # Reshape for multi-head: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Apply scaled dot-product attention for each head
        # Q, K, V shape: (batch, num_heads, seq_len, d_k)

        # Expand mask for multiple heads if needed
        if mask is not None and mask.ndim == 3:
            # (batch, seq_len_q, seq_len_k) → (batch, num_heads, seq_len_q, seq_len_k)
            mask = np.expand_dims(mask, 1)

        # Attention: (batch, num_heads, seq_len_q, d_k)
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout
        )

        # Concatenate heads: (batch, num_heads, seq_len_q, d_k) → (batch, seq_len_q, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)

        # Final linear projection
        output = np.matmul(attn_output, self.W_o)
        if self.b_o is not None:
            output = output + self.b_o

        if return_attention_weights:
            return output, attn_weights
        return output, None

    def __call__(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convenience method to call forward."""
        return self.forward(query, key, value, mask, return_attention_weights)


def visualize_attention_pattern(
    attention_weights: np.ndarray,
    tokens: Optional[list] = None,
) -> str:
    """
    Create a simple text visualization of attention weights.

    Args:
        attention_weights: Attention weights of shape (seq_len_q, seq_len_k)
        tokens: Optional list of token strings

    Returns:
        String representation of attention pattern
    """
    seq_len_q, seq_len_k = attention_weights.shape

    if tokens is None:
        tokens = [f"T{i}" for i in range(seq_len_k)]

    # Create header
    header = "     " + " ".join(f"{t:>6}" for t in tokens)
    lines = [header, "-" * len(header)]

    # Create rows
    for i in range(seq_len_q):
        token = tokens[i] if i < len(tokens) else f"T{i}"
        row = f"{token:>4} "
        for j in range(seq_len_k):
            weight = attention_weights[i, j]
            # Use block characters to show magnitude
            if weight > 0.5:
                char = "█"
            elif weight > 0.3:
                char = "▓"
            elif weight > 0.1:
                char = "▒"
            elif weight > 0.01:
                char = "░"
            else:
                char = " "
            row += f"{char:>6}"
        lines.append(row)

    return "\n".join(lines)

