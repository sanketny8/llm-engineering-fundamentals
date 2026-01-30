"""Tests for attention mechanisms."""
import numpy as np
import pytest
from llm_engineering_fundamentals.attention.core import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    apply_attention_mask,
    softmax,
)


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_k, d_v = 2, 10, 64, 64
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_v)

        output, attn_weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (batch_size, seq_len, d_v)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 across key dimension."""
        batch_size, seq_len, d_k = 2, 5, 32
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_k)

        _, attn_weights = scaled_dot_product_attention(Q, K, V)

        # Sum across key dimension (axis=-1) should be 1
        sums = np.sum(attn_weights, axis=-1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-5)

    def test_attention_weights_nonnegative(self):
        """Test that attention weights are non-negative."""
        batch_size, seq_len, d_k = 2, 5, 32
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_k)

        _, attn_weights = scaled_dot_product_attention(Q, K, V)

        assert np.all(attn_weights >= 0.0)

    def test_with_causal_mask(self):
        """Test attention with causal mask."""
        seq_len, d_k = 5, 32
        Q = np.random.randn(1, seq_len, d_k)
        K = np.random.randn(1, seq_len, d_k)
        V = np.random.randn(1, seq_len, d_k)

        mask = create_causal_mask(seq_len)
        mask = np.expand_dims(mask, 0)  # Add batch dimension

        _, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Check that future positions have near-zero attention
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attn_weights[0, i, j] < 1e-6, f"Position {i} should not attend to future position {j}"


class TestCausalMask:
    """Tests for causal masking."""

    def test_mask_shape(self):
        """Test that mask has correct shape."""
        seq_len = 10
        mask = create_causal_mask(seq_len)
        assert mask.shape == (seq_len, seq_len)

    def test_mask_is_lower_triangular(self):
        """Test that mask is lower triangular."""
        seq_len = 5
        mask = create_causal_mask(seq_len)

        # Check diagonal and below are True
        for i in range(seq_len):
            for j in range(i + 1):
                assert mask[i, j] == True, f"Position ({i},{j}) should be True"

        # Check above diagonal is False
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j] == False, f"Position ({i},{j}) should be False"

    def test_mask_at_position_zero(self):
        """Test that position 0 can only attend to itself."""
        mask = create_causal_mask(10)
        assert mask[0, 0] == True
        assert np.sum(mask[0, :]) == 1  # Only one True value

    def test_mask_at_last_position(self):
        """Test that last position can attend to all positions."""
        seq_len = 10
        mask = create_causal_mask(seq_len)
        assert np.all(mask[-1, :])  # All True


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_initialization(self):
        """Test that multi-head attention initializes correctly."""
        d_model, num_heads = 512, 8
        mha = MultiHeadAttention(d_model, num_heads)

        assert mha.d_model == d_model
        assert mha.num_heads == num_heads
        assert mha.d_k == d_model // num_heads

    def test_invalid_dimensions_raises(self):
        """Test that invalid d_model/num_heads raises error."""
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadAttention(d_model=100, num_heads=7)

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len, d_model = 2, 10, 64
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads)

        Q = np.random.randn(batch_size, seq_len, d_model)
        K = np.random.randn(batch_size, seq_len, d_model)
        V = np.random.randn(batch_size, seq_len, d_model)

        output, _ = mha(Q, K, V)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_returns_attention_weights(self):
        """Test that attention weights are returned when requested."""
        batch_size, seq_len, d_model = 2, 5, 64
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads)

        Q = np.random.randn(batch_size, seq_len, d_model)
        K = np.random.randn(batch_size, seq_len, d_model)
        V = np.random.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(Q, K, V, return_attention_weights=True)

        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_with_causal_mask(self):
        """Test multi-head attention with causal mask."""
        batch_size, seq_len, d_model = 2, 5, 64
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads)

        Q = np.random.randn(batch_size, seq_len, d_model)
        K = np.random.randn(batch_size, seq_len, d_model)
        V = np.random.randn(batch_size, seq_len, d_model)

        mask = create_causal_mask(seq_len)
        mask = np.broadcast_to(mask, (batch_size, seq_len, seq_len))

        output, attn_weights = mha(Q, K, V, mask=mask, return_attention_weights=True)

        # Check that future positions have near-zero attention
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attn_weights[0, h, i, j] < 1e-5

    def test_self_attention(self):
        """Test self-attention (Q=K=V)."""
        batch_size, seq_len, d_model = 2, 5, 64
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads, seed=42)

        X = np.random.randn(batch_size, seq_len, d_model)

        output, _ = mha(X, X, X)

        assert output.shape == X.shape
        # Output should be different from input (transformed)
        assert not np.allclose(output, X)


class TestSoftmax:
    """Tests for softmax function."""

    def test_sums_to_one(self):
        """Test that softmax output sums to 1."""
        x = np.random.randn(10)
        y = softmax(x)
        assert np.isclose(np.sum(y), 1.0)

    def test_all_positive(self):
        """Test that softmax output is all positive."""
        x = np.random.randn(10)
        y = softmax(x)
        assert np.all(y > 0)

    def test_preserves_max(self):
        """Test that softmax preserves the position of maximum."""
        x = np.random.randn(10)
        y = softmax(x)
        assert np.argmax(x) == np.argmax(y)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_attention_pipeline(self):
        """Test complete attention pipeline with masking."""
        batch_size, seq_len, d_model, num_heads = 2, 8, 128, 4

        # Create input
        X = np.random.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = create_causal_mask(seq_len)
        mask = np.broadcast_to(mask, (batch_size, seq_len, seq_len))

        # Apply multi-head attention
        mha = MultiHeadAttention(d_model, num_heads, seed=42)
        output, attn_weights = mha(X, X, X, mask=mask, return_attention_weights=True)

        # Verify output
        assert output.shape == X.shape
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

        # Verify causal property
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attn_weights[0, h, i, j] < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



