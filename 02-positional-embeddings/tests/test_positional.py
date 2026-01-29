"""Tests for positional encoding implementations."""
import numpy as np
import pytest
from llm_engineering_fundamentals.positional.encodings import (
    SinusoidalPositionEncoding,
    LearnedPositionEmbedding,
    RotaryPositionEmbedding,
    ALiBiAttentionBias,
    get_alibi_slopes,
)


class TestSinusoidalEncoding:
    """Tests for sinusoidal positional encoding."""

    def test_shape(self):
        """Test output shape."""
        encoder = SinusoidalPositionEncoding(d_model=64, max_len=100)
        pe = encoder(50)
        assert pe.shape == (50, 64)

    def test_values_in_range(self):
        """Test that values are in [-1, 1]."""
        encoder = SinusoidalPositionEncoding(d_model=128, max_len=100)
        pe = encoder(100)
        assert np.all(pe >= -1.0) and np.all(pe <= 1.0)

    def test_deterministic(self):
        """Test that encoding is deterministic."""
        encoder1 = SinusoidalPositionEncoding(d_model=64, max_len=100)
        encoder2 = SinusoidalPositionEncoding(d_model=64, max_len=100)
        pe1 = encoder1(50)
        pe2 = encoder2(50)
        np.testing.assert_array_equal(pe1, pe2)

    def test_extrapolation(self):
        """Test that encoder can handle longer sequences."""
        encoder = SinusoidalPositionEncoding(d_model=64, max_len=100)
        pe = encoder(200)  # Request more than max_len
        assert pe.shape == (200, 64)

    def test_odd_dimension_raises(self):
        """Test that odd d_model raises ValueError."""
        with pytest.raises(ValueError, match="must be even"):
            SinusoidalPositionEncoding(d_model=63)


class TestLearnedEmbedding:
    """Tests for learned positional embeddings."""

    def test_shape(self):
        """Test output shape."""
        encoder = LearnedPositionEmbedding(max_len=100, d_model=64, seed=42)
        pe = encoder(50)
        assert pe.shape == (50, 64)

    def test_reproducible_with_seed(self):
        """Test that seed makes embeddings reproducible."""
        enc1 = LearnedPositionEmbedding(max_len=100, d_model=64, seed=42)
        enc2 = LearnedPositionEmbedding(max_len=100, d_model=64, seed=42)
        pe1 = enc1(50)
        pe2 = enc2(50)
        np.testing.assert_array_equal(pe1, pe2)

    def test_different_without_seed(self):
        """Test that different seeds give different embeddings."""
        enc1 = LearnedPositionEmbedding(max_len=100, d_model=64, seed=42)
        enc2 = LearnedPositionEmbedding(max_len=100, d_model=64, seed=43)
        pe1 = enc1(50)
        pe2 = enc2(50)
        # Should be different (with very high probability)
        assert not np.allclose(pe1, pe2)

    def test_cannot_extrapolate(self):
        """Test that learned embeddings cannot handle longer sequences."""
        encoder = LearnedPositionEmbedding(max_len=100, d_model=64, seed=42)
        with pytest.raises(ValueError, match="Cannot extrapolate"):
            encoder(200)


class TestRoPE:
    """Tests for Rotary Position Embedding."""

    def test_cos_sin_shape(self):
        """Test that cos/sin matrices have correct shape."""
        encoder = RotaryPositionEmbedding(dim=64, max_len=100)
        cos, sin = encoder(50)
        assert cos.shape == (50, 64)
        assert sin.shape == (50, 64)

    def test_rotation_is_reversible(self):
        """Test that RoPE rotation is mathematically sound."""
        encoder = RotaryPositionEmbedding(dim=64, max_len=100)
        x = np.random.randn(50, 64)
        rotated = encoder.rotate(x, 50)

        # Rotated output should have same shape
        assert rotated.shape == x.shape
        
        # Rotation should change the values (not identity)
        assert not np.allclose(x, rotated)

    def test_extrapolation(self):
        """Test that RoPE can handle longer sequences."""
        encoder = RotaryPositionEmbedding(dim=64, max_len=100)
        cos, sin = encoder(200)
        assert cos.shape == (200, 64)
        assert sin.shape == (200, 64)

    def test_odd_dimension_raises(self):
        """Test that odd dimension raises ValueError."""
        with pytest.raises(ValueError, match="must be even"):
            RotaryPositionEmbedding(dim=63)


class TestALiBi:
    """Tests for ALiBi attention bias."""

    def test_bias_shape(self):
        """Test attention bias has correct shape."""
        encoder = ALiBiAttentionBias(num_heads=8, max_len=100)
        bias = encoder(50)
        assert bias.shape == (8, 50, 50)

    def test_diagonal_is_zero(self):
        """Test that diagonal (self-attention) has zero bias."""
        encoder = ALiBiAttentionBias(num_heads=8, max_len=100)
        bias = encoder(50)
        for h in range(8):
            diag = np.diag(bias[h])
            np.testing.assert_array_equal(diag, np.zeros(50))

    def test_symmetric_around_diagonal(self):
        """Test that bias[i,j] == bias[j,i] (distance is symmetric)."""
        encoder = ALiBiAttentionBias(num_heads=8, max_len=100)
        bias = encoder(50)
        for h in range(8):
            # Check a few positions
            for i in range(0, 50, 10):
                for j in range(0, 50, 10):
                    assert bias[h, i, j] == bias[h, j, i]

    def test_distance_penalty_increases(self):
        """Test that penalty increases with distance."""
        encoder = ALiBiAttentionBias(num_heads=8, max_len=100)
        bias = encoder(50)

        # For each head, bias should become more negative with distance
        for h in range(8):
            query_pos = 10
            bias_values = bias[h, query_pos, :]
            # bias[10, 10] (distance 0) should be less negative than bias[10, 20] (distance 10)
            assert bias_values[10] > bias_values[20]
            assert bias_values[20] > bias_values[40]

    def test_slopes_correct_count(self):
        """Test that we get the right number of slopes."""
        for num_heads in [4, 8, 12, 16]:
            slopes = get_alibi_slopes(num_heads)
            assert len(slopes) == num_heads

    def test_slopes_are_decreasing(self):
        """Test that slopes form a decreasing sequence."""
        slopes = get_alibi_slopes(8)
        for i in range(len(slopes) - 1):
            assert slopes[i] >= slopes[i + 1]

    def test_extrapolation(self):
        """Test that ALiBi can handle longer sequences."""
        encoder = ALiBiAttentionBias(num_heads=8, max_len=100)
        bias = encoder(200)
        assert bias.shape == (8, 200, 200)


class TestComparison:
    """Comparative tests across encoding types."""

    def test_all_encoders_work_at_train_length(self):
        """Test that all encoders work for their training length."""
        seq_len = 128
        d_model = 64

        # Sinusoidal
        sin_enc = SinusoidalPositionEncoding(d_model, max_len=seq_len)
        assert sin_enc(seq_len).shape == (seq_len, d_model)

        # Learned
        learned_enc = LearnedPositionEmbedding(max_len=seq_len, d_model=d_model)
        assert learned_enc(seq_len).shape == (seq_len, d_model)

        # RoPE
        rope_enc = RotaryPositionEmbedding(dim=d_model, max_len=seq_len)
        cos, sin = rope_enc(seq_len)
        assert cos.shape == (seq_len, d_model)

        # ALiBi
        alibi_enc = ALiBiAttentionBias(num_heads=8, max_len=seq_len)
        assert alibi_enc(seq_len).shape == (8, seq_len, seq_len)

    def test_extrapolation_capabilities(self):
        """Test which encoders can extrapolate beyond training length."""
        train_len = 128
        test_len = 256
        d_model = 64

        # Sinusoidal: CAN extrapolate
        sin_enc = SinusoidalPositionEncoding(d_model, max_len=train_len)
        assert sin_enc(test_len).shape == (test_len, d_model)

        # Learned: CANNOT extrapolate
        learned_enc = LearnedPositionEmbedding(max_len=train_len, d_model=d_model)
        with pytest.raises(ValueError):
            learned_enc(test_len)

        # RoPE: CAN extrapolate
        rope_enc = RotaryPositionEmbedding(dim=d_model, max_len=train_len)
        cos, sin = rope_enc(test_len)
        assert cos.shape == (test_len, d_model)

        # ALiBi: CAN extrapolate
        alibi_enc = ALiBiAttentionBias(num_heads=8, max_len=train_len)
        assert alibi_enc(test_len).shape == (8, test_len, test_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

