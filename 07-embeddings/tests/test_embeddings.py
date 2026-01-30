"""Tests for embedding layers."""

import pytest
import numpy as np

from llm_engineering_fundamentals.embeddings.layers import (
    TokenEmbedding,
    CombinedEmbedding,
    TiedEmbedding,
    compare_tied_vs_untied,
)


class TestTokenEmbedding:
    """Tests for TokenEmbedding."""
    
    def test_basic_lookup(self):
        """Test basic token embedding lookup."""
        vocab_size = 100
        d_model = 64
        emb = TokenEmbedding(vocab_size, d_model)
        
        token_ids = np.array([[1, 2, 3]])
        embeddings = emb(token_ids)
        
        assert embeddings.shape == (1, 3, d_model)
    
    def test_batch_lookup(self):
        """Test batch token embedding lookup."""
        emb = TokenEmbedding(vocab_size=50, d_model=32)
        
        token_ids = np.array([[1, 2], [3, 4], [5, 6]])
        embeddings = emb(token_ids)
        
        assert embeddings.shape == (3, 2, 32)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        vocab_size = 1000
        d_model = 256
        emb = TokenEmbedding(vocab_size, d_model)
        
        expected_params = vocab_size * d_model
        assert emb.count_parameters() == expected_params
    
    def test_padding_idx(self):
        """Test that padding token has zero embedding."""
        vocab_size = 100
        d_model = 64
        padding_idx = 0
        
        emb = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Padding token should have zero embedding
        padding_embedding = emb.weight[padding_idx]
        np.testing.assert_array_equal(padding_embedding, np.zeros(d_model))


class TestCombinedEmbedding:
    """Tests for CombinedEmbedding."""
    
    def test_sinusoidal_positional(self):
        """Test with sinusoidal positional embeddings."""
        emb = CombinedEmbedding(
            vocab_size=1000,
            d_model=128,
            max_seq_len=512,
            positional_type="sinusoidal",
            dropout=0.0,
        )
        
        token_ids = np.array([[1, 2, 3, 4]])
        embeddings = emb(token_ids, training=False)
        
        assert embeddings.shape == (1, 4, 128)
    
    def test_learned_positional(self):
        """Test with learned positional embeddings."""
        emb = CombinedEmbedding(
            vocab_size=1000,
            d_model=128,
            max_seq_len=512,
            positional_type="learned",
            dropout=0.0,
        )
        
        token_ids = np.array([[1, 2, 3, 4]])
        embeddings = emb(token_ids, training=False)
        
        assert embeddings.shape == (1, 4, 128)
    
    def test_no_positional(self):
        """Test without positional embeddings."""
        emb = CombinedEmbedding(
            vocab_size=1000,
            d_model=128,
            positional_type="none",
            dropout=0.0,
        )
        
        token_ids = np.array([[1, 2, 3]])
        embeddings = emb(token_ids, training=False)
        
        assert embeddings.shape == (1, 3, 128)
    
    def test_scaling(self):
        """Test embedding scaling."""
        vocab_size = 100
        d_model = 64
        
        emb_no_scale = CombinedEmbedding(
            vocab_size, d_model,
            positional_type="none",
            scale_embeddings=False,
            dropout=0.0,
        )
        
        emb_with_scale = CombinedEmbedding(
            vocab_size, d_model,
            positional_type="none",
            scale_embeddings=True,
            dropout=0.0,
        )
        
        token_ids = np.array([[1, 2, 3]])
        
        out_no_scale = emb_no_scale(token_ids)
        out_with_scale = emb_with_scale(token_ids)
        
        # Scaled version should be sqrt(d_model) times larger
        # Due to random initialization, allow wider tolerance
        expected_ratio = np.sqrt(d_model)
        actual_ratio = np.mean(np.abs(out_with_scale)) / np.mean(np.abs(out_no_scale))
        
        # Check ratio is in reasonable range (random init causes variation)
        assert 5 < actual_ratio < 10  # sqrt(64) = 8, so 5-10 is reasonable
    
    def test_custom_positions(self):
        """Test with custom position indices."""
        emb = CombinedEmbedding(
            vocab_size=100,
            d_model=64,
            max_seq_len=256,
            positional_type="sinusoidal",
            dropout=0.0,
        )
        
        token_ids = np.array([[1, 2, 3]])
        positions = np.array([[10, 20, 30]])  # Non-sequential positions
        
        embeddings = emb(token_ids, positions=positions)
        
        assert embeddings.shape == (1, 3, 64)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        vocab_size = 1000
        d_model = 256
        max_seq_len = 512
        
        # With learned positional
        emb_learned = CombinedEmbedding(
            vocab_size, d_model, max_seq_len,
            positional_type="learned"
        )
        
        params_learned = emb_learned.count_parameters()
        assert params_learned["token_embedding"] == vocab_size * d_model
        assert params_learned["positional_embedding"] == max_seq_len * d_model
        
        # With sinusoidal (no parameters)
        emb_sinusoidal = CombinedEmbedding(
            vocab_size, d_model, max_seq_len,
            positional_type="sinusoidal"
        )
        
        params_sinusoidal = emb_sinusoidal.count_parameters()
        assert params_sinusoidal["positional_embedding"] == 0


class TestTiedEmbedding:
    """Tests for TiedEmbedding."""
    
    def test_embed_shape(self):
        """Test input embedding shape."""
        tied_emb = TiedEmbedding(
            vocab_size=1000,
            d_model=128,
            max_seq_len=512,
        )
        
        token_ids = np.array([[1, 2, 3, 4, 5]])
        embeddings = tied_emb.embed(token_ids)
        
        assert embeddings.shape == (1, 5, 128)
    
    def test_project_to_vocab_shape(self):
        """Test output projection shape."""
        vocab_size = 1000
        d_model = 128
        
        tied_emb = TiedEmbedding(vocab_size, d_model)
        
        # Simulate transformer output
        hidden_states = np.random.randn(2, 10, d_model)
        logits = tied_emb.project_to_vocab(hidden_states)
        
        assert logits.shape == (2, 10, vocab_size)
    
    def test_weight_sharing(self):
        """Test that weights are actually shared."""
        tied_emb = TiedEmbedding(vocab_size=100, d_model=64)
        
        # Get embedding weight
        emb_weight = tied_emb.embedding.token_embedding.weight
        
        # Project using tied weights
        hidden = np.random.randn(1, 5, 64)
        logits = tied_emb.project_to_vocab(hidden)
        
        # Manually compute logits
        expected_logits = hidden @ emb_weight.T
        
        np.testing.assert_allclose(logits, expected_logits)
    
    def test_parameter_count(self):
        """Test tied embedding parameter count."""
        vocab_size = 50000
        d_model = 768
        max_seq_len = 1024
        
        tied_emb = TiedEmbedding(vocab_size, d_model, max_seq_len, positional_type="learned")
        
        params = tied_emb.count_parameters()
        
        # Should have token + positional, but NOT separate output projection
        expected = vocab_size * d_model + max_seq_len * d_model
        assert params["total"] == expected


class TestComparisons:
    """Tests for comparison functions."""
    
    def test_compare_tied_vs_untied(self):
        """Test tied vs untied comparison."""
        comparison = compare_tied_vs_untied(
            vocab_size=50000,
            d_model=768,
            max_seq_len=1024,
        )
        
        # Check all fields present
        assert "tied_parameters" in comparison
        assert "untied_parameters" in comparison
        assert "savings" in comparison
        assert "savings_percentage" in comparison
        
        # Tied should have fewer parameters
        assert comparison["tied_parameters"] < comparison["untied_parameters"]
        
        # Savings should be vocab_size * d_model (the output projection)
        expected_savings = 50000 * 768
        assert comparison["savings"] == expected_savings
        
        # Percentage should be close to 50% (output projection is ~half of untied params)
        # Untied = 2 * vocab * d_model + max_seq * d_model
        # Tied = vocab * d_model + max_seq * d_model
        # Savings % = vocab * d_model / (2 * vocab * d_model + max_seq * d_model) ~= 49.5%
        assert 48 < comparison["savings_percentage"] < 51


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

