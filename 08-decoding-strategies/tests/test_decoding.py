"""Tests for decoding strategies."""

import pytest
import numpy as np

from llm_engineering_fundamentals.decoding.strategies import (
    greedy_decode,
    beam_search,
    BeamSearcher,
    DecodingConfig,
)


def dummy_model(tokens: np.ndarray) -> np.ndarray:
    """
    Dummy model for testing.
    Returns logits that favor token ID = last_token + 1
    """
    batch_size, seq_len = tokens.shape
    vocab_size = 100
    
    # Create logits
    logits = np.random.randn(batch_size, seq_len, vocab_size) * 0.1
    
    # Make next token (last_token + 1) most likely
    last_tokens = tokens[:, -1]
    next_tokens = (last_tokens + 1) % vocab_size
    
    for i in range(batch_size):
        logits[i, -1, next_tokens[i]] = 10.0  # High logit
    
    return logits


class TestGreedyDecode:
    """Tests for greedy decoding."""
    
    def test_basic_generation(self):
        """Test basic greedy generation."""
        initial_tokens = np.array([[0]])
        
        output = greedy_decode(
            model_fn=dummy_model,
            initial_tokens=initial_tokens,
            max_length=10,
        )
        
        # Should generate sequence [0, 1, 2, 3, ...]
        assert output.shape[1] == 10
        expected = np.arange(10)
        np.testing.assert_array_equal(output[0], expected)
    
    def test_eos_stopping(self):
        """Test that generation stops at EOS token."""
        initial_tokens = np.array([[0]])
        eos_token_id = 5
        
        output = greedy_decode(
            model_fn=dummy_model,
            initial_tokens=initial_tokens,
            max_length=20,
            eos_token_id=eos_token_id,
        )
        
        # Should stop at or after EOS token
        assert eos_token_id in output[0]
        eos_pos = np.where(output[0] == eos_token_id)[0][0]
        # Sequence should not extend much beyond EOS
        assert output.shape[1] <= eos_pos + 5
    
    def test_batch_generation(self):
        """Test greedy generation with batch."""
        initial_tokens = np.array([[0], [10]])
        
        output = greedy_decode(
            model_fn=dummy_model,
            initial_tokens=initial_tokens,
            max_length=5,
        )
        
        assert output.shape == (2, 5)
        # First sequence: 0, 1, 2, 3, 4
        np.testing.assert_array_equal(output[0], [0, 1, 2, 3, 4])
        # Second sequence: 10, 11, 12, 13, 14
        np.testing.assert_array_equal(output[1], [10, 11, 12, 13, 14])
    
    def test_max_length_respected(self):
        """Test that max_length is respected."""
        initial_tokens = np.array([[0]])
        max_length = 8
        
        output = greedy_decode(
            model_fn=dummy_model,
            initial_tokens=initial_tokens,
            max_length=max_length,
        )
        
        assert output.shape[1] == max_length


class TestBeamSearch:
    """Tests for beam search."""
    
    def test_basic_beam_search(self):
        """Test basic beam search."""
        initial_tokens = np.array([[0]])
        
        output = beam_search(
            model_fn=dummy_model,
            initial_tokens=initial_tokens,
            num_beams=3,
            max_length=10,
        )
        
        assert len(output) == 10
    
    def test_beam_width_one_equals_greedy(self):
        """Test that beam_width=1 is equivalent to greedy."""
        initial_tokens = np.array([[0]])
        max_length = 8
        
        greedy_output = greedy_decode(
            model_fn=dummy_model,
            initial_tokens=initial_tokens,
            max_length=max_length,
        )
        
        beam_output = beam_search(
            model_fn=dummy_model,
            initial_tokens=initial_tokens,
            num_beams=1,
            max_length=max_length,
        )
        
        # Should be identical
        np.testing.assert_array_equal(greedy_output[0], beam_output)
    
    def test_beam_searcher_class(self):
        """Test BeamSearcher class."""
        searcher = BeamSearcher(
            model_fn=dummy_model,
            num_beams=5,
            length_penalty=0.6,
        )
        
        initial_tokens = np.array([[0]])
        best_seq, best_score = searcher.search(
            initial_tokens=initial_tokens,
            max_length=10,
        )
        
        assert len(best_seq) == 10
        assert isinstance(best_score, float)
    
    def test_length_penalty(self):
        """Test that length penalty affects scores."""
        searcher_no_penalty = BeamSearcher(
            model_fn=dummy_model,
            num_beams=3,
            length_penalty=0.0,
        )
        
        searcher_with_penalty = BeamSearcher(
            model_fn=dummy_model,
            num_beams=3,
            length_penalty=1.0,
        )
        
        initial_tokens = np.array([[0]])
        
        _, score_no_penalty = searcher_no_penalty.search(initial_tokens, max_length=10)
        _, score_with_penalty = searcher_with_penalty.search(initial_tokens, max_length=10)
        
        # Scores should be different
        assert score_no_penalty != score_with_penalty


class TestDecodingConfig:
    """Tests for DecodingConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DecodingConfig()
        
        assert config.max_length == 50
        assert config.num_beams == 1
        assert config.length_penalty == 1.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DecodingConfig(
            max_length=100,
            num_beams=5,
            length_penalty=0.6,
            eos_token_id=2,
        )
        
        assert config.max_length == 100
        assert config.num_beams == 5
        assert config.length_penalty == 0.6
        assert config.eos_token_id == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

