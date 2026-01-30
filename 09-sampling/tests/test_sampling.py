"""Tests for sampling methods."""

import pytest
import numpy as np

from llm_engineering_fundamentals.sampling.methods import (
    sample_with_temperature,
    top_k_sampling,
    top_p_sampling,
    sample_next_token,
)


class TestTemperatureSampling:
    """Tests for temperature sampling."""
    
    def test_temperature_zero_is_greedy(self):
        """Test that temperature=0 is equivalent to argmax."""
        logits = np.array([1.0, 5.0, 2.0, 3.0])
        
        # Temperature 0 should always return argmax
        for _ in range(10):
            token = sample_with_temperature(logits, temperature=0.0)
            assert token == 1  # Index of max value
    
    def test_temperature_one_preserves_distribution(self):
        """Test that temperature=1.0 samples from original distribution."""
        np.random.seed(42)
        logits = np.array([0.0, 0.0, 10.0, 0.0])  # Token 2 is highly favored
        
        samples = [sample_with_temperature(logits, temperature=1.0) for _ in range(100)]
        
        # Most samples should be token 2
        assert samples.count(2) > 80
    
    def test_high_temperature_increases_diversity(self):
        """Test that high temperature increases sampling diversity."""
        np.random.seed(42)
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Low temperature: more peaked
        samples_low = [sample_with_temperature(logits, temperature=0.1) for _ in range(100)]
        unique_low = len(set(samples_low))
        
        # High temperature: more diverse
        samples_high = [sample_with_temperature(logits, temperature=2.0) for _ in range(100)]
        unique_high = len(set(samples_high))
        
        # High temperature should give more unique tokens
        assert unique_high >= unique_low


class TestTopKSampling:
    """Tests for top-k sampling."""
    
    def test_top_k_filters_tokens(self):
        """Test that top-k only samples from top-k tokens."""
        np.random.seed(42)
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        k = 3
        
        samples = [top_k_sampling(logits, k=k, temperature=1.0) for _ in range(100)]
        
        # Should only sample from top-3: [7, 8, 9]
        valid_tokens = {7, 8, 9}
        assert all(s in valid_tokens for s in samples)
    
    def test_top_k_zero_no_filtering(self):
        """Test that k=0 disables filtering."""
        np.random.seed(42)
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        samples = [top_k_sampling(logits, k=0, temperature=1.0) for _ in range(200)]
        
        # Should be able to sample from all tokens
        assert len(set(samples)) >= 3


class TestTopPSampling:
    """Tests for top-p (nucleus) sampling."""
    
    def test_top_p_filters_by_probability(self):
        """Test that top-p filters by cumulative probability."""
        np.random.seed(42)
        # Create peaked distribution
        logits = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        
        samples = [top_p_sampling(logits, p=0.95, temperature=1.0) for _ in range(100)]
        
        # Should mostly sample token 4 (highest prob)
        assert samples.count(4) > 80
    
    def test_top_p_one_no_filtering(self):
        """Test that p=1.0 disables filtering."""
        np.random.seed(42)
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        samples = [top_p_sampling(logits, p=1.0, temperature=1.0) for _ in range(200)]
        
        # Should be able to sample from multiple tokens
        assert len(set(samples)) >= 3


class TestSampleNextToken:
    """Tests for combined sampling."""
    
    def test_sample_next_token_basic(self):
        """Test basic token sampling."""
        logits = np.array([1.0, 5.0, 2.0, 3.0])
        
        token = sample_next_token(logits, temperature=1.0)
        
        assert 0 <= token < len(logits)
    
    def test_repetition_penalty(self):
        """Test that repetition penalty affects sampling."""
        np.random.seed(42)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        previous_tokens = np.array([0, 0, 0])  # Token 0 repeated
        
        # Without penalty
        samples_no_penalty = [
            sample_next_token(logits, temperature=1.0, repetition_penalty=1.0)
            for _ in range(100)
        ]
        
        # With penalty
        samples_with_penalty = [
            sample_next_token(
                logits, temperature=1.0,
                repetition_penalty=2.0, previous_tokens=previous_tokens
            )
            for _ in range(100)
        ]
        
        # Token 0 should appear less frequently with penalty
        freq_no_penalty = samples_no_penalty.count(0) / 100
        freq_with_penalty = samples_with_penalty.count(0) / 100
        
        assert freq_with_penalty < freq_no_penalty
    
    def test_combined_top_k_temperature(self):
        """Test combining top-k and temperature."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        token = sample_next_token(logits, temperature=0.5, top_k=5)
        
        # Should sample from top-5
        assert token >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

