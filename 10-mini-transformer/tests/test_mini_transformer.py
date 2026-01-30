"""Tests for Mini Transformer."""

import pytest
import numpy as np

from llm_engineering_fundamentals.models.mini_transformer import (
    MiniTransformer,
    MiniTransformerConfig,
    create_mini_gpt,
)


class TestMiniTransformer:
    """Tests for MiniTransformer."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        config = MiniTransformerConfig(
            vocab_size=1000,
            d_model=64,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            max_seq_len=128,
        )
        
        model = MiniTransformer(config)
        
        # Create input
        batch_size = 2
        seq_len = 10
        token_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits = model(token_ids)
        
        # Check shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_causal_mask(self):
        """Test that causal mask is applied."""
        config = MiniTransformerConfig(
            vocab_size=100,
            d_model=32,
            num_layers=1,
            num_heads=2,
        )
        
        model = MiniTransformer(config)
        
        # Create mask
        mask = model._create_causal_mask(5)
        
        # Should be lower triangular
        assert mask.shape == (5, 5)
        assert np.all(np.tril(mask) == mask)
    
    def test_greedy_generation(self):
        """Test greedy generation."""
        config = MiniTransformerConfig(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            num_heads=4,
            eos_token_id=1,
        )
        
        model = MiniTransformer(config)
        
        # Generate
        prompt = np.array([[10, 20, 30]])
        generated = model.generate_greedy(prompt, max_length=20)
        
        # Should be longer than prompt
        assert generated.shape[1] >= prompt.shape[1]
        assert generated.shape[1] <= 20
    
    def test_beam_search_generation(self):
        """Test beam search generation."""
        config = MiniTransformerConfig(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            num_heads=4,
        )
        
        model = MiniTransformer(config)
        
        # Generate
        prompt = np.array([[10, 20]])
        generated = model.generate_beam_search(prompt, num_beams=3, max_length=15)
        
        # Should be longer than prompt
        assert len(generated) >= prompt.shape[1]
        assert len(generated) <= 15
    
    def test_sampling_generation(self):
        """Test sampling generation."""
        config = MiniTransformerConfig(
            vocab_size=100,
            d_model=64,
            num_layers=2,
            num_heads=4,
            eos_token_id=1,
        )
        
        model = MiniTransformer(config)
        
        # Generate
        prompt = np.array([[10, 20, 30]])
        generated = model.generate_sample(
            prompt, max_length=20, temperature=1.0, top_k=50
        )
        
        # Should be longer than prompt
        assert generated.shape[1] >= prompt.shape[1]
        assert generated.shape[1] <= 20
    
    def test_parameter_count(self):
        """Test parameter counting."""
        config = MiniTransformerConfig(
            vocab_size=1000,
            d_model=128,
            num_layers=4,
            num_heads=8,
            d_ff=512,
        )
        
        model = MiniTransformer(config)
        params = model.count_parameters()
        
        # Should have embedding and transformer params
        assert "embedding" in params
        assert "transformer" in params
        assert "total" in params
        assert params["total"] > 0


class TestMiniTransformerConfig:
    """Tests for MiniTransformerConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MiniTransformerConfig()
        
        assert config.vocab_size == 10000
        assert config.d_model == 256
        assert config.num_layers == 6
        assert config.norm_first is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MiniTransformerConfig(
            vocab_size=5000,
            d_model=512,
            num_layers=12,
            num_heads=16,
        )
        
        assert config.vocab_size == 5000
        assert config.d_model == 512
        assert config.num_layers == 12
        assert config.num_heads == 16


class TestCreateMiniGPT:
    """Tests for create_mini_gpt helper."""
    
    def test_create_mini_gpt(self):
        """Test creating mini-GPT model."""
        model = create_mini_gpt(
            vocab_size=5000,
            d_model=256,
            num_layers=6,
        )
        
        assert isinstance(model, MiniTransformer)
        assert model.config.vocab_size == 5000
        assert model.config.d_model == 256
        assert model.config.num_layers == 6
    
    def test_forward_pass_mini_gpt(self):
        """Test forward pass on created model."""
        model = create_mini_gpt(vocab_size=1000, d_model=128, num_layers=2)
        
        token_ids = np.array([[1, 2, 3, 4, 5]])
        logits = model(token_ids)
        
        assert logits.shape == (1, 5, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

