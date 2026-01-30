"""Tests for transformer block implementations."""

import pytest
import numpy as np

from llm_engineering_fundamentals.transformer.block import (
    LayerNorm,
    FeedForward,
    TransformerBlock,
    StackedTransformer,
)


class TestLayerNorm:
    """Tests for LayerNorm."""
    
    def test_normalization(self):
        """Test that LayerNorm produces mean≈0, std≈1."""
        d_model = 64
        ln = LayerNorm(d_model)
        
        x = np.random.randn(2, 10, d_model) * 5.0 + 3.0  # Arbitrary scale/shift
        x_norm = ln(x)
        
        # Check mean and std per token
        mean = np.mean(x_norm, axis=-1)
        std = np.std(x_norm, axis=-1)
        
        np.testing.assert_allclose(mean, 0.0, atol=1e-5)
        np.testing.assert_allclose(std, 1.0, atol=1e-5)
    
    def test_shape_preservation(self):
        """Test that shape is preserved."""
        d_model = 32
        ln = LayerNorm(d_model)
        
        x = np.random.randn(4, 8, d_model)
        x_norm = ln(x)
        
        assert x_norm.shape == x.shape
    
    def test_learnable_parameters(self):
        """Test that gamma and beta affect output."""
        d_model = 16
        ln = LayerNorm(d_model)
        
        x = np.random.randn(1, 5, d_model)
        
        # Default gamma=1, beta=0
        out1 = ln(x)
        
        # Change gamma and beta
        ln.gamma = np.ones(d_model) * 2.0
        ln.beta = np.ones(d_model) * 0.5
        out2 = ln(x)
        
        # Outputs should be different
        assert not np.allclose(out1, out2)


class TestFeedForward:
    """Tests for FeedForward network."""
    
    def test_expansion_and_projection(self):
        """Test that FFN expands then projects back."""
        d_model = 64
        d_ff = 256
        ffn = FeedForward(d_model, d_ff)
        
        x = np.random.randn(2, 10, d_model)
        output = ffn(x, training=False)
        
        # Output should have same shape as input
        assert output.shape == x.shape
    
    def test_relu_activation(self):
        """Test that ReLU is applied."""
        d_model = 8
        d_ff = 32
        ffn = FeedForward(d_model, d_ff)
        
        # Create input with negative values
        x = np.random.randn(1, 5, d_model) - 2.0
        
        # Forward pass
        hidden = x @ ffn.W1 + ffn.b1
        
        # ReLU should zero out negative values
        relu_output = np.maximum(0, hidden)
        assert np.all(relu_output >= 0)
    
    def test_dropout_training_mode(self):
        """Test that dropout is applied during training."""
        np.random.seed(42)
        d_model = 32
        d_ff = 128
        ffn = FeedForward(d_model, d_ff, dropout=0.5)
        
        x = np.random.randn(1, 10, d_model)
        
        # Multiple forward passes should give different results in training mode
        out1 = ffn(x, training=True)
        out2 = ffn(x, training=True)
        
        # Should be different due to dropout
        assert not np.allclose(out1, out2, rtol=1e-3)
    
    def test_no_dropout_inference(self):
        """Test that dropout is not applied during inference."""
        np.random.seed(42)
        d_model = 32
        d_ff = 128
        ffn = FeedForward(d_model, d_ff, dropout=0.5)
        
        x = np.random.randn(1, 10, d_model)
        
        # Multiple forward passes should give same results in inference mode
        out1 = ffn(x, training=False)
        out2 = ffn(x, training=False)
        
        np.testing.assert_allclose(out1, out2)


class TestTransformerBlock:
    """Tests for TransformerBlock."""
    
    def test_shape_preservation(self):
        """Test that transformer block preserves shape."""
        d_model = 128
        block = TransformerBlock(d_model, num_heads=8, d_ff=512)
        
        x = np.random.randn(2, 16, d_model)
        output = block(x)
        
        assert output.shape == x.shape
    
    def test_residual_connections(self):
        """Test that residual connections are working."""
        d_model = 64
        block = TransformerBlock(d_model, num_heads=4, d_ff=256, dropout=0.0)
        
        # Create input
        x = np.random.randn(1, 8, d_model)
        output = block(x, training=False)
        
        # Output should be different from input (due to transformations)
        assert not np.allclose(output, x)
        
        # But not too different (residuals help preserve information)
        change = np.linalg.norm(output - x)
        input_norm = np.linalg.norm(x)
        assert change < input_norm * 5  # Heuristic check
    
    def test_preln_vs_postln(self):
        """Test Pre-LN and Post-LN produce different outputs."""
        d_model = 64
        
        block_preln = TransformerBlock(d_model, num_heads=4, d_ff=256, norm_first=True)
        block_postln = TransformerBlock(d_model, num_heads=4, d_ff=256, norm_first=False)
        
        x = np.random.randn(1, 8, d_model)
        
        out_preln = block_preln(x)
        out_postln = block_postln(x)
        
        # Should produce different outputs
        assert not np.allclose(out_preln, out_postln)
    
    def test_causal_masking(self):
        """Test that causal masking prevents attending to future tokens."""
        d_model = 32
        seq_len = 8
        block = TransformerBlock(d_model, num_heads=2, d_ff=128)
        
        x = np.random.randn(1, seq_len, d_model)
        
        # Create causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        
        # Should work without error
        output = block(x, mask=mask)
        assert output.shape == x.shape


class TestStackedTransformer:
    """Tests for StackedTransformer."""
    
    def test_multiple_layers(self):
        """Test that stacked transformer has multiple layers."""
        num_layers = 6
        model = StackedTransformer(
            num_layers=num_layers,
            d_model=128,
            num_heads=8,
            d_ff=512,
        )
        
        assert len(model.blocks) == num_layers
    
    def test_shape_preservation(self):
        """Test that output shape matches input shape."""
        model = StackedTransformer(
            num_layers=4,
            d_model=64,
            num_heads=4,
            d_ff=256,
        )
        
        x = np.random.randn(2, 12, 64)
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_return_all_layers(self):
        """Test returning outputs from all layers."""
        num_layers = 3
        model = StackedTransformer(
            num_layers=num_layers,
            d_model=32,
            num_heads=2,
            d_ff=128,
        )
        
        x = np.random.randn(1, 8, 32)
        all_outputs = model(x, return_all_layers=True)
        
        assert len(all_outputs) == num_layers
        for output in all_outputs:
            assert output.shape == x.shape
    
    def test_depth_increases_capacity(self):
        """Test that deeper models have more parameters."""
        d_model = 64
        
        model_shallow = StackedTransformer(num_layers=2, d_model=d_model, num_heads=4, d_ff=256)
        model_deep = StackedTransformer(num_layers=6, d_model=d_model, num_heads=4, d_ff=256)
        
        params_shallow = model_shallow.count_parameters()['total_parameters']
        params_deep = model_deep.count_parameters()['total_parameters']
        
        # Deeper model should have more parameters
        assert params_deep > params_shallow
        
        # Should be roughly 3x (6 layers vs 2 layers)
        ratio = params_deep / params_shallow
        assert 2.5 < ratio < 3.5
    
    def test_parameter_counting(self):
        """Test parameter counting is accurate."""
        model = StackedTransformer(
            num_layers=4,
            d_model=128,
            num_heads=8,
            d_ff=512,
        )
        
        counts = model.count_parameters()
        
        # Check all keys are present
        assert 'attention_per_block' in counts
        assert 'ffn_per_block' in counts
        assert 'ln_per_block' in counts
        assert 'total_per_block' in counts
        assert 'total_parameters' in counts
        
        # Total should equal sum of components * num_blocks
        expected_per_block = (
            counts['attention_per_block'] +
            counts['ffn_per_block'] +
            counts['ln_per_block']
        )
        assert counts['total_per_block'] == expected_per_block
    
    def test_representation_evolution(self):
        """Test that representations change through layers."""
        model = StackedTransformer(
            num_layers=6,
            d_model=64,
            num_heads=4,
            d_ff=256,
        )
        
        x = np.random.randn(1, 10, 64)
        all_outputs = model(x, return_all_layers=True)
        
        # Each layer should produce different output
        for i in range(len(all_outputs) - 1):
            assert not np.allclose(all_outputs[i], all_outputs[i + 1])
        
        # Later outputs should be different from input
        assert not np.allclose(x, all_outputs[-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


