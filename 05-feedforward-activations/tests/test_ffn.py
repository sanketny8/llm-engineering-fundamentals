"""Tests for feed-forward networks and activation functions."""

import pytest
import numpy as np

from llm_engineering_fundamentals.ffn.activations import (
    relu,
    gelu,
    gelu_approx,
    swish,
    silu,
    relu_derivative,
    gelu_derivative,
)

from llm_engineering_fundamentals.ffn.networks import (
    FeedForwardNetwork,
    SwiGLUFFN,
    GeGLUFFN,
)


class TestActivationFunctions:
    """Tests for activation functions."""
    
    def test_relu_positive(self):
        """Test ReLU on positive inputs."""
        x = np.array([1.0, 2.0, 3.0])
        result = relu(x)
        np.testing.assert_array_equal(result, x)
    
    def test_relu_negative(self):
        """Test ReLU on negative inputs."""
        x = np.array([-1.0, -2.0, -3.0])
        result = relu(x)
        np.testing.assert_array_equal(result, np.zeros_like(x))
    
    def test_relu_mixed(self):
        """Test ReLU on mixed inputs."""
        x = np.array([-1.0, 0.0, 1.0])
        expected = np.array([0.0, 0.0, 1.0])
        result = relu(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_gelu_shape(self):
        """Test GELU preserves shape."""
        x = np.random.randn(2, 3, 4)
        result = gelu(x)
        assert result.shape == x.shape
    
    def test_gelu_at_zero(self):
        """Test GELU at x=0."""
        x = np.array([0.0])
        result = gelu(x)
        # GELU(0) = 0
        np.testing.assert_allclose(result, 0.0, atol=1e-6)
    
    def test_gelu_approx_close_to_exact(self):
        """Test that GELU approximation is close to exact."""
        x = np.linspace(-3, 3, 100)
        exact = gelu(x)
        approx = gelu_approx(x)
        
        # Should be very close
        np.testing.assert_allclose(exact, approx, rtol=1e-2, atol=1e-2)
    
    def test_swish_equals_silu(self):
        """Test that Swish (beta=1) equals SiLU."""
        x = np.random.randn(5, 10)
        result_swish = swish(x, beta=1.0)
        result_silu = silu(x)
        
        np.testing.assert_allclose(result_swish, result_silu)
    
    def test_swish_at_zero(self):
        """Test Swish at x=0."""
        x = np.array([0.0])
        result = swish(x)
        # Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        np.testing.assert_allclose(result, 0.0, atol=1e-6)
    
    def test_relu_derivative(self):
        """Test ReLU derivative."""
        x = np.array([-1.0, 0.0, 1.0])
        expected = np.array([0.0, 0.0, 1.0])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, expected)
    
    def test_gelu_derivative_shape(self):
        """Test GELU derivative preserves shape."""
        x = np.random.randn(3, 4, 5)
        result = gelu_derivative(x)
        assert result.shape == x.shape


class TestFeedForwardNetwork:
    """Tests for standard FFN."""
    
    def test_shape_preservation(self):
        """Test that FFN preserves input shape."""
        d_model = 128
        d_ff = 512
        ffn = FeedForwardNetwork(d_model, d_ff)
        
        x = np.random.randn(2, 10, d_model)
        output = ffn(x, training=False)
        
        assert output.shape == x.shape
    
    def test_different_activations(self):
        """Test FFN with different activations."""
        d_model = 64
        d_ff = 256
        
        ffn_relu = FeedForwardNetwork(d_model, d_ff, activation=relu)
        ffn_gelu = FeedForwardNetwork(d_model, d_ff, activation=gelu)
        
        x = np.random.randn(1, 5, d_model)
        
        out_relu = ffn_relu(x)
        out_gelu = ffn_gelu(x)
        
        # Should produce different outputs
        assert not np.allclose(out_relu, out_gelu)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        d_model = 256
        d_ff = 1024
        
        ffn_with_bias = FeedForwardNetwork(d_model, d_ff, bias=True)
        ffn_without_bias = FeedForwardNetwork(d_model, d_ff, bias=False)
        
        params_with = ffn_with_bias.count_parameters()
        params_without = ffn_without_bias.count_parameters()
        
        # Difference should be bias terms: d_ff + d_model
        expected_diff = d_ff + d_model
        assert params_with - params_without == expected_diff
    
    def test_dropout_effect(self):
        """Test that dropout has effect during training."""
        np.random.seed(42)
        d_model = 128
        d_ff = 512
        ffn = FeedForwardNetwork(d_model, d_ff, dropout=0.5)
        
        x = np.random.randn(1, 10, d_model)
        
        # Multiple forward passes in training mode should differ
        out1 = ffn(x, training=True)
        out2 = ffn(x, training=True)
        
        assert not np.allclose(out1, out2, rtol=1e-3)
    
    def test_no_dropout_inference(self):
        """Test that dropout is disabled during inference."""
        np.random.seed(42)
        d_model = 128
        d_ff = 512
        ffn = FeedForwardNetwork(d_model, d_ff, dropout=0.5)
        
        x = np.random.randn(1, 10, d_model)
        
        # Multiple forward passes in inference mode should be identical
        out1 = ffn(x, training=False)
        out2 = ffn(x, training=False)
        
        np.testing.assert_allclose(out1, out2)


class TestGatedFFN:
    """Tests for gated FFN variants."""
    
    def test_swiglu_shape(self):
        """Test SwiGLU preserves shape."""
        d_model = 128
        d_ff = 512
        ffn = SwiGLUFFN(d_model, d_ff)
        
        x = np.random.randn(2, 8, d_model)
        output = ffn(x, training=False)
        
        assert output.shape == x.shape
    
    def test_geglu_shape(self):
        """Test GeGLU preserves shape."""
        d_model = 128
        d_ff = 512
        ffn = GeGLUFFN(d_model, d_ff)
        
        x = np.random.randn(2, 8, d_model)
        output = ffn(x, training=False)
        
        assert output.shape == x.shape
    
    def test_gated_has_more_parameters(self):
        """Test that gated FFN has more parameters than standard."""
        d_model = 256
        d_ff = 1024
        
        standard = FeedForwardNetwork(d_model, d_ff, bias=False)
        gated = SwiGLUFFN(d_model, d_ff, bias=False)
        
        params_standard = standard.count_parameters()
        params_gated = gated.count_parameters()
        
        # Gated should have 50% more (3 matrices vs 2)
        expected_ratio = 1.5
        actual_ratio = params_gated / params_standard
        
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.01)
    
    def test_swiglu_vs_geglu(self):
        """Test that SwiGLU and GeGLU produce different outputs."""
        d_model = 64
        d_ff = 256
        
        swiglu = SwiGLUFFN(d_model, d_ff)
        geglu = GeGLUFFN(d_model, d_ff)
        
        # Use same weights for fair comparison
        swiglu.W1 = geglu.W1 = np.random.randn(d_model, d_ff) * 0.01
        swiglu.W2 = geglu.W2 = np.random.randn(d_model, d_ff) * 0.01
        swiglu.W3 = geglu.W3 = np.random.randn(d_ff, d_model) * 0.01
        
        x = np.random.randn(1, 5, d_model)
        
        out_swiglu = swiglu(x)
        out_geglu = geglu(x)
        
        # Should differ due to different activation functions
        assert not np.allclose(out_swiglu, out_geglu)
    
    def test_gated_parameter_count(self):
        """Test gated FFN parameter counting."""
        d_model = 128
        d_ff = 512
        
        ffn = SwiGLUFFN(d_model, d_ff, bias=False)
        params = ffn.count_parameters()
        
        # 3 weight matrices: W1 (d_model x d_ff) + W2 (d_model x d_ff) + W3 (d_ff x d_model)
        expected = d_model * d_ff + d_model * d_ff + d_ff * d_model
        assert params == expected


class TestFFNComparison:
    """Tests comparing different FFN architectures."""
    
    def test_all_preserve_shape(self):
        """Test that all FFN variants preserve shape."""
        d_model = 64
        d_ff = 256
        batch_size = 2
        seq_len = 8
        
        x = np.random.randn(batch_size, seq_len, d_model)
        
        ffns = [
            FeedForwardNetwork(d_model, d_ff, activation=relu),
            FeedForwardNetwork(d_model, d_ff, activation=gelu),
            SwiGLUFFN(d_model, d_ff),
            GeGLUFFN(d_model, d_ff),
        ]
        
        for ffn in ffns:
            output = ffn(x, training=False)
            assert output.shape == x.shape
    
    def test_expansion_ratios(self):
        """Test different expansion ratios."""
        d_model = 128
        expansion_ratios = [2, 4, 8]
        
        x = np.random.randn(1, 10, d_model)
        
        for ratio in expansion_ratios:
            d_ff = d_model * ratio
            ffn = FeedForwardNetwork(d_model, d_ff)
            output = ffn(x)
            
            assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

