"""Tests for dropout and regularization."""

import pytest
import numpy as np

from llm_engineering_fundamentals.regularization.dropout import (
    Dropout,
    AttentionDropout,
    DropPath,
    EmbeddingDropout,
)

from llm_engineering_fundamentals.regularization.techniques import (
    gradient_clip_norm,
    gradient_clip_value,
    weight_decay_step,
    label_smoothing,
    compute_l2_loss,
    compute_smoothed_cross_entropy,
    GradientAccumulator,
)


class TestDropout:
    """Tests for Dropout."""
    
    def test_dropout_inference_mode(self):
        """Test that dropout is disabled in inference mode."""
        dropout = Dropout(p=0.5)
        x = np.ones((10, 20))
        
        output = dropout(x, training=False)
        
        # Should be identical in inference mode
        np.testing.assert_array_equal(output, x)
    
    def test_dropout_training_mode(self):
        """Test that dropout drops elements in training mode."""
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        x = np.ones((1000, 100))
        
        output = dropout(x, training=True)
        
        # Should have approximately 50% zeros
        zero_ratio = np.sum(output == 0) / output.size
        assert 0.45 < zero_ratio < 0.55  # Allow 5% deviation
    
    def test_dropout_expectation(self):
        """Test that dropout maintains expectation."""
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        x = np.ones((10000,))
        
        output = dropout(x, training=True)
        
        # Mean should be close to 1.0 due to scaling
        assert abs(np.mean(output) - 1.0) < 0.1
    
    def test_dropout_shape_preservation(self):
        """Test that dropout preserves shape."""
        dropout = Dropout(p=0.3)
        x = np.random.randn(2, 3, 4, 5)
        
        output = dropout(x, training=True)
        
        assert output.shape == x.shape
    
    def test_dropout_zero_rate(self):
        """Test dropout with p=0 (no dropout)."""
        dropout = Dropout(p=0.0)
        x = np.random.randn(5, 10)
        
        output = dropout(x, training=True)
        
        np.testing.assert_array_equal(output, x)


class TestAttentionDropout:
    """Tests for AttentionDropout."""
    
    def test_attention_dropout_applies(self):
        """Test that attention dropout is applied."""
        np.random.seed(42)
        attn_dropout = AttentionDropout(p=0.3)
        
        # Simulated attention weights
        attn_weights = np.random.rand(1, 4, 8, 8)
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        output = attn_dropout(attn_weights, training=True)
        
        # Should have some zeros
        assert np.sum(output == 0) > 0
    
    def test_attention_dropout_inference(self):
        """Test that attention dropout is disabled in inference."""
        attn_dropout = AttentionDropout(p=0.5)
        attn_weights = np.random.rand(1, 2, 4, 4)
        
        output = attn_dropout(attn_weights, training=False)
        
        np.testing.assert_array_equal(output, attn_weights)


class TestDropPath:
    """Tests for DropPath."""
    
    def test_droppath_inference(self):
        """Test that droppath is disabled in inference."""
        droppath = DropPath(drop_prob=0.5)
        x = np.random.randn(4, 8, 64)
        
        output = droppath(x, training=False)
        
        np.testing.assert_array_equal(output, x)
    
    def test_droppath_drops_paths(self):
        """Test that droppath drops paths during training."""
        np.random.seed(42)
        droppath = DropPath(drop_prob=0.9)  # High probability
        x = np.random.randn(100, 8, 64)
        
        num_dropped = 0
        for i in range(100):
            output = droppath(x[[i]], training=True)
            if np.allclose(output, 0):
                num_dropped += 1
        
        # Should drop approximately 90% of paths
        assert 80 < num_dropped < 100
    
    def test_droppath_schedule(self):
        """Test droppath schedule generation."""
        schedule = DropPath.get_drop_prob_schedule(12, 0.2)
        
        assert len(schedule) == 12
        assert schedule[0] == 0.0
        assert schedule[-1] == 0.2
        # Should be monotonically increasing
        assert all(schedule[i] <= schedule[i+1] for i in range(len(schedule)-1))


class TestEmbeddingDropout:
    """Tests for EmbeddingDropout."""
    
    def test_embedding_dropout_element_wise(self):
        """Test element-wise embedding dropout."""
        np.random.seed(42)
        emb_dropout = EmbeddingDropout(p=0.3, drop_entire_tokens=False)
        embeddings = np.ones((2, 8, 64))
        
        output = emb_dropout(embeddings, training=True)
        
        # Should have some zeros but not entire tokens
        assert np.sum(output == 0) > 0
        # At least some tokens should be partially preserved
        assert np.any(np.sum(output, axis=-1) > 0)
    
    def test_embedding_dropout_token_wise(self):
        """Test token-wise embedding dropout."""
        np.random.seed(42)
        emb_dropout = EmbeddingDropout(p=0.5, drop_entire_tokens=True)
        embeddings = np.ones((4, 10, 32))
        
        output = emb_dropout(embeddings, training=True)
        
        # Check for fully dropped tokens
        token_sums = np.sum(output, axis=-1)
        fully_dropped = np.sum(token_sums == 0)
        
        # Should have some fully dropped tokens
        assert fully_dropped > 0


class TestGradientClipping:
    """Tests for gradient clipping."""
    
    def test_gradient_clip_norm_single(self):
        """Test norm clipping with single gradient."""
        grad = np.array([3.0, 4.0])  # Norm = 5.0
        max_norm = 1.0
        
        clipped = gradient_clip_norm(grad, max_norm)
        
        # Norm should be max_norm
        actual_norm = np.linalg.norm(clipped)
        np.testing.assert_allclose(actual_norm, max_norm, rtol=1e-5)
    
    def test_gradient_clip_norm_no_clipping(self):
        """Test that small gradients are not clipped."""
        grad = np.array([0.1, 0.2])
        max_norm = 10.0
        
        clipped = gradient_clip_norm(grad, max_norm)
        
        np.testing.assert_array_equal(clipped, grad)
    
    def test_gradient_clip_value(self):
        """Test value clipping."""
        grad = np.array([-5.0, 3.0, 1.0])
        clip_value = 2.0
        
        clipped = gradient_clip_value(grad, clip_value)
        
        expected = np.array([-2.0, 2.0, 1.0])
        np.testing.assert_array_equal(clipped, expected)


class TestWeightDecay:
    """Tests for weight decay."""
    
    def test_weight_decay_step(self):
        """Test weight decay application."""
        weights = np.ones((10, 10))
        weight_decay = 0.01
        learning_rate = 0.1
        
        decayed = weight_decay_step(weights, weight_decay, learning_rate)
        
        # Weights should be smaller
        assert np.all(decayed < weights)
        
        # Expected decay factor
        expected_factor = 1 - learning_rate * weight_decay
        np.testing.assert_allclose(decayed, weights * expected_factor)
    
    def test_compute_l2_loss(self):
        """Test L2 loss computation."""
        weights = np.array([1.0, 2.0, 3.0])
        l2_loss = compute_l2_loss(weights)
        
        expected = 1.0**2 + 2.0**2 + 3.0**2  # = 14.0
        assert l2_loss == expected


class TestLabelSmoothing:
    """Tests for label smoothing."""
    
    def test_label_smoothing_shape(self):
        """Test that label smoothing produces correct shape."""
        targets = np.array([0, 1, 2])
        num_classes = 5
        
        smoothed = label_smoothing(targets, num_classes, smoothing=0.1)
        
        assert smoothed.shape == (3, 5)
    
    def test_label_smoothing_values(self):
        """Test label smoothing values."""
        targets = np.array([0])
        num_classes = 4
        smoothing = 0.1
        
        smoothed = label_smoothing(targets, num_classes, smoothing)
        
        # Formula: (1 - smoothing) * one_hot + smoothing / num_classes
        # Target class: (1 - 0.1) * 1 + 0.1 / 4 = 0.9 + 0.025 = 0.925
        assert smoothed[0, 0] == pytest.approx(0.925)
        
        # Other classes: (1 - 0.1) * 0 + 0.1 / 4 = 0.025
        assert smoothed[0, 1] == pytest.approx(0.025)
    
    def test_label_smoothing_sum_to_one(self):
        """Test that smoothed labels sum to 1."""
        targets = np.array([0, 1, 2])
        smoothed = label_smoothing(targets, 5, smoothing=0.1)
        
        # Each row should sum to 1
        row_sums = np.sum(smoothed, axis=1)
        np.testing.assert_allclose(row_sums, 1.0)


class TestGradientAccumulator:
    """Tests for GradientAccumulator."""
    
    def test_accumulator_basic(self):
        """Test basic gradient accumulation."""
        accumulator = GradientAccumulator(accumulation_steps=4)
        
        # Accumulate 4 gradients
        for i in range(3):
            grad = np.ones((5, 10))
            should_update = accumulator.accumulate(grad)
            assert not should_update
        
        # 4th gradient should trigger update
        should_update = accumulator.accumulate(np.ones((5, 10)))
        assert should_update
    
    def test_accumulator_averaging(self):
        """Test that gradients are averaged."""
        accumulator = GradientAccumulator(accumulation_steps=2)
        
        grad1 = np.ones((3, 3))
        grad2 = np.ones((3, 3)) * 2.0
        
        accumulator.accumulate(grad1)
        accumulator.accumulate(grad2)
        
        avg_grads = accumulator.get_gradients()
        
        # Average should be 1.5
        np.testing.assert_allclose(avg_grads[0], 1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

