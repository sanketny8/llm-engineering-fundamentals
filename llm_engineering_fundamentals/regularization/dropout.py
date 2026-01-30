"""Dropout implementations for transformers."""

import numpy as np
from typing import Optional


class Dropout:
    """
    Standard Dropout.
    
    Randomly zeroes some elements with probability p during training.
    Scales remaining elements by 1/(1-p) to maintain expectation.
    
    Used in: All modern neural networks
    """
    
    def __init__(self, p: float = 0.1):
        """
        Args:
            p: Probability of dropping an element (drop rate)
        """
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.keep_prob = 1 - p
    
    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply dropout.
        
        Args:
            x: Input array
            training: Whether in training mode
            
        Returns:
            Array with dropout applied (if training)
        """
        if not training or self.p == 0:
            return x
        
        # Generate dropout mask
        mask = np.random.binomial(1, self.keep_prob, size=x.shape)
        
        # Apply mask and scale
        return x * mask / self.keep_prob
    
    def get_mask(self, shape: tuple) -> np.ndarray:
        """
        Generate dropout mask without applying it.
        
        Useful for visualization.
        
        Args:
            shape: Shape of the mask
            
        Returns:
            Binary mask
        """
        return np.random.binomial(1, self.keep_prob, size=shape)


class AttentionDropout:
    """
    Attention Dropout.
    
    Applies dropout to attention weights (after softmax).
    This is different from standard dropout as it's applied to the
    attention probability distribution.
    
    Used in: Transformers (on attention weights)
    """
    
    def __init__(self, p: float = 0.1):
        """
        Args:
            p: Dropout probability
        """
        self.dropout = Dropout(p)
    
    def __call__(
        self,
        attention_weights: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """
        Apply dropout to attention weights.
        
        Args:
            attention_weights: Attention weights of shape (..., seq_len_q, seq_len_k)
                              Should already be normalized (softmax applied)
            training: Whether in training mode
            
        Returns:
            Attention weights with dropout applied
        """
        return self.dropout(attention_weights, training=training)


class DropPath:
    """
    DropPath (Stochastic Depth).
    
    Randomly drops entire paths (residual connections) during training.
    More aggressive than standard dropout - drops entire blocks instead of neurons.
    
    Used in: Vision Transformers, modern architectures for efficiency
    
    Formula:
        if training and random() < drop_prob:
            return x  # Skip the block
        else:
            return x + block(x)  # Normal residual
    """
    
    def __init__(self, drop_prob: float = 0.0):
        """
        Args:
            drop_prob: Probability of dropping the path
        """
        if not 0 <= drop_prob < 1:
            raise ValueError(f"DropPath probability must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob
    
    def __call__(
        self,
        x: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """
        Apply DropPath to residual.
        
        Args:
            x: Residual to potentially drop (shape: batch, seq_len, d_model)
            training: Whether in training mode
            
        Returns:
            x (potentially zeroed out)
        """
        if not training or self.drop_prob == 0:
            return x
        
        # Generate survival mask (one per sample in batch)
        batch_size = x.shape[0]
        keep_mask = np.random.binomial(1, self.keep_prob, size=(batch_size, 1, 1))
        
        # Scale by keep_prob to maintain expectation
        return x * keep_mask / self.keep_prob
    
    @staticmethod
    def get_drop_prob_schedule(
        num_layers: int,
        final_drop_prob: float = 0.1,
    ) -> list[float]:
        """
        Generate linear schedule for drop probabilities.
        
        Deeper layers have higher drop probability (linear increase).
        
        Args:
            num_layers: Total number of layers
            final_drop_prob: Drop probability for final layer
            
        Returns:
            List of drop probabilities per layer
        """
        return [i / (num_layers - 1) * final_drop_prob for i in range(num_layers)]


class EmbeddingDropout:
    """
    Embedding Dropout.
    
    Applies dropout to token embeddings.
    Can optionally drop entire tokens (all embedding dimensions).
    
    Used in: Input embedding layers
    """
    
    def __init__(self, p: float = 0.1, drop_entire_tokens: bool = False):
        """
        Args:
            p: Dropout probability
            drop_entire_tokens: If True, drop entire token embeddings.
                               If False, drop individual dimensions.
        """
        self.p = p
        self.keep_prob = 1 - p
        self.drop_entire_tokens = drop_entire_tokens
    
    def __call__(
        self,
        embeddings: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """
        Apply dropout to embeddings.
        
        Args:
            embeddings: Token embeddings of shape (batch, seq_len, d_model)
            training: Whether in training mode
            
        Returns:
            Embeddings with dropout applied
        """
        if not training or self.p == 0:
            return embeddings
        
        batch, seq_len, d_model = embeddings.shape
        
        if self.drop_entire_tokens:
            # Drop entire tokens (all dimensions)
            mask = np.random.binomial(1, self.keep_prob, size=(batch, seq_len, 1))
        else:
            # Drop individual dimensions
            mask = np.random.binomial(1, self.keep_prob, size=embeddings.shape)
        
        return embeddings * mask / self.keep_prob


def apply_dropout_schedule(
    dropout_rates: dict[str, float],
) -> dict[str, Dropout]:
    """
    Create dropout modules from configuration.
    
    Args:
        dropout_rates: Dictionary mapping dropout type to rate
                      e.g. {"attention": 0.1, "residual": 0.1, "embedding": 0.1}
    
    Returns:
        Dictionary of initialized Dropout modules
    """
    dropout_modules = {}
    
    for name, rate in dropout_rates.items():
        if name == "attention":
            dropout_modules[name] = AttentionDropout(rate)
        elif name == "embedding":
            dropout_modules[name] = EmbeddingDropout(rate)
        elif name == "droppath":
            dropout_modules[name] = DropPath(rate)
        else:
            dropout_modules[name] = Dropout(rate)
    
    return dropout_modules

