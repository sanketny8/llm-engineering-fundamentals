"""Transformer block implementation from scratch."""

import numpy as np
from typing import Optional, Literal

from llm_engineering_fundamentals.attention.core import MultiHeadAttention


class LayerNorm:
    """Layer normalization.
    
    Normalizes activations across the feature dimension (not the batch dimension).
    This helps stabilize training and enables higher learning rates.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Dimension of the model (feature dimension)
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(d_model)  # Scale parameter
        self.beta = np.zeros(d_model)   # Shift parameter
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            
        Returns:
            Normalized output of same shape
        """
        # Compute mean and variance across the feature dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta


class FeedForward:
    """Position-wise feed-forward network.
    
    Applies two linear transformations with a ReLU activation in between:
        FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Typically d_ff = 4 * d_model to add more capacity.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize weights with Xavier/Glorot initialization
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
        self.b2 = np.zeros(d_model)
    
    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Apply feed-forward network.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # First linear transformation + ReLU
        hidden = np.maximum(0, x @ self.W1 + self.b1)
        
        # Dropout (only during training)
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, hidden.shape)
            hidden = hidden * mask / (1 - self.dropout)
        
        # Second linear transformation
        output = hidden @ self.W2 + self.b2
        
        return output


class TransformerBlock:
    """A single Transformer block.
    
    Consists of:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Layer normalization (pre-LN or post-LN)
    4. Residual connections
    5. Dropout
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True,  # Pre-LN (modern) vs Post-LN (original)
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            norm_first: If True, use Pre-LN (LayerNorm before sublayer).
                       If False, use Post-LN (LayerNorm after sublayer).
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_first = norm_first
        
        # Sublayers
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> np.ndarray:
        """Apply transformer block.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            training: Whether in training mode
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        if self.norm_first:
            # Pre-LN: Modern approach (GPT-3, LLaMA)
            # x = x + Attention(LayerNorm(x))
            norm_x = self.norm1(x)
            attn_out, _ = self.attention(norm_x, norm_x, norm_x, mask=mask)
            x = x + self._dropout(attn_out, training)
            
            # x = x + FFN(LayerNorm(x))
            ffn_out = self.feed_forward(self.norm2(x), training=training)
            x = x + self._dropout(ffn_out, training)
        else:
            # Post-LN: Original transformer
            # x = LayerNorm(x + Attention(x))
            attn_out, _ = self.attention(x, x, x, mask=mask)
            x = self.norm1(x + self._dropout(attn_out, training))
            
            # x = LayerNorm(x + FFN(x))
            ffn_out = self.feed_forward(x, training=training)
            x = self.norm2(x + self._dropout(ffn_out, training))
        
        return x
    
    def _dropout(self, x: np.ndarray, training: bool) -> np.ndarray:
        """Apply dropout."""
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape)
            return x * mask / (1 - self.dropout)
        return x


class StackedTransformer:
    """Multiple transformer blocks stacked together.
    
    This is the core of models like GPT, BERT, etc.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        """
        Args:
            num_layers: Number of transformer blocks
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
            norm_first: Pre-LN vs Post-LN
        """
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Stack of transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout, norm_first)
            for _ in range(num_layers)
        ]
        
        # Final layer norm (used in Pre-LN architectures)
        self.final_norm = LayerNorm(d_model) if norm_first else None
    
    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
        return_all_layers: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """Apply stacked transformer.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            training: Whether in training mode
            return_all_layers: If True, return outputs from all layers
            
        Returns:
            Final output, or list of outputs from all layers
        """
        layer_outputs = []
        
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask, training=training)
            if return_all_layers:
                layer_outputs.append(x.copy())
        
        # Apply final layer norm in Pre-LN
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        if return_all_layers:
            return layer_outputs
        return x
    
    def count_parameters(self) -> dict:
        """Count parameters in the model.
        
        Returns:
            Dictionary with parameter counts
        """
        # Parameters per block
        attention_params = 4 * self.d_model * self.d_model  # Q, K, V, O
        ffn_params = 2 * self.d_model * self.blocks[0].d_ff  # W1, W2
        ln_params = 4 * self.d_model  # 2 LayerNorms per block, each with gamma and beta
        
        params_per_block = attention_params + ffn_params + ln_params
        total_params = params_per_block * self.num_layers
        
        if self.final_norm:
            total_params += 2 * self.d_model
        
        return {
            "attention_per_block": attention_params,
            "ffn_per_block": ffn_params,
            "ln_per_block": ln_params,
            "total_per_block": params_per_block,
            "num_blocks": self.num_layers,
            "total_parameters": total_params,
        }

