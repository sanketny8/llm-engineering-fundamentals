"""Feed-Forward Network architectures."""

import numpy as np
from typing import Callable, Optional

from llm_engineering_fundamentals.ffn.activations import relu, gelu, swiglu, geglu


class FeedForwardNetwork:
    """
    Standard Feed-Forward Network (FFN).
    
    FFN(x) = activation(xW1 + b1)W2 + b2
    
    This is the FFN used in the original Transformer and most models.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: Callable[[np.ndarray], np.ndarray] = gelu,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (typically 4 * d_model)
            activation: Activation function (default: GELU)
            dropout: Dropout rate
            bias: Whether to use bias terms
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = dropout
        
        # Initialize weights with Xavier/Glorot initialization
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
        
        if bias:
            self.b1 = np.zeros(d_ff)
            self.b2 = np.zeros(d_model)
        else:
            self.b1 = None
            self.b2 = None
    
    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input of shape (..., d_model)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output of shape (..., d_model)
        """
        # First linear layer
        hidden = x @ self.W1
        if self.b1 is not None:
            hidden = hidden + self.b1
        
        # Activation
        hidden = self.activation(hidden)
        
        # Dropout
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, hidden.shape)
            hidden = hidden * mask / (1 - self.dropout)
        
        # Second linear layer
        output = hidden @ self.W2
        if self.b2 is not None:
            output = output + self.b2
        
        return output
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        params = self.d_model * self.d_ff + self.d_ff * self.d_model
        if self.b1 is not None:
            params += self.d_ff + self.d_model
        return params


class GatedFFN:
    """
    Base class for gated FFN variants (SwiGLU, GeGLU).
    
    Gated FFN uses 3 weight matrices instead of 2:
    GatedFFN(x) = (activation(xW1) ⊗ xW2)W3
    
    The gating mechanism allows the network to control information flow.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = False,  # Modern LLMs often don't use bias
    ):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension
            dropout: Dropout rate
            bias: Whether to use bias terms
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Three weight matrices for gating
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = np.random.randn(d_model, d_ff) * scale  # Gate path
        self.W2 = np.random.randn(d_model, d_ff) * scale  # Value path
        self.W3 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))  # Output projection
        
        if bias:
            self.b1 = np.zeros(d_ff)
            self.b2 = np.zeros(d_ff)
            self.b3 = np.zeros(d_model)
        else:
            self.b1 = self.b2 = self.b3 = None
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        # 3 weight matrices instead of 2
        params = self.d_model * self.d_ff + self.d_model * self.d_ff + self.d_ff * self.d_model
        if self.b1 is not None:
            params += self.d_ff + self.d_ff + self.d_model
        return params


class SwiGLUFFN(GatedFFN):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU(x) = (Swish(xW1) ⊗ xW2)W3
    
    Used in: LLaMA, LLaMA-2, PaLM
    
    This is the state-of-the-art FFN architecture as of 2023.
    """
    
    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input of shape (..., d_model)
            training: Whether in training mode
            
        Returns:
            Output of shape (..., d_model)
        """
        # Gate path: Swish(xW1)
        gate = x @ self.W1
        if self.b1 is not None:
            gate = gate + self.b1
        
        # Apply Swish activation
        from llm_engineering_fundamentals.ffn.activations import swish
        gate = swish(gate)
        
        # Value path: xW2
        value = x @ self.W2
        if self.b2 is not None:
            value = value + self.b2
        
        # Element-wise multiplication (gating)
        hidden = gate * value
        
        # Dropout
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, hidden.shape)
            hidden = hidden * mask / (1 - self.dropout)
        
        # Output projection
        output = hidden @ self.W3
        if self.b3 is not None:
            output = output + self.b3
        
        return output


class GeGLUFFN(GatedFFN):
    """
    GeGLU Feed-Forward Network.
    
    GeGLU(x) = (GELU(xW1) ⊗ xW2)W3
    
    Used in: T5, some transformer variants
    
    Similar to SwiGLU but uses GELU instead of Swish.
    """
    
    def __call__(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input of shape (..., d_model)
            training: Whether in training mode
            
        Returns:
            Output of shape (..., d_model)
        """
        # Gate path: GELU(xW1)
        gate = x @ self.W1
        if self.b1 is not None:
            gate = gate + self.b1
        
        # Apply GELU activation
        from llm_engineering_fundamentals.ffn.activations import gelu
        gate = gelu(gate)
        
        # Value path: xW2
        value = x @ self.W2
        if self.b2 is not None:
            value = value + self.b2
        
        # Element-wise multiplication (gating)
        hidden = gate * value
        
        # Dropout
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, hidden.shape)
            hidden = hidden * mask / (1 - self.dropout)
        
        # Output projection
        output = hidden @ self.W3
        if self.b3 is not None:
            output = output + self.b3
        
        return output


def compare_ffn_architectures(
    x: np.ndarray,
    d_model: int,
    d_ff: int,
) -> dict[str, tuple[np.ndarray, int]]:
    """
    Compare different FFN architectures.
    
    Args:
        x: Input of shape (batch, seq_len, d_model)
        d_model: Model dimension
        d_ff: FFN hidden dimension
        
    Returns:
        Dictionary mapping FFN names to (output, parameter_count)
    """
    np.random.seed(42)
    
    ffns = {
        "Standard (ReLU)": FeedForwardNetwork(d_model, d_ff, activation=relu),
        "Standard (GELU)": FeedForwardNetwork(d_model, d_ff, activation=gelu),
        "SwiGLU": SwiGLUFFN(d_model, d_ff),
        "GeGLU": GeGLUFFN(d_model, d_ff),
    }
    
    results = {}
    for name, ffn in ffns.items():
        output = ffn(x, training=False)
        params = ffn.count_parameters()
        results[name] = (output, params)
    
    return results

