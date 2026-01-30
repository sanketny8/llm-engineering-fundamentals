"""Activation functions used in transformer models."""

import numpy as np
import math


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU).
    
    ReLU(x) = max(0, x)
    
    Used in: Original Transformer (2017)
    
    Pros:
    - Simple and fast
    - No vanishing gradient for x > 0
    
    Cons:
    - Dead neurons (gradient = 0 for x < 0)
    - Not smooth at x = 0
    
    Args:
        x: Input array
        
    Returns:
        Activated array
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(np.float32)


def _erf(x: np.ndarray) -> np.ndarray:
    """
    Error function approximation.
    
    Uses Abramowitz and Stegun approximation (maximum error: 1.5×10⁻⁷)
    """
    # Constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)
    
    # A&S formula
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) - Exact computation.
    
    GELU(x) = x * Φ(x)
    where Φ(x) is the Gaussian CDF: Φ(x) = P(X ≤ x), X ~ N(0, 1)
    
    Equivalently:
    GELU(x) = 0.5 * x * (1 + erf(x / √2))
    
    Used in: BERT, GPT-2, GPT-3, many modern models
    
    Pros:
    - Smooth everywhere (differentiable)
    - Better gradient flow than ReLU
    - Stochastic regularization interpretation
    
    Cons:
    - Slower than ReLU (involves erf computation)
    
    Args:
        x: Input array
        
    Returns:
        Activated array
    """
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def gelu_approx(x: np.ndarray) -> np.ndarray:
    """
    GELU approximation using tanh.
    
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    This is the approximation used in many implementations for speed.
    
    Args:
        x: Input array
        
    Returns:
        Activated array
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of GELU.
    
    d/dx GELU(x) = Φ(x) + x * φ(x)
    where φ(x) is the Gaussian PDF
    """
    # Gaussian CDF
    cdf = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
    # Gaussian PDF
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    return cdf + x * pdf


def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish activation function (also called SiLU when beta=1).
    
    Swish(x) = x * σ(βx)
    where σ is the sigmoid function
    
    When β=1, this is called SiLU (Sigmoid Linear Unit).
    
    Used in: Component of SwiGLU (LLaMA, PaLM)
    
    Args:
        x: Input array
        beta: Scaling parameter (default: 1.0)
        
    Returns:
        Activated array
    """
    return x * sigmoid(beta * x)


def silu(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid Linear Unit (SiLU).
    
    SiLU(x) = x * σ(x) = Swish(x, β=1)
    
    This is equivalent to Swish with β=1.
    
    Args:
        x: Input array
        
    Returns:
        Activated array
    """
    return x * sigmoid(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip for numerical stability


def swiglu(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    SwiGLU (Swish-Gated Linear Unit).
    
    SwiGLU(x) = Swish(xW1) ⊗ (xW2)
    where ⊗ is element-wise multiplication
    
    Used in: LLaMA, PaLM, modern LLMs
    
    This is a gated activation where one path applies Swish and
    the other is linear, then they're multiplied element-wise.
    
    Args:
        x: Input of shape (..., d_model)
        W1: First projection of shape (d_model, d_ff)
        W2: Second projection of shape (d_model, d_ff)
        
    Returns:
        Gated output of shape (..., d_ff)
    """
    swish_path = swish(x @ W1)
    linear_path = x @ W2
    return swish_path * linear_path


def geglu(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    GeGLU (GELU-Gated Linear Unit).
    
    GeGLU(x) = GELU(xW1) ⊗ (xW2)
    
    Similar to SwiGLU but uses GELU instead of Swish.
    
    Used in: T5, some transformer variants
    
    Args:
        x: Input of shape (..., d_model)
        W1: First projection of shape (d_model, d_ff)
        W2: Second projection of shape (d_model, d_ff)
        
    Returns:
        Gated output of shape (..., d_ff)
    """
    gelu_path = gelu(x @ W1)
    linear_path = x @ W2
    return gelu_path * linear_path


def compare_activations(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compare different activation functions on the same input.
    
    Args:
        x: Input array
        
    Returns:
        Dictionary mapping activation names to outputs
    """
    return {
        "relu": relu(x),
        "gelu": gelu(x),
        "gelu_approx": gelu_approx(x),
        "swish": swish(x),
        "silu": silu(x),
    }


def compare_gradients(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compare gradients of different activation functions.
    
    Args:
        x: Input array
        
    Returns:
        Dictionary mapping activation names to gradients
    """
    return {
        "relu": relu_derivative(x),
        "gelu": gelu_derivative(x),
    }

