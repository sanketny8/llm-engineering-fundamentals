"""Regularization techniques for training stability."""

import numpy as np
from typing import Union, List


def gradient_clip_norm(
    gradients: Union[np.ndarray, List[np.ndarray]],
    max_norm: float,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Clip gradients by global norm.
    
    If ||g|| > max_norm, scale g to have norm = max_norm.
    This prevents exploding gradients.
    
    Args:
        gradients: Single gradient array or list of gradient arrays
        max_norm: Maximum allowed norm
        
    Returns:
        Clipped gradients (same structure as input)
    """
    # Handle single gradient or list of gradients
    is_list = isinstance(gradients, list)
    grad_list = gradients if is_list else [gradients]
    
    # Compute global norm
    total_norm = 0.0
    for g in grad_list:
        if g is not None:
            total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        clipped_grads = [g * clip_coef if g is not None else None for g in grad_list]
    else:
        clipped_grads = grad_list
    
    # Return in same format as input
    return clipped_grads if is_list else clipped_grads[0]


def gradient_clip_value(
    gradients: Union[np.ndarray, List[np.ndarray]],
    clip_value: float,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Clip gradients by value.
    
    Clips each gradient element to [-clip_value, clip_value].
    Less common than norm clipping but simpler.
    
    Args:
        gradients: Single gradient array or list of gradient arrays
        clip_value: Maximum absolute value
        
    Returns:
        Clipped gradients
    """
    is_list = isinstance(gradients, list)
    grad_list = gradients if is_list else [gradients]
    
    clipped_grads = [
        np.clip(g, -clip_value, clip_value) if g is not None else None
        for g in grad_list
    ]
    
    return clipped_grads if is_list else clipped_grads[0]


def weight_decay_step(
    weights: np.ndarray,
    weight_decay: float,
    learning_rate: float,
) -> np.ndarray:
    """
    Apply weight decay (L2 regularization).
    
    Weight decay adds a penalty for large weights:
        Loss = L_task + (weight_decay / 2) * ||W||²
    
    This is equivalent to:
        W ← W - learning_rate * weight_decay * W
        W ← (1 - learning_rate * weight_decay) * W
    
    Args:
        weights: Weight matrix
        weight_decay: L2 penalty coefficient (typically 0.01-0.1)
        learning_rate: Current learning rate
        
    Returns:
        Weights after decay
    """
    decay_factor = 1 - learning_rate * weight_decay
    return weights * decay_factor


def compute_l2_loss(weights: Union[np.ndarray, List[np.ndarray]]) -> float:
    """
    Compute L2 regularization loss.
    
    L2_loss = Σ(W²)
    
    Args:
        weights: Single weight matrix or list of weight matrices
        
    Returns:
        L2 loss value
    """
    if isinstance(weights, np.ndarray):
        return float(np.sum(weights ** 2))
    else:
        return sum(float(np.sum(w ** 2)) for w in weights if w is not None)


def label_smoothing(
    targets: np.ndarray,
    num_classes: int,
    smoothing: float = 0.1,
) -> np.ndarray:
    """
    Apply label smoothing to targets.
    
    Instead of hard targets (one-hot), use soft targets:
        y_smooth = (1 - smoothing) * y_hard + smoothing / num_classes
    
    This prevents overconfidence and improves calibration.
    
    Args:
        targets: Target indices of shape (batch_size,)
        num_classes: Number of classes
        smoothing: Smoothing factor (typically 0.1)
        
    Returns:
        Smoothed targets of shape (batch_size, num_classes)
    """
    if not 0 <= smoothing < 1:
        raise ValueError(f"Smoothing must be in [0, 1), got {smoothing}")
    
    batch_size = len(targets)
    
    # Create one-hot encoding
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), targets] = 1.0
    
    # Apply smoothing
    smooth_targets = (1 - smoothing) * one_hot + smoothing / num_classes
    
    return smooth_targets


def compute_smoothed_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 0.1,
) -> float:
    """
    Compute cross-entropy loss with label smoothing.
    
    Args:
        logits: Model output of shape (batch_size, num_classes)
        targets: Target indices of shape (batch_size,)
        smoothing: Smoothing factor
        
    Returns:
        Smoothed cross-entropy loss
    """
    batch_size, num_classes = logits.shape
    
    # Apply softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Create smoothed targets
    smooth_targets = label_smoothing(targets, num_classes, smoothing)
    
    # Compute cross-entropy
    log_probs = np.log(probs + 1e-10)
    loss = -np.sum(smooth_targets * log_probs) / batch_size
    
    return loss


class GradientAccumulator:
    """
    Accumulate gradients over multiple steps.
    
    Used to simulate larger batch sizes when memory is limited.
    """
    
    def __init__(self, accumulation_steps: int):
        """
        Args:
            accumulation_steps: Number of steps to accumulate before update
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_grads = None
    
    def accumulate(self, gradients: Union[np.ndarray, List[np.ndarray]]) -> bool:
        """
        Accumulate gradients.
        
        Args:
            gradients: Gradients from current batch
            
        Returns:
            True if accumulated enough steps (should update), False otherwise
        """
        is_list = isinstance(gradients, list)
        grad_list = gradients if is_list else [gradients]
        
        # Initialize on first call
        if self.accumulated_grads is None:
            self.accumulated_grads = [np.zeros_like(g) for g in grad_list]
        
        # Accumulate
        for i, g in enumerate(grad_list):
            self.accumulated_grads[i] += g
        
        self.current_step += 1
        
        # Check if we should update
        if self.current_step >= self.accumulation_steps:
            return True
        return False
    
    def get_gradients(self) -> List[np.ndarray]:
        """
        Get averaged accumulated gradients.
        
        Returns:
            Averaged gradients
        """
        if self.accumulated_grads is None:
            raise ValueError("No gradients accumulated yet")
        
        # Average over accumulation steps
        averaged_grads = [g / self.accumulation_steps for g in self.accumulated_grads]
        return averaged_grads
    
    def reset(self):
        """Reset accumulator."""
        self.current_step = 0
        self.accumulated_grads = None


def apply_regularization(
    weights: List[np.ndarray],
    gradients: List[np.ndarray],
    weight_decay: float = 0.01,
    clip_norm: float = 1.0,
    learning_rate: float = 0.001,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply regularization techniques (weight decay + gradient clipping).
    
    Args:
        weights: List of weight matrices
        gradients: List of gradients
        weight_decay: L2 penalty coefficient
        clip_norm: Maximum gradient norm
        learning_rate: Current learning rate
        
    Returns:
        Tuple of (regularized_weights, clipped_gradients)
    """
    # Clip gradients
    clipped_grads = gradient_clip_norm(gradients, clip_norm)
    
    # Apply weight decay
    regularized_weights = [
        weight_decay_step(w, weight_decay, learning_rate)
        for w in weights
    ]
    
    return regularized_weights, clipped_grads

