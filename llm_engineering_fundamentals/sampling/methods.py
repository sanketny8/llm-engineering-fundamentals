"""Sampling methods for text generation."""

import numpy as np
from typing import Optional


def sample_with_temperature(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> int:
    """
    Sample from logits with temperature scaling.
    
    Lower temperature → more deterministic (peaked distribution)
    Higher temperature → more random (flatter distribution)
    
    Args:
        logits: Logits of shape (vocab_size,)
        temperature: Temperature parameter
                    - temp < 1.0: More confident/deterministic
                    - temp = 1.0: Original distribution
                    - temp > 1.0: More diverse/random
        
    Returns:
        Sampled token ID
    """
    if temperature <= 0:
        # Temperature 0 → greedy (argmax)
        return int(np.argmax(logits))
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = _softmax(scaled_logits)
    
    # Sample
    token_id = np.random.choice(len(probs), p=probs)
    
    return int(token_id)


def top_k_sampling(
    logits: np.ndarray,
    k: int = 50,
    temperature: float = 1.0,
) -> int:
    """
    Top-k sampling: only sample from top-k most likely tokens.
    
    Filters out unlikely tokens before sampling.
    
    Args:
        logits: Logits of shape (vocab_size,)
        k: Number of top tokens to consider
        temperature: Temperature for sampling
        
    Returns:
        Sampled token ID
    """
    if k <= 0:
        # No filtering
        return sample_with_temperature(logits, temperature)
    
    # Get top-k indices
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Apply temperature
    scaled_logits = top_k_logits / temperature
    
    # Convert to probabilities
    probs = _softmax(scaled_logits)
    
    # Sample from top-k
    sampled_idx = np.random.choice(len(probs), p=probs)
    token_id = top_k_indices[sampled_idx]
    
    return int(token_id)


def top_p_sampling(
    logits: np.ndarray,
    p: float = 0.9,
    temperature: float = 1.0,
) -> int:
    """
    Top-p (nucleus) sampling: sample from smallest set with cumulative prob >= p.
    
    Dynamically determines cutoff based on probability mass.
    
    Args:
        logits: Logits of shape (vocab_size,)
        p: Cumulative probability threshold (0 < p <= 1.0)
        temperature: Temperature for sampling
        
    Returns:
        Sampled token ID
    """
    if p >= 1.0:
        # No filtering
        return sample_with_temperature(logits, temperature)
    
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Get probabilities
    probs = _softmax(scaled_logits)
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Find cutoff where cumulative probability exceeds p
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
    
    # Keep only top-p tokens
    nucleus_indices = sorted_indices[:cutoff_idx]
    nucleus_probs = sorted_probs[:cutoff_idx]
    
    # Renormalize
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
    
    # Sample from nucleus
    sampled_idx = np.random.choice(len(nucleus_probs), p=nucleus_probs)
    token_id = nucleus_indices[sampled_idx]
    
    return int(token_id)


def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    previous_tokens: Optional[np.ndarray] = None,
) -> int:
    """
    Sample next token with multiple sampling strategies.
    
    Combines temperature, top-k, top-p, and repetition penalty.
    
    Args:
        logits: Logits of shape (vocab_size,)
        temperature: Temperature parameter
        top_k: Top-k filtering (0 = disabled)
        top_p: Top-p filtering (1.0 = disabled)
        repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
        previous_tokens: Previously generated tokens for repetition penalty
        
    Returns:
        Sampled token ID
    """
    # Apply repetition penalty
    if repetition_penalty != 1.0 and previous_tokens is not None:
        logits = _apply_repetition_penalty(
            logits, previous_tokens, repetition_penalty
        )
    
    # Apply top-k or top-p
    if top_k > 0:
        return top_k_sampling(logits, k=top_k, temperature=temperature)
    elif top_p < 1.0:
        return top_p_sampling(logits, p=top_p, temperature=temperature)
    else:
        return sample_with_temperature(logits, temperature=temperature)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax with numerical stability."""
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def _apply_repetition_penalty(
    logits: np.ndarray,
    previous_tokens: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """
    Apply repetition penalty to logits.
    
    Penalizes tokens that have already been generated.
    
    Args:
        logits: Original logits
        previous_tokens: Previously generated token IDs
        penalty: Penalty factor (> 1.0 penalizes repetition)
        
    Returns:
        Modified logits
    """
    logits = logits.copy()
    
    # For each previous token, divide its logit by penalty
    for token_id in previous_tokens:
        if 0 <= token_id < len(logits):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    
    return logits


def compare_sampling_methods(
    logits: np.ndarray,
    num_samples: int = 100,
) -> dict:
    """
    Compare different sampling methods.
    
    Args:
        logits: Logits to sample from
        num_samples: Number of samples per method
        
    Returns:
        Dictionary with statistics for each method
    """
    np.random.seed(42)
    
    methods = {
        "greedy": lambda: sample_with_temperature(logits, temperature=0.0),
        "temp_0.5": lambda: sample_with_temperature(logits, temperature=0.5),
        "temp_1.0": lambda: sample_with_temperature(logits, temperature=1.0),
        "temp_2.0": lambda: sample_with_temperature(logits, temperature=2.0),
        "top_k_50": lambda: top_k_sampling(logits, k=50, temperature=1.0),
        "top_p_0.9": lambda: top_p_sampling(logits, p=0.9, temperature=1.0),
    }
    
    results = {}
    for name, method in methods.items():
        samples = [method() for _ in range(num_samples)]
        unique_tokens = len(set(samples))
        most_common = max(set(samples), key=samples.count)
        most_common_freq = samples.count(most_common) / num_samples
        
        results[name] = {
            "unique_tokens": unique_tokens,
            "most_common": most_common,
            "most_common_freq": most_common_freq,
        }
    
    return results

