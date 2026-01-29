#!/usr/bin/env python3
"""
Attention Mechanism Demo

Demonstrates how attention works with concrete examples and visualizations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from llm_engineering_fundamentals.attention.core import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    visualize_attention_pattern,
)


def demo_basic_attention():
    """Demo 1: Basic attention without masking."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Scaled Dot-Product Attention")
    print("=" * 70)

    # Simple example: 3 tokens, 4 dimensions
    Q = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])  # (3, 4)
    K = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0.5, 0.5, 0, 0]])  # (3, 4)
    V = np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])  # (3, 4)

    # Add batch dimension
    Q = Q[np.newaxis, ...]
    K = K[np.newaxis, ...]
    V = V[np.newaxis, ...]

    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print("\nInput:")
    print(f"  Query shape: {Q.shape}")
    print(f"  Key shape: {K.shape}")
    print(f"  Value shape: {V.shape}")

    print("\nAttention Weights (how much each query attends to each key):")
    print(attn_weights[0])

    print("\nOutput (weighted sum of values):")
    print(output[0])

    print("\n💡 Observations:")
    print("  - Token 0 attends mostly to itself (Query 0 matches Key 0)")
    print("  - Token 1 attends mostly to itself (Query 1 matches Key 1)")
    print("  - Token 2 attends somewhat to Keys 0 and 1 (partial matches)")


def demo_causal_attention():
    """Demo 2: Attention with causal masking (GPT-style)."""
    print("\n" + "=" * 70)
    print("DEMO 2: Causal (Autoregressive) Attention")
    print("=" * 70)

    seq_len = 5
    d_k = 16

    # Random query, key, value
    np.random.seed(42)
    Q = np.random.randn(1, seq_len, d_k)
    K = np.random.randn(1, seq_len, d_k)
    V = np.random.randn(1, seq_len, d_k)

    # WITHOUT mask
    _, attn_no_mask = scaled_dot_product_attention(Q, K, V)

    # WITH causal mask
    mask = create_causal_mask(seq_len)
    mask = np.expand_dims(mask, 0)  # Add batch dimension
    _, attn_with_mask = scaled_dot_product_attention(Q, K, V, mask=mask)

    print("\nAttention WITHOUT causal mask:")
    print(attn_no_mask[0])

    print("\nAttention WITH causal mask (lower triangular):")
    print(attn_with_mask[0])

    print("\n💡 Observations:")
    print("  - WITHOUT mask: Each token can attend to future tokens (CHEATING!)")
    print("  - WITH mask: Each token can only attend to past tokens (GPT-style)")
    print("  - Position 0: Can only attend to itself")
    print(f"  - Position {seq_len-1}: Can attend to all {seq_len} positions")


def demo_multihead_attention():
    """Demo 3: Multi-head attention."""
    print("\n" + "=" * 70)
    print("DEMO 3: Multi-Head Attention")
    print("=" * 70)

    batch_size, seq_len, d_model = 1, 4, 64
    num_heads = 4

    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads, seed=42)

    # Input (self-attention: Q=K=V)
    X = np.random.randn(batch_size, seq_len, d_model)

    # Forward pass with causal mask
    mask = create_causal_mask(seq_len)
    mask = np.broadcast_to(mask, (batch_size, seq_len, seq_len))

    output, attn_weights = mha(X, X, X, mask=mask, return_attention_weights=True)

    print(f"\nInput shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"  → (batch={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")

    print("\nAttention weights for each head (position 3 can attend to all):")
    for h in range(num_heads):
        print(f"\nHead {h}:")
        print(attn_weights[0, h])

    print("\n💡 Observations:")
    print(f"  - {num_heads} heads process information in parallel")
    print("  - Each head learns different attention patterns")
    print("  - Some heads focus locally, some globally")
    print("  - Results are concatenated and projected back to d_model")


def visualize_attention_patterns():
    """Create visualizations of attention patterns."""
    print("\n" + "=" * 70)
    print("DEMO 4: Visualizing Attention Patterns")
    print("=" * 70)

    tokens = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(tokens)
    d_model = 64
    num_heads = 4

    # Create input
    X = np.random.randn(1, seq_len, d_model)

    # Multi-head attention with causal mask
    mha = MultiHeadAttention(d_model, num_heads, seed=42)
    mask = create_causal_mask(seq_len)
    mask = np.broadcast_to(mask, (1, seq_len, seq_len))

    _, attn_weights = mha(X, X, X, mask=mask, return_attention_weights=True)

    # Plot all heads
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Multi-Head Attention Patterns (Causal)", fontsize=16)

    for h in range(num_heads):
        ax = axes[h // 2, h % 2]
        im = ax.imshow(attn_weights[0, h], cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"Head {h}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("03-attention/assets/attention_patterns.png", dpi=150)
    print("\n✅ Saved visualization to: 03-attention/assets/attention_patterns.png")

    # Text visualization
    print("\nText Visualization of Head 0:")
    print(visualize_attention_pattern(attn_weights[0, 0], tokens))


def compare_head_behaviors():
    """Analyze different head behaviors."""
    print("\n" + "=" * 70)
    print("DEMO 5: Analyzing Head Specialization")
    print("=" * 70)

    seq_len = 8
    d_model = 128
    num_heads = 8

    # Create longer sequence
    X = np.random.randn(1, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads, seed=42)
    mask = create_causal_mask(seq_len)
    mask = np.broadcast_to(mask, (1, seq_len, seq_len))

    _, attn_weights = mha(X, X, X, mask=mask, return_attention_weights=True)

    # Analyze each head's tendency
    print("\nHead Analysis (at final position, which can attend to all):")
    print(f"{'Head':<6} {'Local':<10} {'Global':<10} {'Prev Token':<12} {'Pattern'}")
    print("-" * 60)

    final_pos = seq_len - 1
    for h in range(num_heads):
        weights = attn_weights[0, h, final_pos, :]

        # Metrics
        local_attention = np.sum(weights[max(0, final_pos-2):final_pos+1])  # Last 3 tokens
        global_attention = np.sum(weights[:max(1, final_pos-2)])  # All earlier tokens
        prev_token_attn = weights[final_pos-1] if final_pos > 0 else 0

        # Determine pattern
        if prev_token_attn > 0.5:
            pattern = "Previous Token"
        elif local_attention > 0.7:
            pattern = "Local"
        elif weights[0] > 0.3:
            pattern = "First Token"
        else:
            pattern = "Mixed/Global"

        print(f"{h:<6} {local_attention:<10.3f} {global_attention:<10.3f} {prev_token_attn:<12.3f} {pattern}")

    print("\n💡 Observations:")
    print("  - Different heads specialize in different patterns")
    print("  - Some focus on recent context (local)")
    print("  - Some look back to beginning (global)")
    print("  - Some primarily attend to previous token")
    print("  - This diversity helps the model capture various linguistic patterns")


def main():
    """Run all demos."""
    print("=" * 70)
    print("ATTENTION MECHANISM DEMONSTRATIONS")
    print("=" * 70)

    import os
    os.makedirs("03-attention/assets", exist_ok=True)

    demo_basic_attention()
    demo_causal_attention()
    demo_multihead_attention()
    visualize_attention_patterns()
    compare_head_behaviors()

    print("\n" + "=" * 70)
    print("🎉 All Demos Complete!")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("1. Attention measures similarity between queries and keys")
    print("2. Causal masking prevents future information leakage (GPT-style)")
    print("3. Multi-head attention provides diverse attention patterns")
    print("4. Different heads specialize in different behaviors")
    print("5. Softmax ensures attention weights sum to 1")
    print("=" * 70)


if __name__ == "__main__":
    main()

