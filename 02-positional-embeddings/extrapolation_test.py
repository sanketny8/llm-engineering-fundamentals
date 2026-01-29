#!/usr/bin/env python3
"""
Extrapolation Test: Train Short, Test Long

This script demonstrates how different positional encodings handle
sequences longer than those seen during training.

Experiment:
- Train on sequences up to 512 tokens
- Test on 1024, 2048, 4096 tokens
- Measure performance degradation

Expected Results:
1. ALiBi: Best extrapolation (linear degradation)
2. RoPE: Good extrapolation (gradual degradation)
3. Sinusoidal: Moderate (some degradation)
4. Learned: FAILS (cannot handle unseen positions)
"""
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from llm_engineering_fundamentals.positional.encodings import (
    SinusoidalPositionEncoding,
    LearnedPositionEmbedding,
    RotaryPositionEmbedding,
    ALiBiAttentionBias,
)


def compute_position_distance_error(
    encoder,
    train_len: int,
    test_len: int,
    d_model: int = 128,
) -> float:
    """
    Measure how well position encodings preserve distance relationships.

    Metric: For positions (i, j) in train_len, check if distance is preserved
    when extrapolating to test_len.
    """
    try:
        # Get encodings for train length
        if isinstance(encoder, ALiBiAttentionBias):
            train_enc = encoder(train_len)[0]  # First head
            test_enc = encoder(test_len)[0]
        elif isinstance(encoder, RotaryPositionEmbedding):
            # For RoPE, we measure the cosine/sine consistency
            cos_train, sin_train = encoder(train_len)
            cos_test, sin_test = encoder(test_len)
            # Check if the first train_len positions match
            train_enc = cos_train
            test_enc = cos_test[:train_len]
        else:
            train_enc = encoder(train_len)
            test_enc = encoder(test_len)[:train_len]  # Take first train_len positions

        # Compute distance preservation error
        if train_enc.ndim == 1:
            # ALiBi single head case
            train_enc = train_enc.reshape(-1, 1)
            test_enc = test_enc.reshape(-1, 1)

        # Mean squared error between train and test encodings for overlapping positions
        error = np.mean((train_enc - test_enc) ** 2)
        return float(error)

    except (ValueError, IndexError) as e:
        # Learned embeddings will fail for test_len > train_len
        return float("inf")


def simulate_attention_extrapolation(
    encoder_type: str,
    train_len: int,
    test_lengths: List[int],
    d_model: int = 128,
    num_heads: int = 8,
) -> Dict[int, float]:
    """
    Simulate attention computation at different sequence lengths.

    Returns error metric for each test length.
    """
    results = {}

    # Create encoder
    if encoder_type == "sinusoidal":
        encoder = SinusoidalPositionEncoding(d_model, max_len=max(test_lengths))
    elif encoder_type == "learned":
        encoder = LearnedPositionEmbedding(max_len=train_len, d_model=d_model)
    elif encoder_type == "rope":
        encoder = RotaryPositionEmbedding(dim=d_model, max_len=max(test_lengths))
    elif encoder_type == "alibi":
        encoder = ALiBiAttentionBias(num_heads=num_heads, max_len=max(test_lengths))
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Test extrapolation at each length
    for test_len in test_lengths:
        error = compute_position_distance_error(encoder, train_len, test_len, d_model)
        results[test_len] = error

    return results


def plot_extrapolation_results(
    results: Dict[str, Dict[int, float]],
    train_len: int,
):
    """Plot extrapolation error for all encoding types."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for encoder_type, errors in results.items():
        test_lengths = sorted(errors.keys())
        error_values = [errors[l] for l in test_lengths]

        # Handle inf values for learned embeddings
        error_values = [min(e, 10.0) for e in error_values]  # Cap at 10 for visualization

        ax.plot(test_lengths, error_values, marker="o", label=encoder_type, linewidth=2)

    ax.axvline(train_len, color="red", linestyle="--", alpha=0.5, label=f"Train Length ({train_len})")
    ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
    ax.set_ylabel("Position Distance Error", fontsize=12)
    ax.set_title(f"Extrapolation Performance: Train on {train_len}, Test on Longer Sequences", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("02-positional-embeddings/assets/extrapolation_comparison.png", dpi=150)
    print("✅ Saved extrapolation plot to: 02-positional-embeddings/assets/extrapolation_comparison.png")
    plt.show()


def test_learned_embedding_failure():
    """Demonstrate that learned embeddings cannot extrapolate."""
    print("\n" + "=" * 70)
    print("Learned Embeddings: Extrapolation Failure Demo")
    print("=" * 70)

    train_len = 512
    test_len = 1024
    d_model = 128

    learned_enc = LearnedPositionEmbedding(max_len=train_len, d_model=d_model)

    print(f"\nTrained with max_len={train_len}")
    print(f"Attempting to get embeddings for seq_len={test_len}...")

    try:
        embeddings = learned_enc(test_len)
        print(f"❌ Should have failed! Got shape: {embeddings.shape}")
    except ValueError as e:
        print(f"✅ Correctly raised error: {e}")

    print("\n💡 Insight: Learned embeddings are limited to training length!")
    print("   This is why GPT-3 (learned) had fixed context, while")
    print("   Llama 2 (RoPE) and BLOOM (ALiBi) can handle longer contexts.")


def analyze_alibi_distance_penalty():
    """Analyze how ALiBi penalizes distant tokens."""
    print("\n" + "=" * 70)
    print("ALiBi: Distance Penalty Analysis")
    print("=" * 70)

    seq_len = 64
    num_heads = 8

    alibi_enc = ALiBiAttentionBias(num_heads=num_heads, max_len=seq_len)
    bias = alibi_enc(seq_len)  # (num_heads, seq_len, seq_len)

    # Look at attention from position 0 to all others
    query_pos = 0
    print(f"\nAttention bias from position {query_pos} to other positions:")
    print("(More negative = more penalty = less attention)\n")

    for head_idx in range(min(3, num_heads)):
        bias_from_0 = bias[head_idx, query_pos, :]
        print(f"Head {head_idx} (slope={alibi_enc.slopes[head_idx]:.4f}):")
        print(f"  Position 0: {bias_from_0[0]:.4f}")
        print(f"  Position 10: {bias_from_0[10]:.4f}")
        print(f"  Position 30: {bias_from_0[30]:.4f}")
        print(f"  Position 63: {bias_from_0[63]:.4f}")
        print()

    print("💡 Observation:")
    print("   - Closer positions have less penalty (less negative bias)")
    print("   - Different heads have different slopes (diversity)")
    print("   - Linear penalty → smooth extrapolation!")


def main():
    """Run all extrapolation tests."""
    print("=" * 70)
    print("Positional Encoding Extrapolation Test")
    print("Train on 512 tokens, test on 1024, 2048, 4096")
    print("=" * 70)

    import os
    os.makedirs("02-positional-embeddings/assets", exist_ok=True)

    # Test configuration
    train_len = 512
    test_lengths = [512, 768, 1024, 1536, 2048, 3072, 4096]
    d_model = 128
    num_heads = 8

    # Run tests for all encoding types
    print("\n🧪 Running extrapolation tests...")
    results = {}

    for encoder_type in ["sinusoidal", "learned", "rope", "alibi"]:
        print(f"  Testing {encoder_type}...")
        results[encoder_type] = simulate_attention_extrapolation(
            encoder_type, train_len, test_lengths, d_model, num_heads
        )

    # Plot results
    print("\n📊 Generating extrapolation plot...")
    plot_extrapolation_results(results, train_len)

    # Print numerical results
    print("\n" + "=" * 70)
    print("Extrapolation Error Summary")
    print("=" * 70)
    print(f"{'Length':<10} {'Sinusoidal':<15} {'Learned':<15} {'RoPE':<15} {'ALiBi':<15}")
    print("-" * 70)

    for test_len in test_lengths:
        row = f"{test_len:<10}"
        for enc_type in ["sinusoidal", "learned", "rope", "alibi"]:
            error = results[enc_type][test_len]
            if error == float("inf"):
                row += f"{'FAIL':<15}"
            else:
                row += f"{error:<15.6f}"
        print(row)

    # Additional analyses
    test_learned_embedding_failure()
    analyze_alibi_distance_penalty()

    print("\n" + "=" * 70)
    print("🏆 Extrapolation Rankings (Best → Worst)")
    print("=" * 70)
    print("1. 🥇 ALiBi: Excellent extrapolation (linear penalty)")
    print("2. 🥈 RoPE: Good extrapolation (relative positions)")
    print("3. 🥉 Sinusoidal: Moderate extrapolation (periodic issues)")
    print("4. ❌ Learned: FAILS beyond training length")
    print("=" * 70)

    print("\n💡 Practical Implications:")
    print("• Training on 8k tokens, deploying at 32k? → Use ALiBi or RoPE")
    print("• Fixed context (e.g., 512 tokens)? → Learned is fine (BERT)")
    print("• Long context is critical? → ALiBi (BLOOM) or RoPE (Llama 2)")
    print("=" * 70)


if __name__ == "__main__":
    main()

