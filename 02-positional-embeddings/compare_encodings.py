#!/usr/bin/env python3
"""
Compare all 4 positional encoding types with visualizations and analysis.
"""
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from llm_engineering_fundamentals.positional.encodings import (
    SinusoidalPositionEncoding,
    LearnedPositionEmbedding,
    RotaryPositionEmbedding,
    ALiBiAttentionBias,
)


def plot_encoding_heatmaps(seq_len: int = 128, d_model: int = 64):
    """Plot heatmaps of all 4 positional encoding types."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Positional Encodings Comparison (seq_len={seq_len}, d_model={d_model})", fontsize=16)

    # 1. Sinusoidal
    sin_enc = SinusoidalPositionEncoding(d_model, max_len=seq_len)
    sin_pe = sin_enc(seq_len)

    ax = axes[0, 0]
    im = ax.imshow(sin_pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Sinusoidal (Classic Transformer)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Position")
    plt.colorbar(im, ax=ax)

    # 2. Learned
    learned_enc = LearnedPositionEmbedding(max_len=seq_len, d_model=d_model, seed=42)
    learned_pe = learned_enc(seq_len)

    ax = axes[0, 1]
    im = ax.imshow(learned_pe, aspect="auto", cmap="RdBu_r")
    ax.set_title("Learned (BERT-style)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Position")
    plt.colorbar(im, ax=ax)

    # 3. RoPE (visualize rotation effect)
    rope_enc = RotaryPositionEmbedding(dim=d_model, max_len=seq_len)
    # Apply to dummy input to show effect
    dummy_input = np.random.randn(seq_len, d_model) * 0.02
    rotated = rope_enc.rotate(dummy_input, seq_len)

    ax = axes[1, 0]
    im = ax.imshow(rotated, aspect="auto", cmap="RdBu_r")
    ax.set_title("RoPE (Llama 2) - Rotation Effect")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Position")
    plt.colorbar(im, ax=ax)

    # 4. ALiBi (visualize attention bias for first head)
    alibi_enc = ALiBiAttentionBias(num_heads=8, max_len=seq_len)
    alibi_bias = alibi_enc(seq_len)[0]  # First head

    ax = axes[1, 1]
    im = ax.imshow(alibi_bias, aspect="auto", cmap="RdBu_r")
    ax.set_title("ALiBi (BLOOM) - Attention Bias (Head 0)")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("02-positional-embeddings/assets/encoding_comparison.png", dpi=150)
    print("✅ Saved heatmaps to: 02-positional-embeddings/assets/encoding_comparison.png")
    plt.show()


def plot_similarity_matrices(seq_len: int = 64, d_model: int = 128):
    """Plot similarity matrices (how similar are nearby positions?)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Position Similarity Matrices (seq_len={seq_len})", fontsize=16)

    # 1. Sinusoidal
    sin_enc = SinusoidalPositionEncoding(d_model, max_len=seq_len)
    sin_pe = sin_enc(seq_len)
    sin_sim = np.dot(sin_pe, sin_pe.T) / d_model  # Normalized dot product

    ax = axes[0]
    im = ax.imshow(sin_sim, aspect="auto", cmap="viridis")
    ax.set_title("Sinusoidal Similarity")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    plt.colorbar(im, ax=ax)

    # 2. Learned
    learned_enc = LearnedPositionEmbedding(max_len=seq_len, d_model=d_model, seed=42)
    learned_pe = learned_enc(seq_len)
    learned_sim = np.dot(learned_pe, learned_pe.T) / d_model

    ax = axes[1]
    im = ax.imshow(learned_sim, aspect="auto", cmap="viridis")
    ax.set_title("Learned Similarity")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    plt.colorbar(im, ax=ax)

    # 3. ALiBi (attention bias similarity)
    alibi_enc = ALiBiAttentionBias(num_heads=8, max_len=seq_len)
    alibi_bias = alibi_enc(seq_len)[0]

    ax = axes[2]
    im = ax.imshow(alibi_bias, aspect="auto", cmap="RdBu_r")
    ax.set_title("ALiBi Bias (closer = more positive)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("02-positional-embeddings/assets/similarity_matrices.png", dpi=150)
    print("✅ Saved similarity matrices to: 02-positional-embeddings/assets/similarity_matrices.png")
    plt.show()


def plot_frequency_spectrum(seq_len: int = 512, d_model: int = 512):
    """Analyze frequency spectrum of sinusoidal encoding."""
    sin_enc = SinusoidalPositionEncoding(d_model, max_len=seq_len)
    sin_pe = sin_enc(seq_len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot first few dimensions over positions
    ax = axes[0]
    for dim in [0, 10, 50, 100, 200]:
        if dim < d_model:
            ax.plot(sin_pe[:, dim], label=f"dim {dim}", alpha=0.7)
    ax.set_xlabel("Position")
    ax.set_ylabel("Encoding Value")
    ax.set_title("Sinusoidal Encoding: Different Dimensions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot encoding at a specific position across dimensions
    ax = axes[1]
    position_idx = seq_len // 2
    ax.plot(sin_pe[position_idx, :])
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Encoding Value")
    ax.set_title(f"Sinusoidal Encoding at Position {position_idx}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("02-positional-embeddings/assets/frequency_spectrum.png", dpi=150)
    print("✅ Saved frequency spectrum to: 02-positional-embeddings/assets/frequency_spectrum.png")
    plt.show()


def analyze_alibi_slopes():
    """Analyze ALiBi slopes for different head counts."""
    print("\n" + "=" * 70)
    print("ALiBi Slopes Analysis")
    print("=" * 70)

    for num_heads in [4, 8, 12, 16, 32]:
        alibi_enc = ALiBiAttentionBias(num_heads=num_heads)
        slopes = alibi_enc.slopes

        print(f"\nnum_heads={num_heads}:")
        print(f"  Slopes: {slopes}")
        print(f"  Min: {slopes.min():.6f}, Max: {slopes.max():.6f}")
        print(f"  Range: {slopes.max() / slopes.min():.2f}x")


def compare_memory_footprint(seq_len: int = 2048, d_model: int = 768):
    """Compare memory requirements of different encodings."""
    print("\n" + "=" * 70)
    print(f"Memory Footprint Comparison (seq_len={seq_len}, d_model={d_model})")
    print("=" * 70)

    # Sinusoidal: Just cache
    sin_memory = seq_len * d_model * 4  # 4 bytes per float32
    print(f"\n1. Sinusoidal:")
    print(f"   Cache: {sin_memory / 1024:.1f} KB")
    print(f"   Runtime: No extra memory (deterministic)")

    # Learned: Parameters
    learned_memory = seq_len * d_model * 4
    print(f"\n2. Learned:")
    print(f"   Parameters: {learned_memory / 1024:.1f} KB")
    print(f"   ⚠️  Cannot extrapolate beyond {seq_len} tokens!")

    # RoPE: Just cache cos/sin
    rope_memory = 2 * seq_len * d_model * 4  # cos + sin
    print(f"\n3. RoPE:")
    print(f"   Cache: {rope_memory / 1024:.1f} KB")
    print(f"   Runtime: Rotation computation (cheap)")

    # ALiBi: No positional embeddings!
    num_heads = 12
    alibi_memory = num_heads * seq_len * seq_len * 4  # Full attention bias
    print(f"\n4. ALiBi:")
    print(f"   Attention Bias: {alibi_memory / (1024**2):.2f} MB (for {num_heads} heads)")
    print(f"   Positional Embeddings: 0 KB (none needed!)")
    print(f"   Note: Bias is computed on-the-fly or cached")

    print("\n" + "=" * 70)
    print("Winner: ALiBi (no positional embeddings at all!)")
    print("Runner-up: Sinusoidal/RoPE (deterministic, can extrapolate)")
    print("=" * 70)


def main():
    """Run all comparisons and analyses."""
    print("=" * 70)
    print("Positional Encodings Comparison")
    print("=" * 70)

    # Create visualization directory
    import os
    os.makedirs("02-positional-embeddings/assets", exist_ok=True)

    # 1. Heatmaps
    print("\n📊 Generating encoding heatmaps...")
    plot_encoding_heatmaps(seq_len=128, d_model=64)

    # 2. Similarity matrices
    print("\n📊 Generating similarity matrices...")
    plot_similarity_matrices(seq_len=64, d_model=128)

    # 3. Frequency spectrum (sinusoidal)
    print("\n📊 Analyzing frequency spectrum...")
    plot_frequency_spectrum(seq_len=512, d_model=512)

    # 4. ALiBi slopes
    analyze_alibi_slopes()

    # 5. Memory comparison
    compare_memory_footprint(seq_len=2048, d_model=768)

    print("\n" + "=" * 70)
    print("🎉 Analysis Complete!")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("1. Sinusoidal: Periodic structure, works for any length")
    print("2. Learned: Unstructured, best for fixed-length training")
    print("3. RoPE: Rotational, excellent for relative positions (Llama 2)")
    print("4. ALiBi: No embeddings needed, best extrapolation (BLOOM)")
    print("=" * 70)


if __name__ == "__main__":
    main()

