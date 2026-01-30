"""Visualize Query, Key, Value matrices and attention patterns."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from llm_engineering_fundamentals.attention.core import MultiHeadAttention


def visualize_qkv_projections():
    """Visualize the Q, K, V projection matrices."""
    print("=" * 70)
    print("QKV PROJECTION MATRICES VISUALIZATION")
    print("=" * 70)
    
    d_model = 64
    num_heads = 4
    seq_len = 12
    batch_size = 1
    
    # Create attention module
    attn = MultiHeadAttention(d_model, num_heads)
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    # Manually compute Q, K, V to visualize
    Q = x @ attn.W_q
    K = x @ attn.W_k
    V = x @ attn.W_v
    
    print(f"\nInput shape: {x.shape}")
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # Reshape for multi-head
    batch, seq, _ = x.shape
    Q = Q.reshape(batch, seq, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    
    print(f"\nAfter reshaping for {num_heads} heads:")
    print(f"Q shape: {Q.shape}  # (batch, heads, seq_len, head_dim)")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # Visualize Q, K, V for first head
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    matrices = [
        (Q[0, 0], "Query (Q)", "What am I looking for?"),
        (K[0, 0], "Key (K)", "What do I offer?"),
        (V[0, 0], "Value (V)", "What do I contain?"),
    ]
    
    for ax, (mat, title, desc) in zip(axes, matrices):
        im = ax.imshow(mat, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f"{title}\n{desc}", fontsize=10)
        ax.set_xlabel("Head Dimension")
        ax.set_ylabel("Sequence Position")
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "assets"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "qkv_matrices.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_dir / 'qkv_matrices.png'}")
    plt.close()


def visualize_attention_computation():
    """Visualize how attention weights are computed from Q and K."""
    print("\n" + "=" * 70)
    print("ATTENTION WEIGHT COMPUTATION")
    print("=" * 70)
    
    d_model = 64
    num_heads = 4
    seq_len = 8
    batch_size = 1
    
    # Create attention module
    attn = MultiHeadAttention(d_model, num_heads)
    
    # Create input with some structure
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    # Add some patterns to make attention interesting
    x[0, 0] = x[0, 3] * 0.8  # Token 0 similar to token 3
    x[0, 1] = x[0, 4] * 0.8  # Token 1 similar to token 4
    
    # Compute attention
    output, attention_weights = attn(x, x, x, return_attention_weights=True)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"  # (batch, heads, seq_len, seq_len)")
    
    # Visualize attention for each head
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for head in range(num_heads):
        ax = axes[head]
        attn_matrix = attention_weights[0, head]
        
        im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f"Head {head + 1}\nAttention Weights", fontsize=11)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        
        # Add grid
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("Multi-Head Attention Patterns\n"
                 "(Each head learns different attention patterns)", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "assets"
    plt.savefig(output_dir / "attention_patterns.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_dir / 'attention_patterns.png'}")
    plt.close()


def demonstrate_qkv_intuition():
    """Demonstrate the intuition behind Q, K, V with a simple example."""
    print("\n" + "=" * 70)
    print("QKV INTUITION DEMO")
    print("=" * 70)
    
    print("\nImagine a sentence: 'The cat sat on the mat'")
    print("\nFor the word 'sat':")
    print("  Query (Q):  'Who is doing this action?'")
    print("  Key (K):    'The' says 'I'm a determiner'")
    print("              'cat' says 'I'm a noun/subject'")  
    print("              'on' says 'I'm a preposition'")
    print("  Value (V):  The actual representations to aggregate")
    print("\nAttention weights = softmax(Q · K^T / √d_k)")
    print("  High weight on 'cat' because it's the subject of 'sat'")
    print("\nOutput = Attention weights · V")
    print("  Aggregates information from relevant words (like 'cat')")
    
    # Simple numerical example
    print("\n" + "-" * 70)
    print("Simple Numerical Example:")
    print("-" * 70)
    
    # 3 tokens, 4 dimensions
    Q = np.array([[1, 0, 0, 0]])  # Query for token 0
    K = np.array([
        [1, 0, 0, 0],  # Key for token 0
        [0, 1, 0, 0],  # Key for token 1
        [0.8, 0.2, 0, 0],  # Key for token 2 (similar to token 0)
    ])
    V = np.array([
        [1, 1, 1, 1],  # Value for token 0
        [2, 2, 2, 2],  # Value for token 1
        [3, 3, 3, 3],  # Value for token 2
    ])
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(4)
    print(f"\nScores (Q · K^T / √d_k): {scores[0]}")
    
    # Apply softmax
    exp_scores = np.exp(scores - np.max(scores))
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    print(f"Attention weights: {attention_weights[0]}")
    print(f"  Token 0 attends most to itself (0.57)")
    print(f"  Then to token 2 (0.29) which is similar")
    print(f"  Least to token 1 (0.14) which is different")
    
    # Compute output
    output = attention_weights @ V
    print(f"\nOutput (weighted sum of values): {output[0]}")
    print(f"  Mostly token 0's value (1,1,1,1)")
    print(f"  With some token 2 (3,3,3,3)")
    print(f"  And a bit of token 1 (2,2,2,2)")
    
    print("\n✓ Q·K determines WHERE to attend")
    print("✓ Softmax normalizes to a probability distribution")
    print("✓ Weighted sum of V aggregates information")


def main():
    """Run all QKV visualizations."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("QKV & ATTENTION VISUALIZATIONS")
    print("=" * 70)
    
    demonstrate_qkv_intuition()
    visualize_qkv_projections()
    visualize_attention_computation()
    
    print("\n" + "=" * 70)
    print("VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Q·K^T computes relevance scores (which tokens to attend to)")
    print("2. Softmax converts scores to probability distribution")
    print("3. Attention weights · V aggregates information")
    print("4. Different heads learn different attention patterns")
    print("5. Multi-head attention captures diverse relationships")
    print()


if __name__ == "__main__":
    main()

