"""Demo script for dropout variants."""

import numpy as np

from llm_engineering_fundamentals.regularization.dropout import (
    Dropout,
    AttentionDropout,
    DropPath,
    EmbeddingDropout,
)


def demo_standard_dropout():
    """Demonstrate standard dropout."""
    print("=" * 70)
    print("STANDARD DROPOUT")
    print("=" * 70)
    
    dropout = Dropout(p=0.5)  # Drop 50% of elements
    
    x = np.ones((2, 4, 8))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Dropout rate: {dropout.p}")
    print(f"Keep probability: {dropout.keep_prob}")
    
    # Apply dropout in training mode
    output_train = dropout(x, training=True)
    
    print(f"\nWith dropout (training):")
    print(f"  Mean (expected ~1.0): {np.mean(output_train):.3f}")
    print(f"  Zeros: {np.sum(output_train == 0) / output_train.size * 100:.1f}%")
    print(f"  Non-zeros: {np.sum(output_train != 0) / output_train.size * 100:.1f}%")
    
    # Apply dropout in inference mode
    output_infer = dropout(x, training=False)
    
    print(f"\nWithout dropout (inference):")
    print(f"  Mean: {np.mean(output_infer):.3f}")
    print(f"  All values preserved: {np.allclose(output_infer, x)}")
    
    print("\n✓ Dropout drops ~50% of activations during training")
    print("✓ Scales remaining by 1/keep_prob to maintain expectation")
    print("✓ No dropout during inference")


def demo_attention_dropout():
    """Demonstrate attention dropout."""
    print("\n" + "=" * 70)
    print("ATTENTION DROPOUT")
    print("=" * 70)
    
    attn_dropout = AttentionDropout(p=0.1)
    
    # Simulate attention weights (already softmax'd)
    np.random.seed(42)
    seq_len = 8
    attention_weights = np.random.rand(1, 4, seq_len, seq_len)  # (batch, heads, seq, seq)
    
    # Normalize to sum to 1 (like real attention weights)
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    print(f"\nAttention weights shape: {attention_weights.shape}")
    print(f"Sum per query (should be ~1.0): {np.sum(attention_weights[0, 0, 0]):.3f}")
    
    # Apply attention dropout
    dropped_weights = attn_dropout(attention_weights, training=True)
    
    print(f"\nAfter attention dropout:")
    print(f"  Sum per query: {np.sum(dropped_weights[0, 0, 0]):.3f}")
    print(f"  Zeros: {np.sum(dropped_weights == 0) / dropped_weights.size * 100:.1f}%")
    
    print("\n✓ Attention dropout applied after softmax")
    print("✓ Prevents over-reliance on specific tokens")


def demo_droppath():
    """Demonstrate DropPath (Stochastic Depth)."""
    print("\n" + "=" * 70)
    print("DROPPATH (STOCHASTIC DEPTH)")
    print("=" * 70)
    
    drop_prob = 0.3
    droppath = DropPath(drop_prob=drop_prob)
    
    # Simulate residual from transformer block
    batch_size = 4
    seq_len = 16
    d_model = 128
    residual = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print(f"\nResidual shape: {residual.shape}")
    print(f"Drop probability: {drop_prob}")
    
    # Apply droppath multiple times
    num_trials = 100
    num_dropped = 0
    
    for _ in range(num_trials):
        output = droppath(residual, training=True)
        # Check if entire batch was dropped
        if np.allclose(output, 0):
            num_dropped += 1
    
    print(f"\nOver {num_trials} trials:")
    print(f"  Paths dropped: {num_dropped} ({num_dropped/num_trials*100:.1f}%)")
    print(f"  Expected: ~{drop_prob*100:.1f}%")
    
    # Show linear schedule
    num_layers = 12
    final_drop_prob = 0.2
    schedule = DropPath.get_drop_prob_schedule(num_layers, final_drop_prob)
    
    print(f"\nLinear DropPath schedule ({num_layers} layers):")
    for i, prob in enumerate(schedule):
        print(f"  Layer {i+1:2}: {prob:.3f}")
    
    print("\n✓ DropPath drops entire residual connections")
    print("✓ More aggressive than standard dropout")
    print("✓ Deeper layers typically have higher drop probability")


def demo_embedding_dropout():
    """Demonstrate embedding dropout."""
    print("\n" + "=" * 70)
    print("EMBEDDING DROPOUT")
    print("=" * 70)
    
    # Element-wise dropout
    emb_dropout_elem = EmbeddingDropout(p=0.1, drop_entire_tokens=False)
    
    # Token-wise dropout
    emb_dropout_token = EmbeddingDropout(p=0.1, drop_entire_tokens=True)
    
    embeddings = np.random.randn(2, 8, 64) * 0.1  # (batch, seq_len, d_model)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    # Element-wise
    output_elem = emb_dropout_elem(embeddings, training=True)
    zeros_elem = np.sum(output_elem == 0) / output_elem.size * 100
    
    print(f"\nElement-wise dropout:")
    print(f"  Zeros: {zeros_elem:.1f}%")
    print(f"  Drops individual dimensions")
    
    # Token-wise
    output_token = emb_dropout_token(embeddings, training=True)
    
    # Count how many tokens are fully dropped
    fully_dropped_tokens = 0
    for b in range(embeddings.shape[0]):
        for t in range(embeddings.shape[1]):
            if np.allclose(output_token[b, t], 0):
                fully_dropped_tokens += 1
    
    total_tokens = embeddings.shape[0] * embeddings.shape[1]
    print(f"\nToken-wise dropout:")
    print(f"  Fully dropped tokens: {fully_dropped_tokens}/{total_tokens} ({fully_dropped_tokens/total_tokens*100:.1f}%)")
    print(f"  Drops entire token embeddings")
    
    print("\n✓ Element-wise: Drop individual embedding dimensions")
    print("✓ Token-wise: Drop entire token embeddings")


def compare_dropout_rates():
    """Compare different dropout rates."""
    print("\n" + "=" * 70)
    print("DROPOUT RATE COMPARISON")
    print("=" * 70)
    
    rates = [0.0, 0.1, 0.2, 0.5]
    x = np.ones((100, 1024))
    
    print(f"\nInput: {x.shape}, all ones")
    print(f"\n{'Rate':>6} | {'Mean':>8} | {'Std':>8} | {'Zeros':>8}")
    print("-" * 40)
    
    for rate in rates:
        dropout = Dropout(p=rate)
        output = dropout(x, training=True)
        
        mean = np.mean(output)
        std = np.std(output)
        zeros = np.sum(output == 0) / output.size * 100
        
        print(f"{rate:>6.1f} | {mean:>8.3f} | {std:>8.3f} | {zeros:>7.1f}%")
    
    print("\n✓ Higher dropout = more regularization but higher variance")
    print("✓ Typical range: 0.1-0.2 for transformers")
    print("✓ Modern LLMs often use less dropout (0.0-0.1)")


def main():
    """Run all demos."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("DROPOUT DEMONSTRATIONS")
    print("=" * 70)
    
    demo_standard_dropout()
    demo_attention_dropout()
    demo_droppath()
    demo_embedding_dropout()
    compare_dropout_rates()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Dropout prevents overfitting by randomly zeroing activations")
    print("2. Attention dropout applied to attention weights")
    print("3. DropPath drops entire residual connections")
    print("4. Embedding dropout can drop elements or entire tokens")
    print("5. Typical rates: 0.1-0.2 for transformers")
    print()


if __name__ == "__main__":
    main()

