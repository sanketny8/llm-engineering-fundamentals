"""Demo script for embedding layers."""

import numpy as np

from llm_engineering_fundamentals.embeddings.layers import (
    TokenEmbedding,
    CombinedEmbedding,
    TiedEmbedding,
    compare_tied_vs_untied,
)


def demo_token_embedding():
    """Demonstrate token embedding."""
    print("=" * 70)
    print("TOKEN EMBEDDING")
    print("=" * 70)
    
    vocab_size = 1000
    d_model = 128
    
    token_emb = TokenEmbedding(vocab_size, d_model)
    
    # Sample token IDs
    token_ids = np.array([[5, 42, 100, 7],
                          [3, 99, 42, 8]])
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {d_model}")
    print(f"  Parameters: {token_emb.count_parameters():,}")
    
    print(f"\nToken IDs shape: {token_ids.shape}")
    
    # Lookup embeddings
    embeddings = token_emb(token_ids)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"  # (batch_size, seq_len, d_model)")
    
    print("\n✓ Token embedding is a simple lookup table")
    print("✓ Each token ID maps to a dense vector")


def demo_combined_embedding():
    """Demonstrate combined token + positional embedding."""
    print("\n" + "=" * 70)
    print("COMBINED EMBEDDING (Token + Positional)")
    print("=" * 70)
    
    vocab_size = 5000
    d_model = 256
    max_seq_len = 512
    
    # Test both sinusoidal and learned
    for pos_type in ["sinusoidal", "learned"]:
        print(f"\n{pos_type.upper()} Positional Embeddings:")
        print("-" * 70)
        
        combined_emb = CombinedEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            positional_type=pos_type,
            dropout=0.1,
            scale_embeddings=False,
        )
        
        # Sample input
        token_ids = np.array([[10, 25, 100, 42, 7]])
        
        # Get embeddings
        embeddings = combined_emb(token_ids, training=False)
        
        params = combined_emb.count_parameters()
        
        print(f"  Token IDs: {token_ids[0]}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Token embedding params: {params['token_embedding']:,}")
        print(f"  Positional embedding params: {params['positional_embedding']:,}")
        print(f"  Total params: {params['total']:,}")
    
    print("\n✓ Combined embedding = Token + Positional")
    print("✓ Sinusoidal: No parameters (fixed)")
    print("✓ Learned: max_seq_len × d_model parameters")


def demo_scaling():
    """Demonstrate embedding scaling."""
    print("\n" + "=" * 70)
    print("EMBEDDING SCALING (Original Transformer)")
    print("=" * 70)
    
    vocab_size = 1000
    d_model = 512
    
    # Without scaling
    emb_no_scale = CombinedEmbedding(
        vocab_size, d_model, scale_embeddings=False
    )
    
    # With scaling
    emb_with_scale = CombinedEmbedding(
        vocab_size, d_model, scale_embeddings=True
    )
    
    token_ids = np.array([[1, 2, 3]])
    
    out_no_scale = emb_no_scale(token_ids)
    out_with_scale = emb_with_scale(token_ids)
    
    print(f"\nd_model: {d_model}")
    print(f"Scaling factor: √{d_model} = {np.sqrt(d_model):.2f}")
    
    print(f"\nWithout scaling:")
    print(f"  Mean magnitude: {np.mean(np.abs(out_no_scale)):.4f}")
    
    print(f"\nWith scaling:")
    print(f"  Mean magnitude: {np.mean(np.abs(out_with_scale)):.4f}")
    print(f"  Ratio: {np.mean(np.abs(out_with_scale)) / np.mean(np.abs(out_no_scale)):.2f}x")
    
    print("\n✓ Original Transformer scales by √d_model")
    print("✓ Helps balance token and positional embeddings")
    print("✓ Modern LLMs often skip this (use LayerNorm instead)")


def demo_tied_embedding():
    """Demonstrate tied embeddings."""
    print("\n" + "=" * 70)
    print("TIED EMBEDDINGS (Input + Output Weight Sharing)")
    print("=" * 70)
    
    vocab_size = 50000  # GPT-2 size
    d_model = 768
    max_seq_len = 1024
    
    tied_emb = TiedEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        positional_type="learned",
    )
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary: {vocab_size:,}")
    print(f"  d_model: {d_model}")
    print(f"  Max sequence length: {max_seq_len}")
    
    # Input embedding
    token_ids = np.array([[1, 2, 3, 4, 5]])
    embeddings = tied_emb.embed(token_ids)
    
    print(f"\nInput embedding:")
    print(f"  Token IDs: {token_ids[0]}")
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Simulate transformer output (same shape as embeddings)
    hidden_states = np.random.randn(1, 5, d_model) * 0.1
    
    # Output projection
    logits = tied_emb.project_to_vocab(hidden_states)
    
    print(f"\nOutput projection:")
    print(f"  Hidden states shape: {hidden_states.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  # (batch, seq_len, vocab_size)")
    
    params = tied_emb.count_parameters()
    print(f"\nParameters:")
    print(f"  Token embedding: {params['token_embedding']:,}")
    print(f"  Positional embedding: {params['positional_embedding']:,}")
    print(f"  Total: {params['total']:,}")
    
    print("\n✓ Input embedding and output projection share weights")
    print("✓ Saves vocab_size × d_model parameters")
    print("✓ Used in GPT-2, BERT, T5, LLaMA")


def compare_parameter_counts():
    """Compare tied vs untied parameter counts."""
    print("\n" + "=" * 70)
    print("TIED VS UNTIED: PARAMETER COMPARISON")
    print("=" * 70)
    
    configs = [
        {"name": "GPT-2 Small", "vocab": 50257, "d_model": 768, "max_seq": 1024},
        {"name": "BERT Base", "vocab": 30522, "d_model": 768, "max_seq": 512},
        {"name": "GPT-2 Large", "vocab": 50257, "d_model": 1280, "max_seq": 1024},
    ]
    
    print(f"\n{'Model':15} | {'Tied':>12} | {'Untied':>12} | {'Savings':>12}")
    print("-" * 60)
    
    for config in configs:
        comparison = compare_tied_vs_untied(
            config["vocab"], config["d_model"], config["max_seq"]
        )
        
        print(f"{config['name']:15} | {comparison['tied_parameters']:>12,} | "
              f"{comparison['untied_parameters']:>12,} | "
              f"{comparison['savings']:>12,}")
    
    # Detailed breakdown for GPT-2 Small
    print("\nDetailed breakdown (GPT-2 Small):")
    print("-" * 70)
    comparison = compare_tied_vs_untied(50257, 768, 1024)
    
    print(f"Token embedding:       {50257 * 768:>12,} parameters")
    print(f"Positional embedding:  {1024 * 768:>12,} parameters")
    print(f"Output projection:     {50257 * 768:>12,} parameters (if untied)")
    print(f"\nTied total:            {comparison['tied_parameters']:>12,} parameters")
    print(f"Untied total:          {comparison['untied_parameters']:>12,} parameters")
    print(f"\nSavings:               {comparison['savings']:>12,} parameters")
    print(f"Savings:               {comparison['savings_percentage']:>11.1f}%")
    
    print("\n✓ Tied embeddings save ~33% of embedding parameters")
    print("✓ For large vocab, this is millions of parameters")


def main():
    """Run all demos."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("EMBEDDING LAYER DEMONSTRATIONS")
    print("=" * 70)
    
    demo_token_embedding()
    demo_combined_embedding()
    demo_scaling()
    demo_tied_embedding()
    compare_parameter_counts()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Token embeddings map discrete IDs to dense vectors")
    print("2. Positional embeddings add position information")
    print("3. Combined embedding = Token + Positional")
    print("4. Tied embeddings share input/output weights (saves params)")
    print("5. Embedding layer is often largest component in smaller models")
    print()


if __name__ == "__main__":
    main()

