"""Demo script for transformer blocks."""

import numpy as np
from llm_engineering_fundamentals.transformer.block import (
    TransformerBlock,
    StackedTransformer,
    LayerNorm,
    FeedForward,
)


def demo_layer_norm():
    """Demonstrate layer normalization."""
    print("=" * 70)
    print("LAYER NORMALIZATION DEMO")
    print("=" * 70)
    
    d_model = 8
    ln = LayerNorm(d_model)
    
    # Create input with different scales per feature
    x = np.random.randn(2, 4, d_model) * np.array([0.1, 1.0, 10.0, 0.5, 2.0, 0.8, 5.0, 0.3])
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input mean per sample: {np.mean(x, axis=-1)}")
    print(f"Input std per sample: {np.std(x, axis=-1)}")
    
    # Apply layer norm
    x_norm = ln(x)
    
    print(f"\nNormalized mean per sample: {np.mean(x_norm, axis=-1)}")
    print(f"Normalized std per sample: {np.std(x_norm, axis=-1)}")
    print("\n✓ After LayerNorm, each token has mean ≈ 0 and std ≈ 1")


def demo_feed_forward():
    """Demonstrate feed-forward network."""
    print("\n" + "=" * 70)
    print("FEED-FORWARD NETWORK DEMO")
    print("=" * 70)
    
    d_model = 64
    d_ff = 256  # 4x expansion
    ffn = FeedForward(d_model, d_ff)
    
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print(f"\nInput shape: {x.shape}")
    print(f"Hidden dimension (d_ff): {d_ff}")
    print(f"Expansion factor: {d_ff / d_model:.1f}x")
    
    output = ffn(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Parameters: {d_model * d_ff + d_ff + d_ff * d_model + d_model:,}")
    print("\n✓ FFN expands to 4x then projects back to d_model")


def demo_transformer_block():
    """Demonstrate a single transformer block."""
    print("\n" + "=" * 70)
    print("TRANSFORMER BLOCK DEMO")
    print("=" * 70)
    
    d_model = 128
    num_heads = 8
    d_ff = 512
    
    # Pre-LN (modern)
    block_preln = TransformerBlock(d_model, num_heads, d_ff, norm_first=True)
    
    # Post-LN (original)
    block_postln = TransformerBlock(d_model, num_heads, d_ff, norm_first=False)
    
    batch_size = 2
    seq_len = 16
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print(f"\nInput shape: {x.shape}")
    print(f"Model dimension (d_model): {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"FFN dimension (d_ff): {d_ff}")
    
    # Test both variants
    output_preln = block_preln(x)
    output_postln = block_postln(x)
    
    print(f"\nPre-LN output shape: {output_preln.shape}")
    print(f"Post-LN output shape: {output_postln.shape}")
    
    print("\n✓ Pre-LN: LayerNorm before each sublayer (modern, more stable)")
    print("✓ Post-LN: LayerNorm after each sublayer (original, harder to train)")


def demo_stacked_transformer():
    """Demonstrate stacked transformer blocks."""
    print("\n" + "=" * 70)
    print("STACKED TRANSFORMER DEMO")
    print("=" * 70)
    
    # Small GPT-like configuration
    num_layers = 6
    d_model = 256
    num_heads = 8
    d_ff = 1024  # 4x d_model
    
    model = StackedTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_first=True,
    )
    
    batch_size = 2
    seq_len = 32
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  d_model: {d_model}")
    print(f"  Heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    
    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameter Counts:")
    print(f"  Attention per block: {param_counts['attention_per_block']:,}")
    print(f"  FFN per block: {param_counts['ffn_per_block']:,}")
    print(f"  LayerNorm per block: {param_counts['ln_per_block']:,}")
    print(f"  Total per block: {param_counts['total_per_block']:,}")
    print(f"  Total parameters: {param_counts['total_parameters']:,}")
    
    # Forward pass
    print(f"\nForward Pass:")
    print(f"  Input shape: {x.shape}")
    
    output = model(x)
    print(f"  Output shape: {output.shape}")
    
    # Get outputs from all layers
    all_outputs = model(x, return_all_layers=True)
    print(f"  Number of layer outputs: {len(all_outputs)}")
    
    # Analyze how representations change through layers
    print(f"\nRepresentation Analysis:")
    input_norm = np.linalg.norm(x)
    for i, layer_out in enumerate(all_outputs):
        output_norm = np.linalg.norm(layer_out)
        change = np.linalg.norm(layer_out - x)
        print(f"  Layer {i+1}: norm={output_norm:.2f}, change_from_input={change:.2f}")
    
    print("\n✓ Deep transformer with 6 layers successfully processes sequence")


def compare_depth_vs_width():
    """Compare different depth and width configurations."""
    print("\n" + "=" * 70)
    print("DEPTH VS WIDTH COMPARISON")
    print("=" * 70)
    
    batch_size = 1
    seq_len = 16
    
    configs = [
        {"name": "Shallow-Wide", "num_layers": 3, "d_model": 512, "num_heads": 8, "d_ff": 2048},
        {"name": "Deep-Narrow", "num_layers": 12, "d_model": 256, "num_heads": 8, "d_ff": 1024},
        {"name": "Balanced", "num_layers": 6, "d_model": 384, "num_heads": 6, "d_ff": 1536},
    ]
    
    print("\nComparing configurations with similar parameter counts:\n")
    
    for config in configs:
        model = StackedTransformer(
            num_layers=config["num_layers"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
        )
        
        param_counts = model.count_parameters()
        x = np.random.randn(batch_size, seq_len, config["d_model"]) * 0.1
        output = model(x)
        
        print(f"{config['name']:15} | Layers: {config['num_layers']:2} | "
              f"d_model: {config['d_model']:3} | Params: {param_counts['total_parameters']:,}")
    
    print("\n✓ Different architectures can have similar parameter counts")
    print("✓ Depth enables more compositional reasoning")
    print("✓ Width provides more representational capacity per layer")


def main():
    """Run all demos."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("TRANSFORMER BLOCK DEMONSTRATIONS")
    print("=" * 70)
    
    demo_layer_norm()
    demo_feed_forward()
    demo_transformer_block()
    demo_stacked_transformer()
    compare_depth_vs_width()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. LayerNorm stabilizes training by normalizing activations")
    print("2. FFN typically uses 4x expansion for more capacity")
    print("3. Pre-LN (modern) is more stable than Post-LN (original)")
    print("4. Residual connections enable gradient flow in deep networks")
    print("5. Depth vs width trade-offs affect model capabilities")
    print()


if __name__ == "__main__":
    main()


