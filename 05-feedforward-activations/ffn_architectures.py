"""Demo script comparing FFN architectures."""

import numpy as np

from llm_engineering_fundamentals.ffn.networks import (
    FeedForwardNetwork,
    SwiGLUFFN,
    GeGLUFFN,
    compare_ffn_architectures,
)
from llm_engineering_fundamentals.ffn.activations import relu, gelu


def demo_standard_ffn():
    """Demonstrate standard FFN."""
    print("=" * 70)
    print("STANDARD FEED-FORWARD NETWORK")
    print("=" * 70)
    
    d_model = 256
    d_ff = 1024  # 4x expansion
    
    # Create FFN with GELU
    ffn = FeedForwardNetwork(d_model, d_ff, activation=gelu)
    
    # Create input
    batch_size = 2
    seq_len = 8
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff} (expansion: {d_ff/d_model:.1f}x)")
    print(f"  Activation: GELU")
    
    # Forward pass
    output = ffn(x, training=False)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {ffn.count_parameters():,}")
    
    # Parameter breakdown
    w1_params = d_model * d_ff
    w2_params = d_ff * d_model
    bias_params = d_ff + d_model
    
    print(f"\nParameter breakdown:")
    print(f"  W1: {w1_params:,} ({w1_params/ffn.count_parameters()*100:.1f}%)")
    print(f"  W2: {w2_params:,} ({w2_params/ffn.count_parameters()*100:.1f}%)")
    print(f"  Biases: {bias_params:,} ({bias_params/ffn.count_parameters()*100:.1f}%)")
    
    print("\n✓ Standard FFN: 2 weight matrices + activation")


def demo_gated_ffn():
    """Demonstrate gated FFN variants."""
    print("\n" + "=" * 70)
    print("GATED FEED-FORWARD NETWORKS")
    print("=" * 70)
    
    d_model = 256
    d_ff = 1024
    
    # Create both variants
    swiglu_ffn = SwiGLUFFN(d_model, d_ff)
    geglu_ffn = GeGLUFFN(d_model, d_ff)
    
    # Create input
    x = np.random.randn(1, 8, d_model) * 0.1
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    
    # SwiGLU
    output_swiglu = swiglu_ffn(x, training=False)
    params_swiglu = swiglu_ffn.count_parameters()
    
    print(f"\nSwiGLU (LLaMA, PaLM):")
    print(f"  Output shape: {output_swiglu.shape}")
    print(f"  Parameters: {params_swiglu:,}")
    print(f"  Formula: (Swish(xW1) ⊗ xW2)W3")
    
    # GeGLU
    output_geglu = geglu_ffn(x, training=False)
    params_geglu = geglu_ffn.count_parameters()
    
    print(f"\nGeGLU (T5):")
    print(f"  Output shape: {output_geglu.shape}")
    print(f"  Parameters: {params_geglu:,}")
    print(f"  Formula: (GELU(xW1) ⊗ xW2)W3")
    
    # Compare with standard
    standard_ffn = FeedForwardNetwork(d_model, d_ff, activation=gelu)
    params_standard = standard_ffn.count_parameters()
    
    print(f"\nParameter comparison:")
    print(f"  Standard FFN: {params_standard:,}")
    print(f"  Gated FFN: {params_swiglu:,}")
    print(f"  Increase: {(params_swiglu/params_standard - 1)*100:.1f}%")
    
    print("\n✓ Gated FFN uses 3 weight matrices (50% more params)")
    print("✓ But often performs better empirically")


def compare_ffn_outputs():
    """Compare outputs from different FFN architectures."""
    print("\n" + "=" * 70)
    print("FFN ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    d_model = 128
    d_ff = 512
    batch_size = 1
    seq_len = 4
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print(f"\nInput shape: {x.shape}")
    print(f"Configuration: d_model={d_model}, d_ff={d_ff}")
    
    # Compare architectures
    results = compare_ffn_architectures(x, d_model, d_ff)
    
    print(f"\n{'Architecture':20} | {'Output Norm':>12} | {'Parameters':>12}")
    print("-" * 52)
    
    for name, (output, params) in results.items():
        output_norm = np.linalg.norm(output)
        print(f"{name:20} | {output_norm:12.2f} | {params:12,}")
    
    print("\n✓ All architectures preserve shape")
    print("✓ Gated variants have more parameters")
    print("✓ Output magnitudes vary due to different activations")


def analyze_expansion_ratios():
    """Analyze different d_ff expansion ratios."""
    print("\n" + "=" * 70)
    print("FFN EXPANSION RATIO ANALYSIS")
    print("=" * 70)
    
    d_model = 256
    expansion_ratios = [2, 4, 8, 16]
    
    print(f"\nd_model: {d_model}")
    print(f"\n{'Expansion':>10} | {'d_ff':>8} | {'Parameters':>12} | {'FLOPs':>12}")
    print("-" * 55)
    
    for ratio in expansion_ratios:
        d_ff = d_model * ratio
        ffn = FeedForwardNetwork(d_model, d_ff, activation=gelu)
        params = ffn.count_parameters()
        
        # FLOPs for forward pass (approximate)
        # 2 matrix multiplications: d_model * d_ff + d_ff * d_model
        flops = 2 * (d_model * d_ff + d_ff * d_model)
        
        print(f"{ratio:10}x | {d_ff:8} | {params:12,} | {flops:12,}")
    
    print("\n✓ Common choices:")
    print("  • GPT-2/3: 4x expansion (d_ff = 4 * d_model)")
    print("  • LLaMA: ~2.7x for SwiGLU (accounts for extra matrix)")
    print("  • Trade-off between capacity and compute")


def compare_with_attention():
    """Compare FFN parameters with attention parameters."""
    print("\n" + "=" * 70)
    print("FFN VS ATTENTION: PARAMETER COUNT")
    print("=" * 70)
    
    d_model = 768  # GPT-2 small
    num_heads = 12
    d_ff = 3072  # 4x
    
    # Attention parameters: 4 projection matrices (Q, K, V, O)
    attention_params = 4 * d_model * d_model
    
    # FFN parameters
    ffn_params = d_model * d_ff + d_ff * d_model
    
    # Total per block
    total_params = attention_params + ffn_params
    
    print(f"\nModel: GPT-2 Small (d_model={d_model})")
    print("-" * 70)
    print(f"Attention parameters: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
    print(f"FFN parameters:       {ffn_params:,} ({ffn_params/total_params*100:.1f}%)")
    print(f"Total per block:      {total_params:,}")
    
    print("\n✓ FFN has ~2x more parameters than attention")
    print("✓ Most model capacity comes from FFN")
    print("✓ This is why FFN optimization matters")


def main():
    """Run all demos."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("FEED-FORWARD NETWORK DEMONSTRATIONS")
    print("=" * 70)
    
    demo_standard_ffn()
    demo_gated_ffn()
    compare_ffn_outputs()
    analyze_expansion_ratios()
    compare_with_attention()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. FFN typically uses 4x expansion (d_ff = 4 * d_model)")
    print("2. Gated FFNs (SwiGLU) have 50% more parameters but better performance")
    print("3. FFN contains ~2/3 of transformer parameters")
    print("4. Modern LLMs use SwiGLU (LLaMA, PaLM) or GeGLU (T5)")
    print("5. Expansion ratio is a key hyperparameter")
    print()


if __name__ == "__main__":
    main()

