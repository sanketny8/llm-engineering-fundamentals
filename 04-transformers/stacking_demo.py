"""Compare different transformer stacking configurations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from llm_engineering_fundamentals.transformer.block import StackedTransformer


def compare_layer_depths():
    """Compare transformers with different numbers of layers."""
    print("=" * 70)
    print("DEPTH COMPARISON: 1-LAYER VS 12-LAYER TRANSFORMERS")
    print("=" * 70)
    
    d_model = 256
    num_heads = 8
    d_ff = 1024
    batch_size = 1
    seq_len = 20
    
    depths = [1, 2, 4, 6, 12]
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  sequence_length: {seq_len}")
    
    print(f"\n{'Depth':>6} | {'Parameters':>12} | {'Params/Layer':>12}")
    print("-" * 40)
    
    results = []
    
    for depth in depths:
        model = StackedTransformer(
            num_layers=depth,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
        )
        
        param_counts = model.count_parameters()
        total_params = param_counts['total_parameters']
        params_per_layer = param_counts['total_per_block']
        
        print(f"{depth:6} | {total_params:>12,} | {params_per_layer:>12,}")
        
        # Test forward pass
        x = np.random.randn(batch_size, seq_len, d_model) * 0.1
        output = model(x)
        
        results.append({
            'depth': depth,
            'params': total_params,
            'output': output,
        })
    
    print("\n✓ Deeper models have more parameters (scales linearly)")
    print("✓ Each layer adds the same number of parameters")
    
    return results


def analyze_representation_evolution():
    """Analyze how representations evolve through layers."""
    print("\n" + "=" * 70)
    print("REPRESENTATION EVOLUTION THROUGH LAYERS")
    print("=" * 70)
    
    num_layers = 12
    d_model = 256
    num_heads = 8
    d_ff = 1024
    batch_size = 1
    seq_len = 16
    
    model = StackedTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    # Get outputs from all layers
    all_outputs = model(x, return_all_layers=True)
    
    print(f"\nAnalyzing {num_layers}-layer transformer:")
    print(f"  Input shape: {x.shape}")
    
    # Compute metrics for each layer
    metrics = {
        'layer': [],
        'norm': [],
        'change_from_input': [],
        'change_from_prev': [],
    }
    
    prev_output = x
    for i, layer_output in enumerate(all_outputs):
        layer_norm = np.linalg.norm(layer_output)
        change_from_input = np.linalg.norm(layer_output - x)
        change_from_prev = np.linalg.norm(layer_output - prev_output)
        
        metrics['layer'].append(i + 1)
        metrics['norm'].append(layer_norm)
        metrics['change_from_input'].append(change_from_input)
        metrics['change_from_prev'].append(change_from_prev)
        
        prev_output = layer_output
    
    # Print summary
    print(f"\n{'Layer':>6} | {'Norm':>10} | {'Δ from input':>14} | {'Δ from prev':>14}")
    print("-" * 55)
    for i in range(num_layers):
        print(f"{metrics['layer'][i]:6} | {metrics['norm'][i]:10.2f} | "
              f"{metrics['change_from_input'][i]:14.2f} | "
              f"{metrics['change_from_prev'][i]:14.2f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Representation norm
    axes[0].plot(metrics['layer'], metrics['norm'], 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('L2 Norm')
    axes[0].set_title('Representation Norm per Layer')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Change from input
    axes[1].plot(metrics['layer'], metrics['change_from_input'], 'o-', 
                 color='orange', linewidth=2, markersize=6)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('L2 Distance from Input')
    axes[1].set_title('How Much Representation Changes')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Change from previous layer
    axes[2].plot(metrics['layer'][1:], metrics['change_from_prev'][1:], 'o-', 
                 color='green', linewidth=2, markersize=6)
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('L2 Distance from Previous Layer')
    axes[2].set_title('Per-Layer Change Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "assets"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "representation_evolution.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_dir / 'representation_evolution.png'}")
    plt.close()
    
    print("\n✓ Representations evolve gradually through layers")
    print("✓ Early layers make bigger changes")
    print("✓ Later layers refine representations")


def compare_preln_vs_postln():
    """Compare Pre-LN and Post-LN architectures."""
    print("\n" + "=" * 70)
    print("PRE-LN VS POST-LN COMPARISON")
    print("=" * 70)
    
    num_layers = 6
    d_model = 128
    num_heads = 4
    d_ff = 512
    batch_size = 1
    seq_len = 16
    
    # Create both models
    model_preln = StackedTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_first=True,  # Pre-LN
    )
    
    model_postln = StackedTransformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_first=False,  # Post-LN
    )
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  d_model: {d_model}")
    
    # Forward pass
    output_preln = model_preln(x)
    output_postln = model_postln(x)
    
    # Get layer-wise outputs
    outputs_preln = model_preln(x, return_all_layers=True)
    outputs_postln = model_postln(x, return_all_layers=True)
    
    print(f"\nOutput shapes:")
    print(f"  Pre-LN:  {output_preln.shape}")
    print(f"  Post-LN: {output_postln.shape}")
    
    # Analyze gradient flow (simulated)
    print(f"\nRepresentation norms by layer:")
    print(f"{'Layer':>6} | {'Pre-LN':>10} | {'Post-LN':>10}")
    print("-" * 32)
    
    for i in range(num_layers):
        norm_preln = np.linalg.norm(outputs_preln[i])
        norm_postln = np.linalg.norm(outputs_postln[i])
        print(f"{i+1:6} | {norm_preln:10.2f} | {norm_postln:10.2f}")
    
    print("\n✓ Pre-LN (modern):")
    print("  • LayerNorm before each sublayer")
    print("  • More stable training, especially for deep networks")
    print("  • Used in GPT-3, LLaMA, etc.")
    
    print("\n✓ Post-LN (original):")
    print("  • LayerNorm after each sublayer")
    print("  • Can be harder to train at scale")
    print("  • Original Transformer architecture")


def analyze_parameter_scaling():
    """Analyze how parameters scale with model dimensions."""
    print("\n" + "=" * 70)
    print("PARAMETER SCALING ANALYSIS")
    print("=" * 70)
    
    print("\nHow do parameters scale with d_model?")
    print(f"{'d_model':>8} | {'d_ff (4x)':>10} | {'Params/Layer':>14} | {'12-Layer Total':>16}")
    print("-" * 60)
    
    for d_model in [128, 256, 512, 1024, 2048]:
        d_ff = 4 * d_model
        
        model = StackedTransformer(
            num_layers=1,
            d_model=d_model,
            num_heads=8,
            d_ff=d_ff,
        )
        
        params_per_layer = model.count_parameters()['total_per_block']
        params_12_layers = params_per_layer * 12
        
        print(f"{d_model:8} | {d_ff:10} | {params_per_layer:14,} | {params_12_layers:16,}")
    
    print("\n✓ Parameters scale with O(d_model²) for attention")
    print("✓ Parameters scale with O(d_model × d_ff) for FFN")
    print("✓ FFN typically dominates (d_ff = 4 × d_model)")
    
    print("\n" + "-" * 70)
    print("Typical LLM Configurations:")
    print("-" * 70)
    
    configs = [
        {"name": "GPT-2 Small", "layers": 12, "d_model": 768, "heads": 12, "d_ff": 3072},
        {"name": "GPT-2 Medium", "layers": 24, "d_model": 1024, "heads": 16, "d_ff": 4096},
        {"name": "GPT-2 Large", "layers": 36, "d_model": 1280, "heads": 20, "d_ff": 5120},
        {"name": "GPT-3 (davinci)", "layers": 96, "d_model": 12288, "heads": 96, "d_ff": 49152},
    ]
    
    print(f"\n{'Model':>20} | {'Layers':>7} | {'d_model':>8} | {'Total Params':>15}")
    print("-" * 60)
    
    for config in configs:
        model = StackedTransformer(
            num_layers=config["layers"],
            d_model=config["d_model"],
            num_heads=config["heads"],
            d_ff=config["d_ff"],
        )
        
        total_params = model.count_parameters()['total_parameters']
        print(f"{config['name']:>20} | {config['layers']:7} | {config['d_model']:8} | {total_params:15,}")
    
    print("\n✓ GPT-3 has ~175B parameters (mostly in the transformer blocks)")
    print("✓ Scaling laws suggest larger models perform better")


def main():
    """Run all stacking demos."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("TRANSFORMER STACKING DEMONSTRATIONS")
    print("=" * 70)
    
    compare_layer_depths()
    analyze_representation_evolution()
    compare_preln_vs_postln()
    analyze_parameter_scaling()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Deeper models have more capacity (parameters scale linearly)")
    print("2. Representations evolve gradually through layers")
    print("3. Pre-LN is more stable than Post-LN for deep networks")
    print("4. Parameters scale quadratically with d_model")
    print("5. FFN typically has 4x expansion (d_ff = 4 × d_model)")
    print()


if __name__ == "__main__":
    main()


