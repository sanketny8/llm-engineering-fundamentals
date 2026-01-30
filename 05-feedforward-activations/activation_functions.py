"""Demo script comparing activation functions."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from llm_engineering_fundamentals.ffn.activations import (
    relu,
    gelu,
    gelu_approx,
    swish,
    silu,
    relu_derivative,
    gelu_derivative,
    compare_activations,
)


def visualize_activation_functions():
    """Visualize different activation functions."""
    print("=" * 70)
    print("ACTIVATION FUNCTIONS VISUALIZATION")
    print("=" * 70)
    
    # Create input range
    x = np.linspace(-5, 5, 1000)
    
    # Compute activations
    activations = {
        "ReLU": relu(x),
        "GELU": gelu(x),
        "GELU (approx)": gelu_approx(x),
        "Swish/SiLU": swish(x),
    }
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Activation functions
    ax = axes[0]
    for name, y in activations.items():
        ax.plot(x, y, label=name, linewidth=2)
    
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_xlabel('Input (x)', fontsize=11)
    ax.set_ylabel('Output f(x)', fontsize=11)
    ax.set_title('Activation Functions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradients
    ax = axes[1]
    gradients = {
        "ReLU": relu_derivative(x),
        "GELU": gelu_derivative(x),
    }
    
    for name, y in gradients.items():
        ax.plot(x, y, label=name, linewidth=2)
    
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_xlabel('Input (x)', fontsize=11)
    ax.set_ylabel("Gradient f'(x)", fontsize=11)
    ax.set_title('Activation Gradients', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.5)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(__file__).parent / "assets"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "activation_functions.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_dir / 'activation_functions.png'}")
    plt.close()


def compare_activation_properties():
    """Compare properties of different activations."""
    print("\n" + "=" * 70)
    print("ACTIVATION FUNCTION COMPARISON")
    print("=" * 70)
    
    # Test inputs
    x_neg = np.array([-2.0, -1.0, -0.5])
    x_zero = np.array([0.0])
    x_pos = np.array([0.5, 1.0, 2.0])
    
    print("\nFor negative inputs (x = -2, -1, -0.5):")
    print("-" * 70)
    results_neg = compare_activations(x_neg)
    for name, values in results_neg.items():
        print(f"{name:15}: {values}")
    
    print("\nFor zero input (x = 0):")
    print("-" * 70)
    results_zero = compare_activations(x_zero)
    for name, values in results_zero.items():
        print(f"{name:15}: {values}")
    
    print("\nFor positive inputs (x = 0.5, 1, 2):")
    print("-" * 70)
    results_pos = compare_activations(x_pos)
    for name, values in results_pos.items():
        print(f"{name:15}: {values}")
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print("\n1. ReLU:")
    print("   • Zeros out all negative inputs (dead neurons)")
    print("   • Linear for positive inputs")
    print("   • Not differentiable at x=0")
    
    print("\n2. GELU:")
    print("   • Smooth everywhere (fully differentiable)")
    print("   • Allows small negative values through")
    print("   • Approaches linear for large positive x")
    
    print("\n3. Swish/SiLU:")
    print("   • Also smooth everywhere")
    print("   • Similar to GELU but slightly different shape")
    print("   • Self-gated (x * sigmoid(x))")
    
    print("\n4. GELU (approx):")
    print("   • Fast approximation of exact GELU")
    print("   • Negligible difference in practice")


def analyze_dead_neurons():
    """Demonstrate ReLU's dead neuron problem."""
    print("\n" + "=" * 70)
    print("DEAD NEURON PROBLEM (ReLU)")
    print("=" * 70)
    
    # Simulate a batch of inputs
    np.random.seed(42)
    x = np.random.randn(100, 512) - 1.0  # Biased towards negative
    
    # Apply ReLU
    relu_out = relu(x)
    
    # Count dead neurons (neurons that always output 0)
    active_neurons = np.any(relu_out > 0, axis=0)
    dead_count = np.sum(~active_neurons)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Mean input value: {np.mean(x):.3f}")
    print(f"Negative inputs: {np.sum(x < 0) / x.size * 100:.1f}%")
    
    print(f"\nAfter ReLU:")
    print(f"  Active neurons: {np.sum(active_neurons)} / {x.shape[1]}")
    print(f"  Dead neurons: {dead_count} / {x.shape[1]} ({dead_count/x.shape[1]*100:.1f}%)")
    print(f"  Mean output: {np.mean(relu_out):.3f}")
    print(f"  Sparsity: {np.sum(relu_out == 0) / relu_out.size * 100:.1f}%")
    
    # Compare with GELU
    gelu_out = gelu(x)
    print(f"\nAfter GELU:")
    print(f"  Mean output: {np.mean(gelu_out):.3f}")
    print(f"  Zeros: {np.sum(gelu_out == 0) / gelu_out.size * 100:.1f}%")
    print(f"  Min value: {np.min(gelu_out):.3f}")
    
    print("\n✓ GELU allows gradient flow even for negative inputs")
    print("✓ ReLU can have neurons that never activate (dead)")


def benchmark_speed():
    """Benchmark speed of different activations."""
    print("\n" + "=" * 70)
    print("ACTIVATION FUNCTION SPEED BENCHMARK")
    print("=" * 70)
    
    import time
    
    # Create large input
    x = np.random.randn(1000, 4096)
    
    activations = {
        "ReLU": relu,
        "GELU (exact)": gelu,
        "GELU (approx)": gelu_approx,
        "Swish/SiLU": swish,
    }
    
    print(f"\nInput shape: {x.shape}")
    print(f"Number of elements: {x.size:,}")
    print("\nTiming 100 iterations:")
    print("-" * 70)
    
    results = {}
    for name, func in activations.items():
        start = time.time()
        for _ in range(100):
            _ = func(x)
        elapsed = time.time() - start
        results[name] = elapsed
        print(f"{name:20}: {elapsed*1000:.2f} ms")
    
    # Relative speed
    baseline = results["ReLU"]
    print("\nRelative speed (vs ReLU):")
    print("-" * 70)
    for name, elapsed in results.items():
        ratio = elapsed / baseline
        print(f"{name:20}: {ratio:.2f}x")
    
    print("\n✓ ReLU is fastest (simple max operation)")
    print("✓ GELU is slower due to erf computation")
    print("✓ Approximations can help bridge the gap")


def main():
    """Run all demos."""
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("ACTIVATION FUNCTIONS DEMONSTRATIONS")
    print("=" * 70)
    
    visualize_activation_functions()
    compare_activation_properties()
    analyze_dead_neurons()
    benchmark_speed()
    
    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. ReLU is simple but has dead neuron problem")
    print("2. GELU is smooth and allows better gradient flow")
    print("3. Swish/SiLU is self-gated and performs well")
    print("4. Modern LLMs prefer GELU or Swish over ReLU")
    print("5. Speed vs performance trade-off exists")
    print()


if __name__ == "__main__":
    main()

