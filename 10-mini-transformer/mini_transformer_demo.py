"""Demo: Complete Mini Transformer Model"""

import numpy as np
from llm_engineering_fundamentals.models import create_mini_gpt, MiniTransformerConfig, MiniTransformer


def demo_model_creation():
    """Demonstrate creating models with different configurations."""
    print("=" * 70)
    print("MODEL CREATION")
    print("=" * 70)
    
    # Small model (for testing/learning)
    print("\n--- Small Model ---")
    small_model = create_mini_gpt(vocab_size=5000, d_model=128, num_layers=4)
    small_params = small_model.count_parameters()
    print(f"Vocabulary: 5,000 tokens")
    print(f"Hidden size: 128")
    print(f"Layers: 4")
    print(f"Parameters: {small_params['total']:,}")
    
    # Medium model (default)
    print("\n--- Medium Model (Default) ---")
    medium_model = create_mini_gpt(vocab_size=10000, d_model=256, num_layers=6)
    medium_params = medium_model.count_parameters()
    print(f"Vocabulary: 10,000 tokens")
    print(f"Hidden size: 256")
    print(f"Layers: 6")
    print(f"Parameters: {medium_params['total']:,}")
    
    # Custom configuration
    print("\n--- Custom Configuration ---")
    config = MiniTransformerConfig(
        vocab_size=50000,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        max_seq_len=2048,
        positional_type="learned",
        dropout=0.1,
        norm_first=True,  # Pre-LN (modern)
    )
    custom_model = MiniTransformer(config)
    custom_params = custom_model.count_parameters()
    print(f"Vocabulary: 50,000 tokens")
    print(f"Hidden size: 512")
    print(f"Layers: 12")
    print(f"Parameters: {custom_params['total']:,}")
    print("(Similar to GPT-2 Small)")


def demo_forward_pass():
    """Demonstrate forward pass through the model."""
    print("\n" + "=" * 70)
    print("FORWARD PASS")
    print("=" * 70)
    
    # Create model
    model = create_mini_gpt(vocab_size=1000, d_model=128, num_layers=4)
    
    # Create input
    batch_size = 2
    seq_len = 10
    token_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nInput shape: {token_ids.shape}")
    print(f"Input tokens (first sequence): {token_ids[0]}")
    
    # Forward pass
    logits = model(token_ids)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, vocab_size=1000)")
    
    # Get probabilities for next token (last position)
    last_logits = logits[0, -1, :]
    probs = np.exp(last_logits - np.max(last_logits))
    probs = probs / np.sum(probs)
    
    top_5_tokens = np.argsort(probs)[-5:][::-1]
    print(f"\nTop 5 most likely next tokens:")
    for i, token in enumerate(top_5_tokens):
        print(f"  {i+1}. Token {token}: {probs[token]:.4f}")


def demo_greedy_generation():
    """Demonstrate greedy generation."""
    print("\n" + "=" * 70)
    print("GREEDY GENERATION")
    print("=" * 70)
    
    model = create_mini_gpt(vocab_size=100, d_model=64, num_layers=2)
    
    # Starting prompt
    prompt = np.array([[5, 10, 15]])
    print(f"\nPrompt: {prompt[0]}")
    print(f"Generating...")
    
    # Generate (deterministic)
    output = model.generate_greedy(prompt, max_length=20)
    
    print(f"\nGenerated sequence: {output[0]}")
    print(f"Length: {len(output[0])}")
    print("\n(Greedy: always picks most likely token - fast but deterministic)")


def demo_beam_search():
    """Demonstrate beam search generation."""
    print("\n" + "=" * 70)
    print("BEAM SEARCH GENERATION")
    print("=" * 70)
    
    model = create_mini_gpt(vocab_size=100, d_model=64, num_layers=2)
    
    prompt = np.array([[5, 10, 15]])
    print(f"\nPrompt: {prompt[0]}")
    
    beam_widths = [1, 3, 5]
    
    for num_beams in beam_widths:
        print(f"\n--- Num Beams: {num_beams} ---")
        
        output = model.generate_beam_search(
            prompt,
            num_beams=num_beams,
            max_length=20,
            length_penalty=0.6
        )
        
        print(f"Generated: {output[:15]}...")  # First 15 tokens
        print(f"Length: {len(output)}")


def demo_sampling():
    """Demonstrate sampling-based generation."""
    print("\n" + "=" * 70)
    print("SAMPLING-BASED GENERATION")
    print("=" * 70)
    
    model = create_mini_gpt(vocab_size=100, d_model=64, num_layers=2)
    
    prompt = np.array([[5, 10, 15]])
    print(f"\nPrompt: {prompt[0]}")
    
    configs = [
        {"name": "Low Temperature", "temp": 0.5, "top_k": 0, "top_p": 1.0},
        {"name": "Balanced", "temp": 0.8, "top_k": 0, "top_p": 0.95},
        {"name": "Creative", "temp": 1.2, "top_k": 0, "top_p": 0.98},
        {"name": "Top-k", "temp": 0.8, "top_k": 20, "top_p": 1.0},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"  temperature={config['temp']}, top_k={config['top_k']}, top_p={config['top_p']}")
        
        output = model.generate_sample(
            prompt,
            max_length=20,
            temperature=config['temp'],
            top_k=config['top_k'],
            top_p=config['top_p'],
        )
        
        print(f"  Generated: {output[0][:15]}...")  # First 15 tokens


def demo_parameter_breakdown():
    """Show detailed parameter breakdown."""
    print("\n" + "=" * 70)
    print("PARAMETER BREAKDOWN")
    print("=" * 70)
    
    model = create_mini_gpt(vocab_size=10000, d_model=256, num_layers=6)
    params = model.count_parameters()
    
    print(f"\nEmbedding Layer:")
    print(f"  Parameters: {params['embedding']:,}")
    print(f"  (vocab_size × d_model + positional)")
    
    print(f"\nTransformer Layers:")
    print(f"  Parameters: {params['transformer']:,}")
    print(f"  (6 layers × attention + FFN)")
    
    print(f"\nTotal Parameters: {params['total']:,}")
    
    # Estimate per-layer parameters
    params_per_layer = params['transformer'] // 6
    print(f"\nPer-layer parameters: ~{params_per_layer:,}")
    
    # Memory estimate (fp32)
    memory_mb = (params['total'] * 4) / (1024 * 1024)
    print(f"\nMemory (FP32): ~{memory_mb:.1f} MB")


def demo_architecture_comparison():
    """Compare different architectural choices."""
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    base_config = {
        "vocab_size": 10000,
        "d_model": 256,
        "num_layers": 6,
    }
    
    # Depth vs Width
    print("\n--- Depth vs Width ---")
    
    deep_model = create_mini_gpt(vocab_size=10000, d_model=128, num_layers=12)
    deep_params = deep_model.count_parameters()
    print(f"Deep Model (12 layers, d_model=128): {deep_params['total']:,} params")
    
    wide_model = create_mini_gpt(vocab_size=10000, d_model=384, num_layers=4)
    wide_params = wide_model.count_parameters()
    print(f"Wide Model (4 layers, d_model=384): {wide_params['total']:,} params")
    
    print("\n(Modern LLMs tend to be deep rather than wide)")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "MINI TRANSFORMER DEMO" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    
    np.random.seed(42)  # For reproducibility
    
    # Demo 1: Model creation
    demo_model_creation()
    
    # Demo 2: Forward pass
    demo_forward_pass()
    
    # Demo 3: Greedy generation
    demo_greedy_generation()
    
    # Demo 4: Beam search
    demo_beam_search()
    
    # Demo 5: Sampling
    demo_sampling()
    
    # Demo 6: Parameters
    demo_parameter_breakdown()
    
    # Demo 7: Architecture
    demo_architecture_comparison()
    
    # Summary
    print("\n" + "=" * 70)
    print("WHAT YOU'VE BUILT")
    print("=" * 70)
    print("""
A COMPLETE GPT-STYLE TRANSFORMER that can:

✓ Process token sequences (forward pass)
✓ Generate text (autoregressive)
✓ Multiple generation modes:
  - Greedy (fast, deterministic)
  - Beam search (better quality)
  - Sampling (creative, diverse)
✓ Configurable architecture:
  - Any vocab size
  - Any model size
  - Any number of layers
✓ Modern techniques:
  - Pre-LN (training stability)
  - Tied embeddings (parameter efficiency)
  - Causal masking (autoregressive generation)

This is the SAME architecture as GPT-2/GPT-3/LLaMA!
The only difference is scale (they have billions of parameters).

You now understand transformers at a deep, implementation level.
    """)
    
    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)
    print("\nNext: Deploy to GitHub and showcase your expertise! 🚀")


if __name__ == "__main__":
    main()

