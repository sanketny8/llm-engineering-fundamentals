"""Demo: Sampling Methods (Temperature, Top-k, Top-p)"""

import numpy as np
from llm_engineering_fundamentals.sampling.methods import (
    sample_with_temperature,
    top_k_sampling,
    top_p_sampling,
    sample_next_token,
)


def demo_temperature():
    """Demonstrate temperature sampling."""
    print("=" * 70)
    print("TEMPERATURE SAMPLING")
    print("=" * 70)
    
    # Create a peaked distribution (like a confident model)
    logits = np.array([1.0, 2.0, 5.0, 1.5, 0.5])
    
    print(f"\nOriginal logits: {logits}")
    print(f"Most likely token: {np.argmax(logits)} (token 2)")
    
    temperatures = [0.0, 0.5, 1.0, 2.0]
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        
        # Sample multiple times
        samples = [sample_with_temperature(logits, temp) for _ in range(100)]
        
        # Count unique tokens
        unique = len(set(samples))
        most_common = max(set(samples), key=samples.count)
        frequency = samples.count(most_common) / 100
        
        print(f"Unique tokens sampled: {unique} / {len(logits)}")
        print(f"Most common token: {most_common} ({frequency:.1%})")
        
        if temp == 0.0:
            print("(Temperature 0 = Greedy: always pick most likely)")
        elif temp < 1.0:
            print("(Low temperature: more confident, less diverse)")
        elif temp == 1.0:
            print("(Temperature 1: original distribution)")
        else:
            print("(High temperature: more random, more diverse)")


def demo_top_k():
    """Demonstrate top-k sampling."""
    print("\n" + "=" * 70)
    print("TOP-K SAMPLING")
    print("=" * 70)
    
    # Create logits where some tokens are clearly better
    logits = np.array([1.0, 2.0, 5.0, 4.5, 4.0, 0.5, 0.2, 0.1])
    
    print(f"\nLogits: {logits}")
    print(f"Top-5 tokens by logit: {np.argsort(logits)[-5:][::-1]}")
    
    k_values = [3, 5, 0]  # 0 = no filtering
    
    for k in k_values:
        print(f"\n--- Top-k: k={k} ---")
        
        samples = [top_k_sampling(logits, k=k, temperature=1.0) for _ in range(200)]
        
        unique = set(samples)
        print(f"Tokens sampled: {sorted(unique)}")
        print(f"Number of unique tokens: {len(unique)}")
        
        if k == 3:
            print("(k=3: Only samples from top 3 most likely tokens)")
        elif k == 5:
            print("(k=5: More diversity while filtering unlikely tokens)")
        else:
            print("(k=0: No filtering, samples from all tokens)")


def demo_top_p():
    """Demonstrate top-p (nucleus) sampling."""
    print("\n" + "=" * 70)
    print("TOP-P (NUCLEUS) SAMPLING")
    print("=" * 70)
    
    # Test with different distribution shapes
    
    # Peaked distribution
    print("\n--- Peaked Distribution ---")
    peaked_logits = np.array([0.1, 0.2, 10.0, 0.1, 0.1])
    print(f"Logits: {peaked_logits}")
    
    samples = [top_p_sampling(peaked_logits, p=0.95, temperature=1.0) for _ in range(100)]
    print(f"Tokens sampled: {set(samples)}")
    print("(Peaked distribution → small nucleus)")
    
    # Flat distribution
    print("\n--- Flat Distribution ---")
    flat_logits = np.array([2.0, 2.1, 2.2, 2.0, 1.9, 2.1])
    print(f"Logits: {flat_logits}")
    
    samples = [top_p_sampling(flat_logits, p=0.95, temperature=1.0) for _ in range(100)]
    print(f"Tokens sampled: {set(samples)}")
    print("(Flat distribution → larger nucleus)")
    
    print("\n--- p value comparison ---")
    logits = np.array([1.0, 2.0, 5.0, 4.0, 3.0, 0.5])
    
    for p in [0.8, 0.9, 0.95, 1.0]:
        samples = [top_p_sampling(logits, p=p, temperature=1.0) for _ in range(200)]
        unique = len(set(samples))
        print(f"p={p}: {unique} unique tokens sampled")


def demo_repetition_penalty():
    """Demonstrate repetition penalty."""
    print("\n" + "=" * 70)
    print("REPETITION PENALTY")
    print("=" * 70)
    
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    previous_tokens = np.array([0, 0, 0])  # Token 0 repeated 3 times
    
    print(f"Logits: {logits}")
    print(f"Previously generated: {previous_tokens} (token 0 repeated)")
    
    penalties = [1.0, 1.2, 1.5]
    
    for penalty in penalties:
        print(f"\n--- Repetition Penalty: {penalty} ---")
        
        samples = [
            sample_next_token(
                logits,
                temperature=1.0,
                repetition_penalty=penalty,
                previous_tokens=previous_tokens if penalty != 1.0 else None
            )
            for _ in range(100)
        ]
        
        token_0_freq = samples.count(0) / 100
        token_1_freq = samples.count(1) / 100
        
        print(f"Token 0 frequency: {token_0_freq:.1%} (was most likely)")
        print(f"Token 1 frequency: {token_1_freq:.1%} (second most likely)")
        
        if penalty == 1.0:
            print("(No penalty)")
        else:
            print(f"(Token 0 penalized → appears less frequently)")


def demo_combined():
    """Demonstrate combining multiple sampling strategies."""
    print("\n" + "=" * 70)
    print("COMBINED SAMPLING (Real-world Usage)")
    print("=" * 70)
    
    logits = np.array([1.0, 2.0, 5.0, 4.5, 4.0, 3.5, 0.5, 0.2])
    
    configs = [
        {"name": "Deterministic", "temp": 0.0, "top_k": 0, "top_p": 1.0},
        {"name": "Balanced", "temp": 0.8, "top_k": 0, "top_p": 0.95},
        {"name": "Creative", "temp": 1.2, "top_k": 0, "top_p": 0.98},
        {"name": "Top-k", "temp": 0.8, "top_k": 50, "top_p": 1.0},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"  temperature={config['temp']}, top_k={config['top_k']}, top_p={config['top_p']}")
        
        samples = [
            sample_next_token(
                logits,
                temperature=config['temp'],
                top_k=config['top_k'],
                top_p=config['top_p']
            )
            for _ in range(100)
        ]
        
        unique = len(set(samples))
        most_common = max(set(samples), key=samples.count)
        frequency = samples.count(most_common) / 100
        
        print(f"  Unique tokens: {unique}")
        print(f"  Most common: token {most_common} ({frequency:.1%})")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "SAMPLING METHODS DEMO" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    
    np.random.seed(42)  # For reproducibility
    
    # Demo 1: Temperature
    demo_temperature()
    
    # Demo 2: Top-k
    demo_top_k()
    
    # Demo 3: Top-p
    demo_top_p()
    
    # Demo 4: Repetition penalty
    demo_repetition_penalty()
    
    # Demo 5: Combined
    demo_combined()
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. TEMPERATURE:
   - Controls randomness/creativity
   - 0.0 = greedy, 0.7-0.9 = balanced, >1.0 = creative
   
2. TOP-K:
   - Filters to top-k most likely tokens
   - k=50 is common (GPT-2 default)
   - Fixed cutoff (doesn't adapt to distribution)
   
3. TOP-P (NUCLEUS):
   - Adaptive cutoff based on cumulative probability
   - p=0.95 is common (GPT-3 default)
   - Better than top-k (adapts to distribution shape)
   
4. REPETITION PENALTY:
   - Reduces repetitive outputs
   - penalty=1.2 is typical
   - Important for dialogue systems
   
5. TYPICAL SETTINGS:
   - Balanced: temperature=0.8, top_p=0.95
   - Creative: temperature=1.2, top_p=0.98
   - Deterministic: temperature=0.0
    """)
    
    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()

