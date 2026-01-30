"""Demo: Decoding Strategies (Greedy vs Beam Search)"""

import numpy as np
from llm_engineering_fundamentals.decoding.strategies import (
    greedy_decode,
    beam_search,
    BeamSearcher,
)


def dummy_model(tokens: np.ndarray) -> np.ndarray:
    """
    Dummy model for demonstration.
    Creates somewhat realistic probability distributions.
    """
    batch_size, seq_len = tokens.shape
    vocab_size = 100
    
    # Create logits with some structure
    logits = np.random.randn(batch_size, seq_len, vocab_size) * 0.5
    
    # Make next token somewhat predictable based on last token
    for i in range(batch_size):
        last_token = tokens[i, -1]
        # Favor next few tokens
        for offset in range(1, 5):
            next_token = (last_token + offset) % vocab_size
            logits[i, -1, next_token] += 2.0 + np.random.rand()
    
    return logits


def demo_greedy_vs_beam():
    """Compare greedy and beam search."""
    print("=" * 70)
    print("GREEDY VS BEAM SEARCH COMPARISON")
    print("=" * 70)
    
    # Starting prompt
    prompt = np.array([[5]])
    max_length = 15
    
    print(f"\nStarting prompt: {prompt[0]}")
    print(f"Max length: {max_length}")
    
    # Greedy decoding
    print("\n" + "-" * 70)
    print("GREEDY DECODING (deterministic, fast)")
    print("-" * 70)
    
    greedy_output = greedy_decode(
        model_fn=dummy_model,
        initial_tokens=prompt,
        max_length=max_length,
    )
    
    print(f"Output: {greedy_output[0]}")
    print(f"Length: {len(greedy_output[0])}")
    
    # Beam search with different beam widths
    beam_widths = [1, 3, 5]
    
    for num_beams in beam_widths:
        print("\n" + "-" * 70)
        print(f"BEAM SEARCH (num_beams={num_beams})")
        print("-" * 70)
        
        beam_output = beam_search(
            model_fn=dummy_model,
            initial_tokens=prompt,
            num_beams=num_beams,
            max_length=max_length,
            length_penalty=0.6,
        )
        
        print(f"Output: {beam_output}")
        print(f"Length: {len(beam_output)}")
        
        if num_beams == 1:
            # Beam width 1 should match greedy
            if np.array_equal(greedy_output[0], beam_output):
                print("✓ Beam width 1 matches greedy (as expected)")


def demo_length_penalty():
    """Demonstrate length penalty effects."""
    print("\n" + "=" * 70)
    print("LENGTH PENALTY COMPARISON")
    print("=" * 70)
    
    prompt = np.array([[5]])
    
    penalties = [0.0, 0.6, 1.0]
    
    for penalty in penalties:
        print(f"\n--- Length Penalty: {penalty} ---")
        
        output = beam_search(
            model_fn=dummy_model,
            initial_tokens=prompt,
            num_beams=5,
            max_length=20,
            length_penalty=penalty,
        )
        
        print(f"Output length: {len(output)}")
        print(f"Sequence: {output[:10]}...")  # First 10 tokens
        
        if penalty == 0.0:
            print("(No penalty: favors shorter sequences)")
        elif penalty == 0.6:
            print("(Moderate penalty: balanced)")
        elif penalty == 1.0:
            print("(Full penalty: encourages longer sequences)")


def demo_beam_searcher_class():
    """Demonstrate BeamSearcher class for more control."""
    print("\n" + "=" * 70)
    print("BEAM SEARCHER CLASS (Advanced Usage)")
    print("=" * 70)
    
    # Create searcher
    searcher = BeamSearcher(
        model_fn=dummy_model,
        num_beams=5,
        length_penalty=0.6,
        early_stopping=True,
    )
    
    prompt = np.array([[5]])
    
    print(f"Configuration:")
    print(f"  - Num beams: {searcher.num_beams}")
    print(f"  - Length penalty: {searcher.length_penalty}")
    print(f"  - Early stopping: {searcher.early_stopping}")
    
    # Search
    best_sequence, best_score = searcher.search(
        initial_tokens=prompt,
        max_length=20,
        eos_token_id=50,  # Stop if token 50 is generated
    )
    
    print(f"\nBest sequence: {best_sequence}")
    print(f"Score (log probability): {best_score:.4f}")
    print(f"Length: {len(best_sequence)}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "DECODING STRATEGIES DEMO" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    
    np.random.seed(42)  # For reproducibility
    
    # Demo 1: Greedy vs Beam
    demo_greedy_vs_beam()
    
    # Demo 2: Length penalty
    demo_length_penalty()
    
    # Demo 3: BeamSearcher class
    demo_beam_searcher_class()
    
    # Summary
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. GREEDY:
   - Fast (O(1) per step)
   - Deterministic
   - Can get stuck in local optima
   
2. BEAM SEARCH:
   - Better quality (explores multiple paths)
   - Slower (O(B) per step, B = beam width)
   - B=5 is common for translation
   
3. LENGTH PENALTY:
   - Prevents bias toward shorter sequences
   - penalty=0.6 is common (Google NMT)
   - penalty=0.0 favors shorter, penalty=1.0 is neutral
   
4. WHEN TO USE:
   - Greedy: Testing, simple generation, speed critical
   - Beam: Translation, summarization, quality matters
    """)
    
    print("=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()

