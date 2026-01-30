#!/usr/bin/env python3
"""
Embedding Comparison: One-Hot vs Learned Embeddings
Demonstrates different embedding strategies and cosine similarity.
"""
import numpy as np
from typing import Dict, List


def one_hot_embedding(vocab_size: int, token_id: int) -> np.ndarray:
    """Create a one-hot vector for a token."""
    vec = np.zeros(vocab_size)
    vec[token_id] = 1.0
    return vec


def random_learned_embedding(vocab_size: int, embedding_dim: int, seed: int = 42) -> np.ndarray:
    """Simulate learned embeddings with random initialization."""
    np.random.seed(seed)
    # Small random values similar to typical initialization
    embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def compare_embeddings_demo():
    """Demonstrate differences between embedding types."""
    # Simple vocabulary
    vocab = ["hello", "world", "tokenization", "embedding", "neural", "network"]
    vocab_size = len(vocab)
    embedding_dim = 16

    word_to_id = {word: i for i, word in enumerate(vocab)}

    print("=" * 70)
    print("Embedding Comparison Demo")
    print("=" * 70)

    # Generate learned embeddings
    learned_emb = random_learned_embedding(vocab_size, embedding_dim)

    # Compare pairs of words
    word_pairs = [
        ("hello", "world"),  # Related words
        ("tokenization", "embedding"),  # Technical terms
        ("neural", "network"),  # Very related words
        ("hello", "tokenization"),  # Unrelated
    ]

    print("\n### One-Hot Embeddings ###")
    print(f"Dimension: {vocab_size} (vocab size)")
    print("\nCosine Similarities:")

    for word1, word2 in word_pairs:
        id1 = word_to_id[word1]
        id2 = word_to_id[word2]

        vec1 = one_hot_embedding(vocab_size, id1)
        vec2 = one_hot_embedding(vocab_size, id2)

        sim = cosine_similarity(vec1, vec2)
        print(f"  {word1:15} ↔ {word2:15}: {sim:.4f}")

    print("\n⚠️  Problem: All pairs have similarity = 0.0!")
    print("    One-hot vectors are orthogonal → no semantic relationship captured")

    print("\n" + "=" * 70)
    print("### Learned Embeddings ###")
    print(f"Dimension: {embedding_dim} (much smaller than vocab size)")
    print("\nCosine Similarities:")

    for word1, word2 in word_pairs:
        id1 = word_to_id[word1]
        id2 = word_to_id[word2]

        vec1 = learned_emb[id1]
        vec2 = learned_emb[id2]

        sim = cosine_similarity(vec1, vec2)
        print(f"  {word1:15} ↔ {word2:15}: {sim:.4f}")

    print("\n✅ Learned embeddings capture semantic relationships")
    print("   (Note: These are random; real embeddings would show clearer patterns)")

    # Memory comparison
    print("\n" + "=" * 70)
    print("### Memory Efficiency ###")

    one_hot_memory = vocab_size * vocab_size * 4  # 4 bytes per float32
    learned_memory = vocab_size * embedding_dim * 4

    print(f"One-hot:  {vocab_size:,} words × {vocab_size:,} dims = {one_hot_memory:,} bytes")
    print(f"Learned:  {vocab_size:,} words × {embedding_dim:,} dims = {learned_memory:,} bytes")
    print(f"\nMemory reduction: {one_hot_memory / learned_memory:.1f}x smaller")

    # For GPT-style models
    print("\n" + "=" * 70)
    print("### Real-World Example: GPT-3 ###")

    gpt3_vocab = 50257
    gpt3_dim = 12288  # GPT-3 175B model

    one_hot_size_mb = (gpt3_vocab * gpt3_vocab * 4) / (1024**2)
    learned_size_mb = (gpt3_vocab * gpt3_dim * 4) / (1024**2)

    print(f"Vocabulary size: {gpt3_vocab:,}")
    print(f"Embedding dimension: {gpt3_dim:,}")
    print(f"\nOne-hot embeddings: {one_hot_size_mb:,.1f} MB (impractical!)")
    print(f"Learned embeddings: {learned_size_mb:,.1f} MB")
    print(f"Reduction: {one_hot_size_mb / learned_size_mb:.0f}x smaller")

    print("\n" + "=" * 70)


def distance_metrics_demo():
    """Demonstrate different distance metrics."""
    print("\n### Distance Metrics Comparison ###\n")

    vec1 = np.array([1.0, 0.5, 0.2])
    vec2 = np.array([0.9, 0.6, 0.1])
    vec3 = np.array([0.0, 0.0, 1.0])

    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    print(f"Vector 3: {vec3}")

    # Cosine similarity
    print("\nCosine Similarity (range: -1 to 1, higher = more similar):")
    print(f"  vec1 ↔ vec2: {cosine_similarity(vec1, vec2):.4f} (similar)")
    print(f"  vec1 ↔ vec3: {cosine_similarity(vec1, vec3):.4f} (different)")

    # Euclidean distance
    print("\nEuclidean Distance (lower = more similar):")
    print(f"  vec1 ↔ vec2: {np.linalg.norm(vec1 - vec2):.4f}")
    print(f"  vec1 ↔ vec3: {np.linalg.norm(vec1 - vec3):.4f}")

    # Dot product
    print("\nDot Product (higher = more similar, but not normalized):")
    print(f"  vec1 ↔ vec2: {np.dot(vec1, vec2):.4f}")
    print(f"  vec1 ↔ vec3: {np.dot(vec1, vec3):.4f}")

    print("\n💡 Cosine similarity is preferred for embeddings because:")
    print("   - Normalized (independent of vector magnitude)")
    print("   - Captures angular similarity")
    print("   - Standard in NLP/LLM applications")


if __name__ == "__main__":
    compare_embeddings_demo()
    distance_metrics_demo()

    print("\n" + "=" * 70)
    print("🎓 Key Takeaways:")
    print("=" * 70)
    print("1. One-hot: Simple but no semantic relationships, huge memory")
    print("2. Learned: Dense, captures semantics, memory-efficient")
    print("3. LLMs use learned embeddings (e.g., GPT: 50K vocab → 12K dim)")
    print("4. Cosine similarity is the standard metric for embedding comparison")
    print("=" * 70)



