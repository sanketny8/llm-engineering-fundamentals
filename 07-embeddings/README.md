# Project 07: Embeddings (Token + Positional Combined)

## 🎯 Concept

Build the complete input embedding layer that combines token embeddings and positional embeddings, forming the foundation of transformer input processing.

## 🏗️ What We'll Build

### 1. Token Embeddings
- Embedding table initialization
- Vocabulary size and embedding dimension
- Embedding lookup
- Tied embeddings (input/output weight sharing)

### 2. Positional Embeddings
- Integration of sinusoidal positional encodings
- Integration of learned positional embeddings
- Integration of RoPE (applied later in attention)
- Integration of ALiBi (applied as attention bias)

### 3. Complete Embedding Layer
- Token + Positional combination
- Dropout on embeddings
- Layer normalization (optional, used in some models)
- Scaling (used in original Transformer)

## 📊 Key Insights

### Why Embeddings?

Transformers work with **continuous vector representations**, not discrete tokens.

**Token Embeddings**: Map discrete token IDs to dense vectors
```
Token ID 42 → [0.1, -0.3, 0.5, ..., 0.2]  (d_model dimensions)
```

**Positional Embeddings**: Add position information
```
Position 3 → [0.0, 0.8, -0.2, ..., 0.4]  (d_model dimensions)
```

**Combined**: Token meaning + position information
```
Input = TokenEmbed(token) + PositionalEmbed(position)
```

### Embedding Dimensions

Typical sizes:
- **GPT-2 Small**: vocab=50257, d_model=768
- **GPT-2 Medium**: vocab=50257, d_model=1024
- **BERT Base**: vocab=30522, d_model=768
- **LLaMA**: vocab=32000, d_model=4096

Parameters in embedding table:
```
vocab_size × d_model = 50257 × 768 = ~38.6M parameters
```

Often the **largest single component** in smaller models!

### Initialization Strategies

**Xavier/Glorot** (most common):
```python
std = 1 / sqrt(d_model)
embeddings ~ N(0, std²)
```

**Normal** (GPT-2):
```python
std = 0.02
embeddings ~ N(0, 0.02²)
```

**Uniform**:
```python
limit = sqrt(3 / d_model)
embeddings ~ U(-limit, limit)
```

### Tied Embeddings

Share weights between:
1. Input embedding layer
2. Output projection layer (before softmax)

Benefits:
- Reduces parameters (vocab_size × d_model saved)
- Improves performance (empirically)
- Used in: GPT-2, BERT, T5, most modern LLMs

Constraint: d_model must equal output dimension

### Scaling

**Original Transformer** (Vaswani et al., 2017):
```python
embeddings = embeddings * sqrt(d_model)
```

Reason: Positional embeddings have variance ~1, token embeddings also ~1.  
After addition, variance increases. Scaling helps balance gradients.

**Modern LLMs**: Often don't scale (use layer norm instead)

## 🔬 Implementations

### Files

```
07-embeddings/
├── README.md                          # This file
├── embedding_demo.py                  # Complete embedding layer demo
├── tied_embeddings_demo.py            # Tied embeddings explanation
└── tests/
    └── test_embeddings.py            # Test suite
```

## 🚀 Usage

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals

# Demo complete embedding layer
python 07-embeddings/embedding_demo.py

# Tied embeddings demo
python 07-embeddings/tied_embeddings_demo.py

# Run tests
pytest 07-embeddings/tests/
```

## 📐 Math

### Token Embedding Lookup

```
Given token IDs: [5, 42, 100]
Embedding table E: [vocab_size, d_model]

Lookup:
  token_5_embedding = E[5, :]    # Row 5
  token_42_embedding = E[42, :]  # Row 42
  token_100_embedding = E[100, :]  # Row 100

Result shape: [3, d_model]
```

### Combined Embeddings

**Standard** (GPT-2, BERT):
```
Input = TokenEmbed(tokens) + PositionalEmbed(positions)
Input = Dropout(Input)
```

**With Scaling** (Original Transformer):
```
Input = TokenEmbed(tokens) * sqrt(d_model) + PositionalEmbed(positions)
```

**With LayerNorm** (Some variants):
```
Input = LayerNorm(TokenEmbed(tokens) + PositionalEmbed(positions))
```

### Tied Embeddings

```
# Input embedding
hidden = InputEmbedding(tokens)  # Shape: [batch, seq_len, d_model]

# ... transformer layers ...

# Output projection (tied weights)
logits = hidden @ InputEmbedding.weight.T  # Shape: [batch, seq_len, vocab_size]
```

Instead of separate output projection matrix, reuse input embedding matrix transposed.

## 🎨 Visualizations

1. **Embedding Space**: t-SNE/UMAP of token embeddings
2. **Positional Patterns**: Visualize positional embedding values
3. **Combined Effect**: Show token + positional for sample sequence

## 🧪 Experiments

1. **Initialization Comparison**: Xavier vs Normal vs Uniform
2. **Scaling Impact**: With/without sqrt(d_model) scaling
3. **Positional Encoding Comparison**: Sinusoidal vs Learned vs RoPE
4. **Tied vs Untied**: Compare parameter count and performance

## 📚 References

- "Attention Is All You Need" (Vaswani et al., 2017) - Embedding scaling
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2 embeddings
- "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017) - Tied embeddings
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)

## 🎯 Learning Outcomes

After this project, you'll understand:
- How token IDs are converted to continuous vectors
- How positional information is added
- Why embeddings are often the largest parameter group
- Tied embeddings and why they work
- Different initialization and scaling strategies
- How to combine token and positional embeddings

