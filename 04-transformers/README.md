# Project 04: Transformers, QKV, & Stacking

## 🎯 Concept

Build a complete Transformer block from scratch, dissecting the Query-Key-Value (QKV) mechanism and understanding how blocks stack to form deep networks like GPT.

## 🏗️ What We'll Build

### 1. Full Transformer Block
- Multi-head self-attention
- Feed-forward network (FFN)
- Layer normalization (Pre-LN vs Post-LN)
- Residual connections
- Dropout

### 2. QKV Dissection
- Separate Q, K, V projections
- Attention weight computation
- Output projection
- Visualization of attention patterns

### 3. Stacked Transformer
- 6-12 layer transformer
- Parameter sharing experiments
- Depth vs. width analysis
- Gradient flow visualization

### 4. Variants
- **Pre-LN** (GPT-3, LLaMA): LayerNorm before attention/FFN
- **Post-LN** (Original Transformer): LayerNorm after attention/FFN
- **Parallel** (GPT-J, PaLM): Attention and FFN in parallel

## 📊 Key Insights

### Why QKV?
- **Query**: "What am I looking for?"
- **Key**: "What do I offer?"
- **Value**: "What do I actually contain?"

The dot product `Q·K^T` determines *where* to attend, and the result is used to weight the *values*.

### Why Residual Connections?
- Enable gradient flow in deep networks
- Allow identity mapping (layer can learn to do nothing)
- Stabilize training

### Why Layer Normalization?
- Stabilize activation distributions
- Reduce internal covariate shift
- Enable higher learning rates

### Pre-LN vs Post-LN
- **Post-LN** (original): More expressive but harder to train at scale
- **Pre-LN** (modern): Easier to train, used in GPT-3, LLaMA, etc.
- Pre-LN performs LN on the input to each sublayer, Post-LN on the output

## 🔬 Implementations

### Files

```
04-transformers/
├── README.md                          # This file
├── transformer_block.py               # Demo script
├── qkv_visualizer.py                  # Visualize Q, K, V matrices
├── stacking_demo.py                   # Compare 1-layer vs 12-layer
└── tests/
    └── test_transformer.py            # Test suite
```

## 🚀 Usage

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals

# Run transformer demo
python 04-transformers/transformer_block.py

# Visualize QKV
python 04-transformers/qkv_visualizer.py

# Compare stacking depths
python 04-transformers/stacking_demo.py

# Run tests
pytest 04-transformers/tests/
```

## 📐 Math

### Transformer Block

```
# Pre-LN (modern)
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-LN (original)
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

### Feed-Forward Network

```
FFN(x) = max(0, xW1 + b1)W2 + b2

Where:
- W1: [d_model, d_ff] (typically d_ff = 4 * d_model)
- W2: [d_ff, d_model]
```

### Parameter Count

For a single transformer block:
- Attention: 4 * d_model * d_model (Q, K, V, O projections)
- FFN: 2 * d_model * d_ff
- Layer norms: 4 * d_model (2 layer norms, each with scale and bias)

**Example:** GPT-3 (d_model=12288, d_ff=49152, 96 layers)
- Per layer: ~1.8B parameters
- Total: ~175B parameters

## 🎨 Visualizations

1. **QKV Attention Patterns**: Heatmaps showing what each head attends to
2. **Activation Flow**: Track how representations change through layers
3. **Gradient Magnitude**: Compare gradient flow in shallow vs deep networks
4. **Pre-LN vs Post-LN**: Loss curves comparing training stability

## 🧪 Experiments

1. **Depth Scaling**: Train 2, 4, 6, 12 layer models on same task
2. **Width Scaling**: Compare d_model=256, 512, 1024 at fixed depth
3. **FFN Ratio**: Test d_ff = 2x, 4x, 8x d_model
4. **Residual Ablation**: Remove residuals, see training collapse

## 📚 References

- "Attention Is All You Need" (Vaswani et al., 2017)
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- "GLU Variants Improve Transformer" (Shazeer, 2020)
- "GPT-3" (Brown et al., 2020)
- "LLaMA" (Touvron et al., 2023)

## 🎯 Learning Outcomes

After this project, you'll understand:
- How transformers actually work (not just the diagram)
- Why modern LLMs use Pre-LN
- Why FFN is 4x wider than d_model
- How gradients flow in deep networks
- Why residual connections are critical
- Trade-offs between depth and width


