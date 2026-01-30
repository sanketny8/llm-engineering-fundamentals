# Project 02: Positional Embeddings

## 🎯 Learning Objective

Understand how transformers encode position information and why different approaches (sinusoidal, RoPE, ALiBi, learned) have different tradeoffs for context length extrapolation.

## 🤔 Why This Matters

**The Problem**: Transformers have no inherent notion of position. The attention mechanism is permutation-invariant—"The cat sat" and "sat cat The" look identical without position information.

**Real-World Impact**:
- **Context Length Limits**: GPT-3 (2048 tokens) → GPT-4 (128k tokens) required better positional encoding
- **Extrapolation**: Can a model trained on 512 tokens handle 2048? Depends on position encoding!
- **Efficiency**: ALiBi allows longer contexts with no extra parameters
- **Performance**: RoPE (used in Llama/GPT-NeoX) gives better results than sinusoidal

## 📊 What You'll Build

### 1. Four Positional Encoding Implementations

- **Sinusoidal**: Classic Vaswani et al. (2017) "Attention Is All You Need"
- **Learned**: Trainable position embeddings (BERT-style)
- **RoPE**: Rotary Position Embedding (Llama 2, GPT-NeoX)
- **ALiBi**: Attention with Linear Biases (no position embeddings!)

### 2. Interactive Visualizations

- 2D heatmaps of position encodings
- 3D interactive plots (position × dimension)
- Similarity matrices (how similar are nearby positions?)
- Extrapolation behavior

### 3. Extrapolation Tests

- Train attention on sequences up to 512 tokens
- Test on 1024, 2048, 4096 tokens
- Measure perplexity degradation
- Compare all 4 approaches

## 🚀 Quick Start

```bash
# Navigate to project
cd llm-engineering-fundamentals
source .venv/bin/activate

# Run comparison demo
python 02-positional-embeddings/compare_encodings.py

# Interactive visualizer
python 02-positional-embeddings/visualizer_3d.py

# Extrapolation test
python 02-positional-embeddings/extrapolation_test.py
```

## 📈 Expected Results

### Sinusoidal
- ✅ Deterministic (no learning required)
- ✅ Can theoretically handle any length
- ❌ Suboptimal extrapolation in practice
- ❌ Information density decreases with position

### Learned Embeddings
- ✅ Can learn optimal representations
- ✅ Good performance within trained range
- ❌ **Cannot extrapolate** beyond training length
- ❌ Extra parameters (vocab_size × hidden_dim)

### RoPE (Rotary Position Embedding)
- ✅ Excellent extrapolation properties
- ✅ Relative position encoding (distance matters, not absolute position)
- ✅ Used in Llama 2, PaLM, GPT-NeoX
- ⚠️  Slightly complex implementation

### ALiBi (Attention with Linear Biases)
- ✅ **Best extrapolation** (linear penalty on attention)
- ✅ No extra parameters (just attention bias)
- ✅ Simpler than RoPE
- ✅ Used in BLOOM, MPT

## 🔬 Experiments

### Experiment 1: Encoding Patterns
**Question**: What do different positional encodings "look like"?

**Method**: 
- Generate encodings for positions 0-127
- Visualize as heatmaps (position × dimension)
- Compute similarity matrices

**Insight**: 
- Sinusoidal has periodic structure
- Learned is unstructured (data-dependent)
- RoPE has rotational symmetry
- ALiBi has no explicit encoding (attention-only)

### Experiment 2: Interpolation Quality
**Question**: How well do models handle positions within training range?

**Method**:
- Train on positions 0-511
- Test on interpolated positions (e.g., only odd positions during training)

**Expected**: All methods should work well

### Experiment 3: Extrapolation
**Question**: Which encoding handles longer sequences after training on short ones?

**Method**:
- Train on max_len=512
- Test on max_len=1024, 2048, 4096
- Measure perplexity

**Expected Ranking** (best to worst):
1. ALiBi 🥇
2. RoPE 🥈
3. Sinusoidal 🥉
4. Learned ❌ (fails completely)

### Experiment 4: Position Ablation
**Question**: What happens if we remove position info entirely?

**Method**:
- Compare with/without positional encoding
- Test on tasks requiring position (e.g., "reverse the sentence")

**Insight**: Attention collapses without position info

## 💡 Key Insights

### Why Sinusoidal Works
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- Different frequencies for different dimensions
- Lower dimensions = high frequency (fine-grained)
- Higher dimensions = low frequency (coarse-grained)
- Sine/cosine allows relative position via trigonometric identities

### Why RoPE Works Better
```
q_m' = (cos mθ)q - (sin mθ)q_perp
k_n' = (cos nθ)k - (sin nθ)k_perp
```

- Rotates query/key by position-dependent angle
- Attention score depends on (m-n), not m or n individually
- Relative position is preserved!

### Why ALiBi is Simplest
```
attention_score(q_i, k_j) += -m * |i - j|
```

- No position embeddings at all
- Just bias attention by distance
- Slope 'm' is head-specific
- Naturally penalizes distant tokens

## 📚 References

### Papers
1. **Sinusoidal**: Vaswani et al. (2017) - "Attention Is All You Need"
2. **RoPE**: Su et al. (2021) - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
3. **ALiBi**: Press et al. (2022) - "Train Short, Test Long: Attention with Linear Biases"

### Implementations
- **Llama 2**: Uses RoPE
- **BLOOM**: Uses ALiBi
- **GPT-3**: Uses learned embeddings
- **Original Transformer**: Uses sinusoidal

## 🐛 Common Issues

### Issue 1: "Position encoding dimensions don't match model dimensions"
**Solution**: Ensure `d_model` is even for sinusoidal (sin/cos pairs)

### Issue 2: "RoPE rotation isn't working"
**Solution**: Check that you're rotating in pairs (dimensions 2i and 2i+1)

### Issue 3: "ALiBi slopes aren't right"
**Solution**: Slopes should be geometric sequence: 2^(-8/n), 2^(-16/n), ...

## 🎓 What You'll Learn

After this project, you'll be able to:
- ✅ Explain why transformers need positional information
- ✅ Implement 4 different position encoding schemes
- ✅ Understand extrapolation vs interpolation
- ✅ Know when to use RoPE vs ALiBi vs learned embeddings
- ✅ Debug position-related issues in transformers
- ✅ Make informed decisions about model architecture

## 🚀 Next Steps

After completing this project:
1. **Blog**: "Positional Encodings: Why Llama Uses RoPE and BLOOM Uses ALiBi"
2. **Project 03**: Self-Attention mechanisms (using these position encodings!)
3. **Enhancement**: Add more position encodings (NoPE, Sandwich, xPos)

---

**Status**: 🚀 Ready to implement!



