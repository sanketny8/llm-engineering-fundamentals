# Project 05: Feedforward Networks & Activation Functions

## 🎯 Concept

Deep dive into the Feed-Forward Network (FFN) component of transformers and explore different activation functions used in modern LLMs.

## 🏗️ What We'll Build

### 1. Activation Functions
- **ReLU**: `max(0, x)` - Original Transformer
- **GELU**: Gaussian Error Linear Unit - GPT, BERT
- **SwiGLU**: Swish-Gated Linear Unit - LLaMA, PaLM
- **GeGLU**: GELU-Gated Linear Unit - T5, variants

### 2. FFN Architectures
- **Standard FFN**: Two linear layers with activation
- **Gated FFN**: SwiGLU/GeGLU with gating mechanism
- **Expert FFN**: Mixture of Experts (MoE) preview

### 3. Analysis & Comparison
- Activation landscape visualization
- Gradient flow comparison
- Performance benchmarks
- Memory and compute trade-offs

## 📊 Key Insights

### Why FFN?
The Feed-Forward Network provides:
- **Non-linearity**: Enables complex transformations
- **Capacity**: Typically 2/3 of model parameters
- **Position-wise**: Independent processing per token
- **Expansion**: Usually 4x d_model for more expressiveness

### Why 4x Expansion?
```
d_model = 512  → d_ff = 2048
Parameters: 512 * 2048 + 2048 * 512 ≈ 2.1M

Empirically found to be optimal trade-off between:
- Capacity (larger is better)
- Compute (smaller is faster)
- Memory (smaller uses less)
```

### Activation Function Evolution

**ReLU** (2017 - Original Transformer):
- Simple: `max(0, x)`
- Fast to compute
- Dead neurons problem (gradient = 0 for x < 0)

**GELU** (2018 - BERT, GPT-2/3):
- Smooth: `x * Φ(x)` where Φ is Gaussian CDF
- Better gradient flow
- Probabilistic interpretation

**SwiGLU** (2020 - LLaMA, PaLM):
- Gated: `(xW1) ⊗ σ(xW2) W3`
- 3 linear layers instead of 2
- Best empirical performance
- Used in modern LLMs

## 🔬 Implementations

### Files

```
05-feedforward-activations/
├── README.md                          # This file
├── activation_functions.py            # All activation implementations
├── ffn_architectures.py               # Different FFN variants
├── performance_comparison.py          # Benchmark activations
└── tests/
    └── test_ffn.py                   # Test suite
```

## 🚀 Usage

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals

# Compare activation functions
python 05-feedforward-activations/activation_functions.py

# Test FFN architectures
python 05-feedforward-activations/ffn_architectures.py

# Run performance benchmarks
python 05-feedforward-activations/performance_comparison.py

# Run tests
pytest 05-feedforward-activations/tests/
```

## 📐 Math

### Standard FFN
```
FFN(x) = max(0, xW1 + b1)W2 + b2

Where:
- W1: [d_model, d_ff]
- W2: [d_ff, d_model]
- Activation: ReLU (or GELU)
```

### SwiGLU FFN (LLaMA)
```
SwiGLU(x) = (Swish(xW1) ⊗ xW2)W3

Where:
- Swish(x) = x · σ(x)  (σ is sigmoid)
- ⊗ is element-wise multiplication
- W1, W2: [d_model, d_ff]
- W3: [d_ff, d_model]
- Note: 3 matrices instead of 2, but same compute
```

### GeGLU FFN (T5)
```
GeGLU(x) = (GELU(xW1) ⊗ xW2)W3

Similar to SwiGLU but uses GELU instead of Swish
```

## 🎨 Visualizations

1. **Activation Landscapes**: Plot activation functions and their gradients
2. **Dead Neuron Analysis**: Show ReLU dead zones vs GELU/SwiGLU
3. **Performance Comparison**: Speed, memory, accuracy trade-offs
4. **FFN Output Distribution**: Analyze output statistics

## 🧪 Experiments

1. **Activation Comparison**: Train small model with different activations
2. **Width Scaling**: Test d_ff = 2x, 4x, 8x d_model
3. **Gating Impact**: Compare standard vs gated FFN
4. **Gradient Flow**: Measure gradients through different activations

## 📚 References

- "Attention Is All You Need" (Vaswani et al., 2017) - ReLU FFN
- "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
- "GLU Variants Improve Transformer" (Shazeer, 2020) - SwiGLU
- "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
- "Switch Transformers" (Fedus et al., 2021) - MoE

## 🎯 Learning Outcomes

After this project, you'll understand:
- Why modern LLMs use GELU/SwiGLU instead of ReLU
- How gated activation functions work
- The capacity-compute trade-off in FFN design
- Why FFN has more parameters than attention
- Performance characteristics of different activations

