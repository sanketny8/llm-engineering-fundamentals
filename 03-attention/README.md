# Project 03: Self-Attention & Multihead Attention

## 🎯 Learning Objective

Understand and implement the **attention mechanism** - the core innovation that powers transformers and modern LLMs.

## 🤔 Why This Matters

**Attention Is All You Need** (Vaswani et al., 2017) - This single mechanism replaced RNNs and became the foundation for:
- GPT (all variants)
- BERT, RoBERTa, ALBERT
- T5, BART, mBART
- Llama, PaLM, Claude
- Every modern LLM!

**Real-World Impact**:
- Enables parallel processing (vs sequential RNNs)
- Captures long-range dependencies
- Powers translation, summarization, coding, chat
- Multi-head attention adds diversity and specialization

## 📊 What You'll Build

1. **Scaled Dot-Product Attention** (hand-coded from scratch)
2. **Multi-Head Attention** (GPT-style with 12+ heads)
3. **Causal Masking** (for autoregressive generation)
4. **Attention Visualizer** (see what the model "looks at")
5. **Pattern Analyzer** (detect induction heads, copying behavior)

## 🚀 Quick Start

```bash
cd llm-engineering-fundamentals
source .venv/bin/activate

# Run attention demo
python 03-attention/attention_demo.py

# Visualize attention weights
python 03-attention/visualize_attention.py

# Test suite
pytest 03-attention/tests/ -v
```

## 🧮 The Math (Simplified)

### Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): What am I looking for?
- K (Key): What do I contain?
- V (Value): What information do I hold?
- d_k: Dimension of keys (for scaling)
```

### Multi-Head Attention

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Why multiple heads?**
- Different heads learn different patterns
- Some focus on syntax, some on semantics
- Some look locally, some globally
- Increases model capacity without huge param growth

## 📈 Expected Results

### Attention Patterns You'll See

1. **Diagonal Pattern**: Tokens attend to themselves
2. **Next Token**: GPT-style causal attention
3. **Induction Heads**: Detect and repeat patterns
4. **Copying**: Direct content copying
5. **Positional**: Attend based on distance

## 🔬 Key Experiments

### Experiment 1: Attention Without Mask
**Question**: What happens without causal masking?

**Result**: Future tokens leak information → cheating!

### Experiment 2: Single vs Multi-Head
**Question**: Why not just use one big head?

**Result**: Multi-head provides diversity & specialization

### Experiment 3: Attention Patterns
**Question**: What do different heads learn?

**Findings**:
- Head 0: Often focuses on previous token
- Head 5: May detect patterns/repetitions  
- Head 11: Often looks at first/last tokens

## 💡 Key Insights

### Why Dot Product?
- Measures similarity between query and key
- Efficient to compute (matrix multiplication)
- Differentiable (can backprop through it)

### Why Scale by √d_k?
- Prevents softmax saturation for large d_k
- Keeps gradients healthy
- Critical for training stability

### Why Softmax?
- Converts scores to probabilities
- Sum to 1 → proper attention weights
- Differentiable for gradient descent

### Causal Masking
- Prevents "cheating" in language modeling
- Each position can only attend to earlier positions
- Essential for GPT-style autoregressive models

## 📚 References

**Papers**:
1. Vaswani et al. (2017) - "Attention Is All You Need"
2. Elhage et al. (2021) - "A Mathematical Framework for Transformer Circuits"

**Models Using This**:
- GPT-2/3/4: 12-96 attention heads
- BERT: 12 heads (base), 16 (large)
- Llama 2: 32 heads (7B), 64 heads (70B)

---

**Status**: 🚀 Ready to implement!



