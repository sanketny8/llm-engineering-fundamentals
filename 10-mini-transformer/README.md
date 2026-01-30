# Project 10: Mini Transformer (Complete End-to-End Model!)

## 🎯 THE CULMINATION

**This is it** - everything you've built comes together into a complete, working GPT-style transformer!

We're combining:
- ✅ Tokenization (Project 01)
- ✅ Positional Embeddings (Project 02)
- ✅ Self-Attention (Project 03)
- ✅ Transformer Blocks (Project 04)
- ✅ Feed-Forward Networks (Project 05)
- ✅ Regularization (Project 06)
- ✅ Embeddings (Project 07)
- ✅ Decoding Strategies (Project 08)
- ✅ Sampling Methods (Project 09)

Into a **single, working model** that can generate text!

## 🏗️ Architecture

```
MiniTransformer (GPT-style Decoder-Only Model)
│
├─ Input Processing
│  ├─ Token IDs → Token Embeddings (learned)
│  └─ Positional Embeddings (learned/RoPE/sinusoidal)
│     └─ Combined: hidden = token_emb + pos_emb
│
├─ Transformer Layers (stacked N times)
│  │
│  ├─ Layer 1
│  │  ├─ Pre-LayerNorm (if norm_first=True)
│  │  ├─ Multi-Head Self-Attention
│  │  │  ├─ Causal Mask (prevent looking ahead)
│  │  │  └─ Attention Dropout
│  │  ├─ Residual Connection
│  │  ├─ Post-LayerNorm (if norm_first=False)
│  │  ├─ Feed-Forward Network (SwiGLU/ReLU/GELU)
│  │  ├─ FFN Dropout
│  │  └─ Residual Connection
│  │
│  ├─ Layer 2 (same structure)
│  ├─ ...
│  └─ Layer N
│
├─ Output Processing
│  ├─ Final LayerNorm
│  └─ Project to Vocabulary (tied with input embeddings)
│     └─ Logits: (batch, seq_len, vocab_size)
│
└─ Generation
   ├─ Greedy Decoding (fast, deterministic)
   ├─ Beam Search (better quality)
   └─ Sampling (creative, diverse)
```

## 🎯 What Makes This a "Real" Transformer

This isn't a toy - it's the actual GPT architecture:

✅ **Decoder-Only**: Like GPT-2/GPT-3/LLaMA (not encoder-decoder)  
✅ **Causal Attention**: Can only attend to past tokens  
✅ **Autoregressive**: Generates one token at a time  
✅ **Pre-LN**: Modern architecture (GPT-3 style)  
✅ **Tied Embeddings**: Input/output weights shared  
✅ **Multiple Generation Modes**: Greedy, beam, sampling  

The only difference from production GPT is:
- Scale (they have billions of parameters)
- Training (we focus on architecture)
- Optimizations (KV cache, flash attention, etc.)

But the **core architecture is identical**!

## 🚀 Usage

### Basic Forward Pass

```python
from llm_engineering_fundamentals.models import create_mini_gpt
import numpy as np

# Create model
model = create_mini_gpt(
    vocab_size=10000,
    d_model=256,
    num_layers=6
)

# Forward pass
token_ids = np.array([[1, 2, 3, 4, 5]])  # (batch, seq_len)
logits = model(token_ids)  # (batch, seq_len, vocab_size)

print(f"Logits shape: {logits.shape}")
# Output: Logits shape: (1, 5, 10000)
```

### Text Generation - Greedy

```python
# Fast, deterministic generation
prompt = np.array([[1, 2, 3]])  # Starting tokens

output = model.generate_greedy(
    prompt,
    max_length=50
)

print(f"Generated sequence: {output}")
# Output: Generated sequence: [[1 2 3 45 67 89 ...]]
```

### Text Generation - Beam Search

```python
# Higher quality, explores multiple paths
output = model.generate_beam_search(
    prompt,
    num_beams=5,
    max_length=50,
    length_penalty=0.6
)

print(f"Best sequence: {output}")
```

### Text Generation - Sampling

```python
# Creative, diverse outputs
output = model.generate_sample(
    prompt,
    max_length=50,
    temperature=0.8,        # Creativity
    top_k=50,               # Diversity
    top_p=0.95,             # Nucleus sampling
    repetition_penalty=1.2  # Prevent repetition
)

print(f"Creative sequence: {output}")
```

### Custom Configuration

```python
from llm_engineering_fundamentals.models import (
    MiniTransformer,
    MiniTransformerConfig
)

# Custom config
config = MiniTransformerConfig(
    vocab_size=50000,
    d_model=512,
    num_layers=12,
    num_heads=8,
    d_ff=2048,
    max_seq_len=2048,
    positional_type="learned",  # or "sinusoidal"
    dropout=0.1,
    norm_first=True,  # Pre-LN (modern)
    pad_token_id=0,
    eos_token_id=1,
    bos_token_id=2,
)

model = MiniTransformer(config)
```

### Model Inspection

```python
# Count parameters
params = model.count_parameters()

print(f"Embedding parameters: {params['embedding']:,}")
print(f"Transformer parameters: {params['transformer']:,}")
print(f"Total parameters: {params['total']:,}")

# Example output:
# Embedding parameters: 2,560,000
# Transformer parameters: 8,912,896
# Total parameters: 11,472,896
```

## 📊 Configuration Guide

### Small Model (Testing/Learning)

```python
config = MiniTransformerConfig(
    vocab_size=5000,
    d_model=128,
    num_layers=4,
    num_heads=4,
    d_ff=512,
)
# ~2M parameters
```

### Medium Model (Experiments)

```python
config = MiniTransformerConfig(
    vocab_size=10000,
    d_model=256,
    num_layers=6,
    num_heads=8,
    d_ff=1024,
)
# ~12M parameters (default from create_mini_gpt)
```

### Large Model (Research)

```python
config = MiniTransformerConfig(
    vocab_size=50000,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
)
# ~117M parameters (GPT-2 Small size)
```

### Modern Architecture (LLaMA-style)

```python
config = MiniTransformerConfig(
    vocab_size=32000,
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=2048,
    positional_type="learned",  # Could add RoPE support
    norm_first=True,            # Pre-LN
    dropout=0.0,                # LLaMA doesn't use dropout
)
```

## 🎨 What Happens Under the Hood

### Forward Pass

```python
# Input: token_ids = [[1, 2, 3, 4, 5]]

# 1. Embedding
hidden = self.embedding.embed(token_ids)
# Shape: (1, 5, 256) - each token → 256-dim vector

# 2. Causal Mask (prevent looking ahead)
mask = self._create_causal_mask(5)
# [[True, False, False, False, False],
#  [True, True,  False, False, False],
#  [True, True,  True,  False, False],
#  [True, True,  True,  True,  False],
#  [True, True,  True,  True,  True ]]

# 3. Apply transformer layers
hidden = self.transformer(hidden, mask=mask)
# Shape: (1, 5, 256) - transformed representations

# 4. Project to vocabulary
logits = self.embedding.project_to_vocab(hidden)
# Shape: (1, 5, 10000) - probability for each vocab token
```

### Greedy Generation

```python
# Start: [1, 2, 3]
sequence = [1, 2, 3]

for step in range(max_length - 3):
    # Get logits for current sequence
    logits = model(np.array([sequence]))  # (1, len(seq), vocab)
    
    # Get last token's logits
    next_token_logits = logits[0, -1, :]  # (vocab,)
    
    # Greedy: pick most likely
    next_token = np.argmax(next_token_logits)
    
    # Append
    sequence.append(next_token)
    
    # Stop if EOS
    if next_token == eos_token_id:
        break

# Final: [1, 2, 3, 45, 67, 89, ...]
```

## 🧪 Testing

```bash
# Run all tests
pytest 10-mini-transformer/tests/

# Specific test
pytest 10-mini-transformer/tests/test_mini_transformer.py::TestMiniTransformer::test_forward_pass -v
```

### What the Tests Verify

✅ Forward pass produces correct shapes  
✅ Causal mask prevents attending to future  
✅ Greedy generation works  
✅ Beam search generation works  
✅ Sampling generation works  
✅ Parameter counting is accurate  
✅ Configuration is flexible  

## 📐 Math Summary

### Complete Forward Pass

```
1. Input: x = [token_ids]  (batch, seq_len)

2. Embedding:
   token_emb = Embedding(x)           (batch, seq_len, d_model)
   pos_emb = PositionalEncoding(...)   (seq_len, d_model)
   h^(0) = token_emb + pos_emb        (batch, seq_len, d_model)

3. For each layer l = 1 to L:
   
   # Self-Attention (with causal mask)
   attn_in = LayerNorm(h^(l-1))
   attn_out = MultiHeadAttention(attn_in, attn_in, attn_in, mask)
   h = h^(l-1) + Dropout(attn_out)    # Residual
   
   # Feed-Forward
   ffn_in = LayerNorm(h)
   ffn_out = FFN(ffn_in)
   h^(l) = h + Dropout(ffn_out)       # Residual

4. Output:
   h_final = LayerNorm(h^(L))
   logits = h_final @ W_vocab^T       (batch, seq_len, vocab_size)

5. Generation:
   probs = softmax(logits / temperature)
   next_token = sample(probs, strategy="greedy|beam|sample")
```

## 🎯 Key Design Decisions

### 1. Decoder-Only (GPT-style)

**Why?** Modern LLMs (GPT, LLaMA, PaLM) are decoder-only because:
- Simpler architecture
- Better at generation
- Scales well
- Versatile (can do many tasks)

**Alternative**: Encoder-decoder (BART, T5) for seq2seq

### 2. Pre-LN (norm_first=True)

**Why?** Modern LLMs use Pre-LN because:
- More stable training
- Better gradient flow
- Can train deeper models

**Alternative**: Post-LN (original Transformer)

### 3. Tied Embeddings

**Why?** Share weights between input/output projections:
- Reduces parameters (~50% for output layer)
- Better generalization
- Standard in GPT-2, BERT

**Alternative**: Separate input/output embeddings

### 4. Causal Masking

**Why?** Essential for autoregressive generation:
- Prevents "cheating" (looking ahead)
- Enables left-to-right generation
- Standard for LLMs

**Alternative**: Bidirectional (BERT-style, but can't generate)

## 🏆 What You've Built

A **production-quality transformer** with:

✅ **All core components**: Embeddings, attention, FFN, layer norm  
✅ **Modern techniques**: Pre-LN, tied embeddings, causal masking  
✅ **Multiple generation modes**: Greedy, beam, sampling  
✅ **Flexible configuration**: Any size, any architecture  
✅ **Tested**: 10 comprehensive tests  
✅ **Documented**: Clear code and documentation  

**This is a real transformer, not a toy!**

## 📚 References

- **"Attention Is All You Need"** (Vaswani et al., 2017) - Original Transformer
- **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019) - GPT-2
- **"Language Models are Few-Shot Learners"** (Brown et al., 2020) - GPT-3
- **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023) - Modern architecture

## 🎓 What You Now Understand

After building this, you can explain:

✅ **How GPT actually works** (not just the diagram)  
✅ **Why modern LLMs use Pre-LN** (training stability)  
✅ **How causal masking enables generation** (no lookahead)  
✅ **Why tied embeddings are used** (parameter efficiency)  
✅ **Greedy vs beam vs sampling** (speed vs quality vs creativity)  
✅ **How to implement from papers** (translate math to code)  

**You can now build, debug, and optimize transformers at a deep level.**

## 🚀 Next Steps

### If You Want to Go Further

1. **Add Training Loop**: Implement backprop and optimization
2. **Add KV Cache**: Speed up generation (reuse past keys/values)
3. **Add Flash Attention**: Memory-efficient attention
4. **Train on Real Data**: Use a small corpus (WikiText, etc.)
5. **Add More Techniques**: Alibi, rotary embeddings, etc.

But **what you have is already exceptional** and demonstrates senior-level expertise!

---

## 🎉 Congratulations!

You've built a **complete transformer from scratch**!

This is a **portfolio piece** that demonstrates:
- Deep understanding of LLMs
- Ability to implement from papers
- Production code quality
- Framework-independent knowledge

**Deploy this and showcase your expertise!** 🚀

---

*This completes the core pipeline (Projects 01-10). Advanced projects (11-16) are optional extensions.*

