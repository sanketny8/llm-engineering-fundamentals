# LLM Engineering Fundamentals

> **Building Transformers from Scratch**: A complete, production-ready implementation of modern LLM architectures using pure NumPy.

[![Tests](https://img.shields.io/badge/tests-170%20passing-brightgreen)]()
[![Code](https://img.shields.io/badge/code-5.6k%20LOC-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🎯 What Is This?

A **complete transformer implementation from first principles** - everything from tokenization to text generation, built using only NumPy. No PyTorch, no TensorFlow - just clean, understandable code that implements modern LLM techniques.

**170 passing tests** prove it works. **5,600 lines of code** document deep understanding. **10 complete projects** demonstrate production quality.

---

## ✨ Why This Matters

**99% of AI engineers use frameworks.** They call `model.generate()` without understanding what happens inside.

**You'll understand every detail.** You'll implement attention, positional encodings, beam search, and more - from scratch.

**This demonstrates senior-level expertise** that sets you apart in interviews, research, and production systems.

---

## 🏗️ What's Included

### Core Pipeline (10 Projects - All Complete ✅)

1. **[Tokenization & BPE](01-tokenization-bpe/)** - Byte-Pair Encoding from scratch
2. **[Positional Embeddings](02-positional-embeddings/)** - Sinusoidal, Learned, RoPE, ALiBi
3. **[Self-Attention](03-attention/)** - Multi-head attention with causal masking
4. **[Transformers](04-transformers/)** - Complete blocks with Pre-LN/Post-LN
5. **[Feed-Forward Networks](05-feedforward-activations/)** - ReLU, GELU, SwiGLU, GeGLU
6. **[Regularization](06-dropout-regularization/)** - Dropout, gradient clipping, weight decay
7. **[Embeddings](07-embeddings/)** - Token + positional combined
8. **[Decoding](08-decoding-strategies/)** - Greedy + beam search
9. **[Sampling](09-sampling/)** - Temperature, top-k, top-p (nucleus)
10. **[Mini Transformer](10-mini-transformer/)** - Complete GPT-style model

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/sanketny8/llm-engineering-fundamentals.git
cd llm-engineering-fundamentals

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Run tests (all 170 should pass!)
pytest

# Try the complete transformer
python examples/generate_text.py
```

---

## 📚 Projects

### 1. Tokenization & BPE
**20 tests ✅ | ~400 LOC**

Implement Byte-Pair Encoding from scratch. Learn how text becomes numbers.

```python
from llm_engineering_fundamentals.tokenization import BPETokenizer

tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(["hello world", "tokenization rocks"])

tokens = tokenizer.encode("hello tokenization")
print(tokens)  # [42, 17, 89]

text = tokenizer.decode(tokens)
print(text)  # "hello tokenization"
```

**Learn**: Subword tokenization, vocabulary building, merge rules

---

### 2. Positional Embeddings
**22 tests ✅ | ~600 LOC**

Four variants: Sinusoidal (Transformer), Learned (BERT), RoPE (LLaMA), ALiBi (BLOOM).

```python
from llm_engineering_fundamentals.positional import (
    sinusoidal_positional_encoding,
    RotaryPositionEmbedding,
)

# Sinusoidal (original Transformer)
pos_encoding = sinusoidal_positional_encoding(max_seq_len=512, d_model=256)

# RoPE (modern LLMs like LLaMA)
rope = RotaryPositionEmbedding(d_model=256, max_len=2048)
q_rot, k_rot = rope.apply(query, key, positions)
```

**Learn**: Why position matters, how RoPE enables length extrapolation

---

### 3. Self-Attention & Multi-Head
**22 tests ✅ | ~800 LOC**

The core mechanism that makes transformers work.

```python
from llm_engineering_fundamentals.attention import MultiHeadAttention

attention = MultiHeadAttention(d_model=256, num_heads=8)

output, attn_weights = attention(
    query, key, value,
    mask=causal_mask,
    return_attention=True
)
```

**Learn**: Scaled dot-product attention, Q/K/V, multi-head mechanism

---

### 4. Complete Transformers
**17 tests ✅ | ~900 LOC**

Full transformer blocks with layer norm, residuals, and stacking.

```python
from llm_engineering_fundamentals.transformer import StackedTransformer

model = StackedTransformer(
    num_layers=6,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    norm_first=True,  # Pre-LN (modern)
)

output = model(x, mask=causal_mask)
```

**Learn**: Pre-LN vs Post-LN, depth vs width, parameter scaling

---

### 5. Feed-Forward & Activations
**22 tests ✅ | ~800 LOC**

Standard FFN plus modern variants (SwiGLU, GeGLU).

```python
from llm_engineering_fundamentals.ffn import SwiGLUFFN

# Modern FFN used in LLaMA
ffn = SwiGLUFFN(d_model=256, d_ff=1024)
output = ffn(x)
```

**Learn**: Why LLaMA uses SwiGLU, activation function evolution

---

### 6. Dropout & Regularization
**22 tests ✅ | ~700 LOC**

Training stability techniques.

```python
from llm_engineering_fundamentals.regularization import (
    Dropout, DropPath, gradient_clip_norm, label_smoothing
)

dropout = Dropout(p=0.1)
droppath = DropPath(drop_prob=0.1)

clipped_grads = gradient_clip_norm(gradients, max_norm=1.0)
smooth_targets = label_smoothing(targets, num_classes=1000, smoothing=0.1)
```

**Learn**: Dropout variants, gradient clipping, label smoothing

---

### 7. Embeddings
**15 tests ✅ | ~400 LOC**

Token + positional embeddings with weight tying.

```python
from llm_engineering_fundamentals.embeddings import TiedEmbedding

embedding = TiedEmbedding(vocab_size=10000, d_model=256)

# Input embedding
hidden = embedding.embed(token_ids)

# Output projection (shared weights)
logits = embedding.project_to_vocab(hidden)
```

**Learn**: Tied embeddings, parameter sharing, embedding initialization

---

### 8. Decoding Strategies
**10 tests ✅ | ~300 LOC**

Generate text from model outputs.

```python
from llm_engineering_fundamentals.decoding import greedy_decode, beam_search

# Greedy (fast)
output = greedy_decode(model_fn, prompt, max_length=50)

# Beam search (better quality)
output = beam_search(model_fn, prompt, num_beams=5, length_penalty=0.6)
```

**Learn**: Greedy vs beam search, length normalization

---

### 9. Sampling Methods
**10 tests ✅ | ~300 LOC**

Creative text generation.

```python
from llm_engineering_fundamentals.sampling import sample_next_token

next_token = sample_next_token(
    logits,
    temperature=0.8,      # Creativity
    top_k=50,             # Diversity
    top_p=0.95,           # Nucleus sampling
    repetition_penalty=1.2  # Reduce repetition
)
```

**Learn**: Temperature scaling, top-k, top-p (nucleus) sampling

---

### 10. Mini Transformer
**10 tests ✅ | ~400 LOC**

**Complete end-to-end GPT-style model!**

```python
from llm_engineering_fundamentals.models import create_mini_gpt

# Create model
model = create_mini_gpt(vocab_size=10000, d_model=256, num_layers=6)

# Forward pass
logits = model(token_ids)

# Generate text (greedy)
output = model.generate_greedy(prompt, max_length=100)

# Generate text (sampling)
output = model.generate_sample(
    prompt, temperature=0.8, top_k=50, top_p=0.95
)

# Count parameters
params = model.count_parameters()
print(f"Total parameters: {params['total']:,}")
```

**This is it - a working transformer from scratch!**

---

## 🧪 Testing

All components are thoroughly tested:

```bash
# Run all tests
pytest

# Run specific project tests
pytest 01-tokenization-bpe/tests/
pytest 10-mini-transformer/tests/

# With coverage
pytest --cov=llm_engineering_fundamentals

# Verbose mode
pytest -v
```

**Result**: 170/170 tests passing ✅

---

## 📊 Modern Techniques

This isn't just the original 2017 Transformer - it includes cutting-edge techniques from modern LLMs:

| Technique | Used In | Project |
|-----------|---------|---------|
| **RoPE** | LLaMA, GPT-4 | Project 02 |
| **SwiGLU** | LLaMA, PaLM | Project 05 |
| **Pre-LN** | GPT-3, modern LLMs | Project 04 |
| **ALiBi** | BLOOM | Project 02 |
| **Tied Embeddings** | GPT-2, BERT, T5 | Project 07 |
| **Nucleus Sampling** | GPT-3 | Project 09 |

---

## 🎓 Learning Path

### Beginner → Start Here
1. **Project 01** - Understand how text becomes numbers
2. **Project 02** - Learn why position information matters
3. **Project 03** - Grasp the attention mechanism

### Intermediate → Build Understanding
4. **Project 04** - See how layers stack
5. **Project 05** - Understand feed-forward networks
6. **Project 06** - Learn training stability

### Advanced → Complete System
7. **Project 07** - Combine components
8. **Project 08** - Generate text deterministically
9. **Project 09** - Generate creatively
10. **Project 10** - **Put it all together!**

---

## 💡 Why From Scratch?

**Understanding > Using**

Most tutorials teach you to call `transformers.AutoModel`. This teaches you to **build** AutoModel.

**Benefits**:
- Debug production issues
- Optimize performance
- Innovate new architectures
- Interview confidently
- Research effectively

**Framework-independent knowledge** means you can work with any tool.

---

## 🏆 What Makes This Special

### 1. Production Quality
- ✅ 170 passing tests
- ✅ Type hints throughout
- ✅ Comprehensive documentation
- ✅ Clean, readable code
- ✅ Error handling

### 2. Modern Techniques
- ✅ RoPE (LLaMA, GPT-4)
- ✅ SwiGLU (LLaMA, PaLM)
- ✅ Pre-LN (GPT-3)
- ✅ ALiBi (BLOOM)
- ✅ Nucleus sampling

### 3. Complete
- ✅ Tokenization → Generation
- ✅ Multiple variants
- ✅ Training-ready
- ✅ Well-documented

### 4. From Scratch
- ✅ Pure NumPy
- ✅ No framework magic
- ✅ Understand every line

---

## 📖 Documentation

Each project has:
- **README**: Concepts, math, usage
- **Code**: Clean, commented implementation
- **Tests**: Comprehensive test suite
- **Demos**: Interactive examples

Plus:
- **[PROGRESS_SUMMARY.md](PROGRESS_SUMMARY.md)** - Complete status
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Detailed breakdown

---

## 🤝 Contributing

This is a learning project, but improvements are welcome!

Areas for contribution:
- Additional positional encoding variants
- More sampling strategies
- Training loop implementation
- Optimization techniques
- Documentation improvements

---

## 📝 Citation

If you use this in your research or learning, please cite:

```bibtex
@misc{llm_engineering_fundamentals,
  author = {Sanket Nyayadhish},
  title = {LLM Engineering Fundamentals: Transformers from Scratch},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/sanketny8/llm-engineering-fundamentals}
}
```

---

## 📚 References

### Papers Implemented
- **Attention Is All You Need** (Vaswani et al., 2017) - Original Transformer
- **BERT** (Devlin et al., 2018) - Learned positional embeddings
- **GPT-2** (Radford et al., 2019) - Language model pre-training
- **RoFormer** (Su et al., 2021) - Rotary Position Embedding
- **Train Short, Test Long** (Press et al., 2021) - ALiBi
- **GLU Variants** (Shazeer, 2020) - SwiGLU
- **LLaMA** (Touvron et al., 2023) - Modern architecture

### Additional Resources
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## 📬 Contact

**Sanket Nyayadhish**
- GitHub: [@sanketny8](https://github.com/sanketny8)
- X (Twitter): [@Ny8Sanket](https://x.com/Ny8Sanket)
- LinkedIn: [in/ny8sanket](https://www.linkedin.com/in/ny8sanket)

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🎉 Acknowledgments

Built with inspiration from:
- Andrej Karpathy's educational content
- The original Transformer paper authors
- The open-source ML community

---

**⭐ If this helped you understand transformers, please star the repo!**

**🚀 Ready to deploy? This demonstrates senior-level LLM engineering skills.**

---

*Last Updated: 2026-01-30*  
*Status: Complete core pipeline (10/10 projects)*  
*Tests: 170/170 passing ✅*
