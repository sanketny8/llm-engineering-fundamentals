# 01 - Tokenization & Embeddings

This module teaches:

- How **byte pair encoding (BPE)** learns a subword vocabulary
- How text maps to **token IDs** (and why tokenization can be “weird”)
- The difference between **one-hot** and **learned embeddings**, using cosine distance

## Quick start

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Train a tiny BPE model and encode/decode

```bash
python 01-tokenization/bpe_from_scratch.py --train --merges 200 --out 01-tokenization/benchmarks/tiny_bpe.json
python 01-tokenization/bpe_from_scratch.py --encode --model 01-tokenization/benchmarks/tiny_bpe.json --text "hello tokenization"
python 01-tokenization/bpe_from_scratch.py --decode --model 01-tokenization/benchmarks/tiny_bpe.json --ids "0,1,2"
```

### Run the token visualizer

```bash
make token-viz
```

## Notes

- This is an **educational** BPE implementation aimed at clarity and correctness.
- Real production tokenizers add many optimizations (regex pre-tokenization, byte fallback, special tokens, etc.).




