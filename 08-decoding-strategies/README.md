# Project 08: Decoding Strategies

## 🎯 Concept

Implement different decoding strategies for generating text from transformer models. Decoding is the process of selecting the next token given the model's output probabilities.

## 🏗️ What We'll Build

### 1. Greedy Decoding
- Select highest probability token at each step
- Deterministic and fast
- Used for: Simple generation, testing

### 2. Beam Search
- Maintain top-k hypotheses
- Balance between exploration and exploitation
- Used for: Translation, summarization

### 3. Decoding Loop
- Autoregressive generation
- Stop conditions (max length, EOS token)
- Token-by-token generation

## 📊 Key Insights

### Why Decoding Matters

Given model output logits `[vocab_size]`, how do we choose the next token?

**Problem**: Naive argmax (greedy) can lead to:
- Repetitive text
- Getting stuck in local optima
- No way to recover from mistakes

**Solution**: Different decoding strategies balance:
- Quality vs Speed
- Diversity vs Coherence
- Exploration vs Exploitation

### Greedy Decoding

```
At each step:
1. Get logits from model
2. next_token = argmax(logits)
3. Append to sequence
4. Repeat
```

**Pros:**
- Fast (O(1) per step)
- Deterministic
- Simple to implement

**Cons:**
- Can produce repetitive text
- Gets stuck in local optima
- No way to backtrack

**Use cases:** Testing, simple generation, when speed is critical

### Beam Search

```
Maintain B beams (hypotheses):
1. For each beam, get top-k tokens
2. Create B × k new hypotheses
3. Keep top-B by cumulative score
4. Repeat until all beams end
```

**Pros:**
- Better quality than greedy
- Explores multiple paths
- Can recover from suboptimal choices

**Cons:**
- Slower (O(B) per step)
- Can still be repetitive
- Beam size is a hyperparameter

**Use cases:** Translation, summarization, when quality matters

**Beam Width:**
- B=1: Equivalent to greedy
- B=5: Common for translation
- B=10-20: Diminishing returns

### Length Normalization

Raw beam search favors shorter sequences (fewer multiplications = higher probability).

**Length Penalty:**
```
score = log_prob / (length^alpha)

where:
- alpha = 0.0: No normalization (favors shorter)
- alpha = 0.6: Moderate (common for translation)
- alpha = 1.0: Full normalization
```

## 🔬 Implementations

### Files

```
08-decoding-strategies/
├── README.md                          # This file
├── decoding_demo.py                   # Demo all strategies
├── beam_search_demo.py                # Detailed beam search
└── tests/
    └── test_decoding.py              # Test suite
```

## 🚀 Usage

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals

# Demo all decoding strategies
python 08-decoding-strategies/decoding_demo.py

# Detailed beam search demo
python 08-decoding-strategies/beam_search_demo.py

# Run tests
pytest 08-decoding-strategies/tests/
```

## 📐 Math

### Greedy Decoding

```
At step t:
  logits_t = model(x_0, ..., x_{t-1})
  x_t = argmax(logits_t)

Continue until x_t == EOS or t == max_len
```

### Beam Search

```
Initialize: beams = [([], 0.0)]  # (sequence, score)

At step t:
  candidates = []
  for seq, score in beams:
    logits = model(seq)
    top_k_tokens, top_k_logprobs = topk(logits, k)
    
    for token, logprob in zip(top_k_tokens, top_k_logprobs):
      new_seq = seq + [token]
      new_score = score + logprob
      candidates.append((new_seq, new_score))
  
  # Keep top B candidates
  beams = topk(candidates, B, key=lambda x: x[1])

Return: Best beam
```

### Length Normalization

```
normalized_score = score / (length^alpha + epsilon)

Common values:
- alpha = 0.6 (Google NMT)
- alpha = 0.0 (no normalization)
- alpha = 1.0 (full normalization)
```

## 🎨 Visualizations

1. **Beam Search Tree**: Show how beams branch and merge
2. **Score Evolution**: Track beam scores over time
3. **Token Selection**: Visualize probability distributions

## 🧪 Experiments

1. **Greedy vs Beam**: Compare quality and speed
2. **Beam Width**: Test B=1,2,5,10,20
3. **Length Penalty**: Test alpha=0.0,0.6,1.0
4. **Early Stopping**: Compare different stopping criteria

## 📚 References

- "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
- "Google's Neural Machine Translation System" (Wu et al., 2016) - Length normalization
- "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) - Problems with beam search

## 🎯 Learning Outcomes

After this project, you'll understand:
- How to generate text from transformer models
- Greedy vs beam search trade-offs
- Why length normalization is needed
- How to implement autoregressive generation
- Stopping criteria and sequence completion

