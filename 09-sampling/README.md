# Project 09: Temperature & Sampling

## 🎯 Concept

Implement sampling methods for creative text generation. While decoding strategies (Project 08) focus on finding the "best" sequence, sampling methods introduce controlled randomness for more diverse and creative outputs.

## 🏗️ What We'll Build

### 1. Temperature Sampling
- Scale logits to control randomness
- Low temperature → more deterministic
- High temperature → more diverse

### 2. Top-k Sampling
- Sample from top-k most likely tokens
- Filter out unlikely options
- Used in: GPT-2, GPT-3

### 3. Top-p (Nucleus) Sampling
- Sample from smallest set with cumulative probability ≥ p
- Dynamic cutoff based on distribution
- Used in: GPT-3, modern LLMs

### 4. Repetition Penalty
- Reduce probability of already-generated tokens
- Prevent repetitive outputs
- Common in dialogue systems

## 📊 Key Insights

### The Problem with Greedy/Beam Search

**Greedy and beam search are deterministic** - they always produce the same output for the same input. This causes:

1. **Boring outputs**: No creativity or variation
2. **Repetition**: Gets stuck in loops
3. **Generic responses**: Always picks "safe" options

**Human language is creative and diverse** - we don't always pick the most probable next word!

### The Solution: Sampling

Instead of picking the **most likely** token, **sample** from the probability distribution.

But naive sampling has problems:
- **Too random**: Can generate nonsense
- **Need control**: Balance creativity with coherence

### Temperature Scaling

Control randomness by scaling logits before sampling:

```
scaled_logits = logits / temperature

then: probs = softmax(scaled_logits)
```

**Effect**:
- `temperature = 0.0` → Greedy (argmax)
- `temperature < 1.0` → More confident (peaked distribution)
- `temperature = 1.0` → Original distribution
- `temperature > 1.0` → More random (flatter distribution)

**Example**:

```
Original probs: [0.7, 0.2, 0.08, 0.02]

Temperature 0.5 (less random):  [0.85, 0.12, 0.025, 0.005]
Temperature 1.0 (original):     [0.7, 0.2, 0.08, 0.02]
Temperature 2.0 (more random):  [0.55, 0.25, 0.15, 0.05]
```

### Top-k Sampling

**Problem**: Sampling from full vocabulary can still produce nonsense.

**Solution**: Only sample from top-k most likely tokens.

```
1. Sort tokens by probability
2. Keep only top-k tokens
3. Renormalize probabilities
4. Sample from this smaller set
```

**Benefits**:
- Filters out very unlikely tokens
- Maintains diversity within reasonable options
- k=50 is common (GPT-2 default)

**Drawback**: Fixed k doesn't adapt to distribution shape

### Top-p (Nucleus) Sampling

**Problem**: Fixed k doesn't work well for all distributions.

Sometimes:
- Distribution is peaked → k=50 includes many bad options
- Distribution is flat → k=50 excludes many good options

**Solution**: Adaptive cutoff based on cumulative probability.

```
1. Sort tokens by probability (descending)
2. Find smallest set with cumulative probability ≥ p
3. Sample from this "nucleus"
```

**Benefits**:
- Adapts to distribution shape
- Peaked distribution → smaller nucleus
- Flat distribution → larger nucleus
- p=0.9 or p=0.95 are common

**Used in**: GPT-3, modern dialogue systems

### Repetition Penalty

**Problem**: Models often repeat the same phrases.

**Solution**: Penalize tokens that have already been generated.

```
For each previous token:
  if logit > 0:
    logit = logit / penalty
  else:
    logit = logit * penalty
```

**Common values**: 1.0-1.5 (1.0 = no penalty, 1.5 = strong penalty)

## 🔬 Implementations

### Files

```
09-sampling/
├── README.md                          # This file
├── sampling_demo.py                   # Demo all methods
├── temperature_demo.py                # Temperature deep dive
└── tests/
    └── test_sampling.py              # Test suite
```

## 🚀 Usage

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals

# Demo all sampling methods
python 09-sampling/sampling_demo.py

# Temperature deep dive
python 09-sampling/temperature_demo.py

# Run tests
pytest 09-sampling/tests/
```

## 📐 Math

### Temperature Scaling

```
Original logits: z = [z_1, z_2, ..., z_V]

Scaled logits: z' = z / τ  (where τ = temperature)

Probabilities: p_i = exp(z'_i) / Σ exp(z'_j)
```

**Effect on entropy**:
- Low τ → Low entropy (peaked distribution)
- High τ → High entropy (flat distribution)

### Top-k Sampling

```
1. Sort: indices = argsort(logits, descending=True)
2. Keep: top_k_indices = indices[:k]
3. Filter: top_k_logits = logits[top_k_indices]
4. Softmax: probs = softmax(top_k_logits / temperature)
5. Sample: token_idx = sample(probs)
6. Map back: token = top_k_indices[token_idx]
```

### Top-p (Nucleus) Sampling

```
1. Get probabilities: probs = softmax(logits / temperature)
2. Sort descending: sorted_probs, sorted_indices = sort(probs)
3. Cumulative sum: cumsum = cumsum(sorted_probs)
4. Find cutoff: cutoff = searchsorted(cumsum, p) + 1
5. Nucleus: nucleus_probs = sorted_probs[:cutoff]
6. Renormalize: nucleus_probs = nucleus_probs / sum(nucleus_probs)
7. Sample: token_idx = sample(nucleus_probs)
8. Map back: token = sorted_indices[token_idx]
```

### Repetition Penalty

```
For token i that appeared before:
  
  if logit_i > 0:
    logit_i = logit_i / penalty
  else:
    logit_i = logit_i * penalty

Then apply temperature and sample as usual.
```

## 🎨 Visualizations

1. **Temperature Effect**: Show how distribution changes
2. **Top-k vs Top-p**: Compare fixed vs adaptive cutoffs
3. **Diversity Metrics**: Measure output diversity
4. **Repetition Analysis**: Track repeated n-grams

## 🧪 Experiments

1. **Temperature Sweep**: Test 0.1, 0.5, 1.0, 1.5, 2.0
2. **Top-k Sweep**: Test k=10, 20, 50, 100
3. **Top-p Sweep**: Test p=0.8, 0.9, 0.95, 1.0
4. **Combined Effects**: Temperature + top-p
5. **Repetition Penalty**: Test 1.0, 1.2, 1.5

## 📚 References

- **"Hierarchical Neural Story Generation"** (Fan et al., 2018) - Top-k sampling
- **"The Curious Case of Neural Text Degeneration"** (Holtzman et al., 2019) - Top-p (nucleus) sampling
- **"CTRL: A Conditional Transformer Language Model"** (Keskar et al., 2019) - Repetition penalty
- **GPT-2** (Radford et al., 2019) - Top-k=40 default
- **GPT-3** (Brown et al., 2020) - Nucleus sampling

## 🎯 Learning Outcomes

After this project, you'll understand:
- How to control generation creativity
- Temperature scaling and its effects
- Top-k vs top-p trade-offs
- How modern LLMs generate diverse text
- When to use each sampling method

## 💡 Practical Guidelines

### When to Use Each Method

**Temperature Only**:
- Simple use case
- Want predictable behavior
- temperature=0.7-0.9 for balanced creativity

**Top-k (k=50)**:
- When you want diversity
- But need to avoid completely unlikely tokens
- Good for open-ended generation

**Top-p (p=0.95)**:
- Modern default (GPT-3)
- Adapts to context
- Best for dialogue and creative writing

**Combined (temperature + top-p)**:
- Most flexible
- temperature=0.8, top_p=0.95 is common
- Use in production systems

**Repetition Penalty**:
- Dialogue systems
- Long-form generation
- penalty=1.2 is typical

### Typical Settings

```python
# Deterministic (testing, classification)
temperature=0.0

# Balanced (most use cases)
temperature=0.8, top_p=0.95

# Creative (stories, poetry)
temperature=1.2, top_p=0.98

# Very creative (experimental)
temperature=1.5, top_k=100

# Dialogue (prevent repetition)
temperature=0.8, top_p=0.95, repetition_penalty=1.2
```

## 🔍 Common Pitfalls

1. **Too high temperature**: Generates nonsense
2. **Too low temperature**: Boring, repetitive
3. **Top-k too small**: Limited diversity
4. **Top-k too large**: Includes bad tokens
5. **Top-p too high**: May include unlikely tokens
6. **Top-p too low**: Too restrictive
7. **Repetition penalty too high**: Incoherent text

## 🎓 Advanced Topics

- **Typical Sampling**: Alternative to nucleus sampling
- **Contrastive Search**: Balance coherence and diversity
- **Mirostat**: Dynamic sampling to maintain target entropy
- **Classifier-Free Guidance**: Control generation with guidance

These are beyond the scope of this project but worth exploring!

---

**Next**: Combine everything in Project 10 (Mini Transformer) for complete text generation!

