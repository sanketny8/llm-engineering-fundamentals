# Project 06: Dropout & Regularization

## 🎯 Concept

Understand regularization techniques that prevent overfitting and improve generalization in transformer models.

## 🏗️ What We'll Build

### 1. Dropout Variants
- **Standard Dropout**: Random neuron dropping
- **Attention Dropout**: Dropout on attention weights
- **DropPath** (Stochastic Depth): Drop entire layers
- **Embedding Dropout**: Dropout on token embeddings

### 2. Regularization Techniques
- **Weight Decay** (L2 regularization)
- **Gradient Clipping**: Prevent exploding gradients
- **Label Smoothing**: Soften target distributions

### 3. Analysis & Visualization
- Dropout mask visualization
- Impact on training dynamics
- Generalization gap analysis
- Optimal dropout rate exploration

## 📊 Key Insights

### Why Dropout?

**Problem**: Deep networks memorize training data  
**Solution**: Randomly "drop" neurons during training

Benefits:
- Prevents co-adaptation of neurons
- Creates ensemble effect (2^n sub-networks)
- Improves generalization
- Acts as regularization

### Dropout Evolution

**Standard Dropout** (Hinton et al., 2012):
```python
if training:
    mask = binomial(1, keep_prob, shape)
    output = (x * mask) / keep_prob  # Scale to maintain expectation
```

**Attention Dropout** (Transformer, 2017):
```python
attn_weights = softmax(scores)
if training:
    attn_weights = dropout(attn_weights)
output = attn_weights @ value
```

**DropPath/Stochastic Depth** (Huang et al., 2016):
```python
# Randomly skip entire transformer blocks
if training and random() < drop_prob:
    return x  # Skip this block
else:
    return x + block(x)
```

### Where to Apply Dropout in Transformers?

1. **Attention Dropout**: On attention weights (after softmax)
2. **Residual Dropout**: After attention & FFN, before residual add
3. **Embedding Dropout**: On input embeddings
4. **FFN Dropout**: Inside feed-forward network

Typical rates:
- GPT-2/3: 0.1 (10%)
- BERT: 0.1 (10%)
- LLaMA: 0.0 (no dropout, uses other regularization)

### Weight Decay vs Dropout

**Weight Decay (L2 Regularization)**:
- Penalizes large weights
- Loss = CrossEntropy + λ * ||W||²
- Prevents any single weight from dominating

**Dropout**:
- Randomly zeroes activations
- Forces redundancy
- Ensemble effect

Modern LLMs often use **both** or prefer weight decay.

## 🔬 Implementations

### Files

```
06-dropout-regularization/
├── README.md                          # This file
├── dropout_demo.py                    # Dropout variants demo
├── regularization_demo.py             # Weight decay & clipping
├── droppath_demo.py                   # Stochastic depth
└── tests/
    └── test_dropout.py               # Test suite
```

## 🚀 Usage

```bash
cd /Users/sanketny8/Desktop/MyGithub/llm-engineering-fundamentals

# Demo dropout variants
python 06-dropout-regularization/dropout_demo.py

# Regularization techniques
python 06-dropout-regularization/regularization_demo.py

# DropPath / Stochastic Depth
python 06-dropout-regularization/droppath_demo.py

# Run tests
pytest 06-dropout-regularization/tests/
```

## 📐 Math

### Standard Dropout

During training:
```
y = x ⊙ m / p

where:
- m ~ Bernoulli(p) is the dropout mask
- p is keep probability (1 - drop_rate)
- Division by p maintains expectation: E[y] = E[x]
```

During inference:
```
y = x  (no dropout, scaling already handled)
```

### Attention Dropout

```
scores = (Q @ K^T) / √d_k
attn_weights = softmax(scores)

if training:
    attn_weights = dropout(attn_weights, p)

output = attn_weights @ V
```

### DropPath (Stochastic Depth)

```
Block survival probability: p(l) = 1 - l/L * (1 - p_L)

where:
- l: current layer
- L: total layers
- p_L: final layer survival probability
- Linear decay: deeper layers more likely to be dropped
```

### Weight Decay

```
Loss = L_task + λ * Σ(W²)

Gradient update:
W ← W - η(∇L_task + 2λW)
W ← W - ηg - 2ληW
W ← (1 - 2ηλ)W - ηg  ← Decay factor
```

### Gradient Clipping

```
if ||g|| > threshold:
    g ← g * (threshold / ||g||)

Prevents exploding gradients in deep networks.
```

## 🎨 Visualizations

1. **Dropout Masks**: Show which neurons are dropped
2. **Training Curves**: With/without dropout
3. **Generalization Gap**: Train vs validation loss
4. **DropPath Schedule**: Layer survival rates

## 🧪 Experiments

1. **Dropout Rate Search**: Test 0.0, 0.1, 0.2, 0.3, 0.5
2. **Dropout Placement**: Compare different dropout locations
3. **DropPath vs Standard**: Measure effectiveness
4. **Weight Decay Sweep**: Find optimal λ

## 📚 References

- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (Srivastava et al., 2014)
- "Deep Networks with Stochastic Depth" (Huang et al., 2016)
- "Attention Is All You Need" (Vaswani et al., 2017) - Dropout in Transformers
- "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017) - AdamW
- "When Does Label Smoothing Help?" (Müller et al., 2019)

## 🎯 Learning Outcomes

After this project, you'll understand:
- How dropout prevents overfitting
- Different dropout variants and when to use them
- Why modern LLMs use less dropout
- Weight decay vs dropout trade-offs
- Gradient clipping for training stability
- Label smoothing for better calibration

