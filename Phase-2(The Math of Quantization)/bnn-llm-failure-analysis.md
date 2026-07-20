# Why Binary Neural Networks Failed for Large Language Models: An Information-Theoretic Analysis

## Executive Summary

This investigation examines why early Binary Neural Networks (BNNs) like XNOR-Net (Rastegari et al., 2016) suffered catastrophic accuracy degradation when applied to Large Language Models (LLMs). The analysis reveals that the failure stems from fundamental information-theoretic constraints: **LLMs encode critical information in a small fraction of high-magnitude "outlier" weights that binary quantization destroys indiscriminately**. Unlike image classification tasks where XNOR-Net succeeded, language models exhibit a heavy-tailed weight distribution with structural outliers that carry disproportionate representational importance.

---

## 1. The Empirical Failure: From ImageNet Success to LLM Collapse

### 1.1 XNOR-Net's ImageNet Achievement

XNOR-Net (Rastegari et al., 2016, arXiv:1603.05279) demonstrated that binary neural networks could achieve competitive performance on ImageNet classification:

| Model | Top-1 Accuracy | Memory Savings |
|-------|----------------|----------------|
| Full-Precision AlexNet | 56.6% | baseline |
| Binary-Weight-Network AlexNet | 56.8% | 32x |
| XNOR-Net AlexNet | 44.2% | 32x + 58x speedup |
| BinaryNet AlexNet (baseline) | 27.9% | 32x |

The key innovations were:
1. **Optimal scaling factors**: $α^* = \frac{1}{n}\|W\|_{\ell_1}$ (average absolute weight magnitude)
2. **Architectural modifications**: Batch normalization before binarization, pooling after binary convolution
3. **Straight-Through Estimator (STE)**: Gradients pass through sign function unchanged for $|x| \leq 1$

### 1.2 Catastrophic Failure on LLMs

When the same binarization techniques were applied to LLMs, results were devastating. Shang et al. (2023, arXiv:2310.00034) reported that **five renowned binarization methods produced models that performed worse than random guessing**:

| Method | BoolQ | PIQA | HellaSwag | Average |
|--------|-------|------|-----------|---------|
| Random Baseline | 0.50 | 0.50 | 0.25 | 0.36 |
| Full-Precision OPT-1.3B | 0.595 | 0.63 | 0.415 | 0.46 |
| BNN | 0.38 | 0.545 | 0.235 | 0.30 |
| XNOR | 0.37 | 0.525 | 0.265 | 0.31 |
| Bi-Real | 0.395 | 0.50 | 0.25 | 0.32 |

This is not merely degraded performance—it is **complete representational collapse**.

---

## 2. Information-Theoretic Foundation

### 2.1 Bits Per Weight: A Misleading Metric

A fundamental insight from information theory clarifies the failure. The **effective information capacity** per parameter is not determined by storage bits alone, but by the entropy of the weight distribution:

$$H(W) = -\sum_{w \in \mathcal{W}} p(w) \log_2 p(w)$$

For a well-trained neural network, the actual information content per weight can be far less than 16 or 32 bits. However, this does not mean all bits are equally dispensable.

### 2.2 The Heavy-Tailed Distribution Problem

Analysis of LLM weight distributions reveals a critical structural property: **weights follow a heavy-tailed distribution, not a Gaussian**.

Li et al. (2025, COLM) measured that in Llama models:
- **5% of outlier weights occupy approximately 50% of the total value range**
- Outlier positions are uniformly distributed across each output channel
- These outliers cannot be grouped away—small groups still contain outliers

This means the quantization step size in uniform quantization is dominated by outliers:

$$\Delta = \frac{x_{max} - x_{min}}{2^b - 1}$$

When $x_{max}$ is 100× larger than typical values, the step size forces all normal weights into a handful of discrete levels, destroying precision where it matters most.

### 2.3 The Information Bottleneck Interpretation

Sharpee (2017, PMC) established that optimal discretization in neural systems depends on **matching threshold distributions to input distributions**. The optimal quantization strategy must:

1. Allocate more quantization levels to high-probability regions
2. Handle the "bulk" and "tail" of distributions differently
3. Preserve information through distributed coding when noise is low

Binary quantization $\{+1, -1\}$ violates all three principles for LLMs:
- It treats all weights with identical magnitude thresholds
- It has no mechanism for the heavy tail
- It assumes noise dominates (redundant coding), but LLM weights are highly structured

---

## 3. Representational Capacity Analysis

### 3.1 Shannon Capacity of Binary Weights

A binary weight matrix $W_B \in \{-1, +1\}^{m \times n}$ can represent at most:

$$|\mathcal{W}_{binary}| = 2^{mn}$$

distinct matrices. However, the **effective capacity** for representing learned functions is far more constrained.

Consider the linear transformation $Y = WX$ where $W \in \mathbb{R}^{m \times n}$. The rank and condition number of $W$ determine how information flows through the network. Binarization:

$$W_B = \text{sign}(W)$$

This operation has two catastrophic effects:

1. **Magnitude information loss**: The scaling factor $\alpha = \frac{1}{n}\|W\|_{\ell_1}$ preserves only the **average** magnitude, not individual weight importance. Weights with magnitude 10.0 and 0.1 receive identical binary representations.

2. **Rank destruction**: Binary matrices have highly constrained spectral properties. A matrix of all +1s has rank 1, making most dimensions collapse.

### 3.2 The Scaling Factor Insufficiency

XNOR-Net's optimal scaling factor:

$$\alpha^* = \frac{\sum |W_i|}{n} = \frac{1}{n}\|W\|_{\ell_1}$$

minimizes the L2 quantization error:

$$J(\alpha) = \|W - \alpha \cdot \text{sign}(W)\|_2^2$$

But this assumes **weights are exchangeable**—that the error contribution from each weight is equally important. In LLMs, this assumption catastrophically fails.

### 3.3 The Outlier Problem: Where Information Concentrates

Dettmers et al. (2022, arXiv:2208.07339) discovered that LLMs exhibit emergent outlier dimensions:
- Models above 6.7B parameters spontaneously develop feature dimensions with values 50–100× larger than others
- These outliers appear in consistent positions across all layers and inputs
- They are not noise—they are **systematic, functional components**

Sun et al. (2024, arXiv:2402.17762) identified "massive activations"—specific neurons with values 10,000–100,000× larger than neighbors:
- These appear at fixed positions (typically BOS and delimiter tokens)
- They serve as **attention sinks**, absorbing excess attention probability
- Removing them causes complete model collapse

**Critical insight**: In LLMs, information is **not uniformly distributed** across weights. A vanishingly small fraction of weights carry disproportionate representational importance.

---

## 4. Quantization Theory: Local vs. Global Objective Misalignment

### 4.1 The Layer-wise Optimization Trap

Xu et al. (2024, arXiv:2410.14570) demonstrated a fundamental misalignment between local and global optimization objectives in quantization:

- **Post-training quantization (PTQ)** minimizes layer-wise MSE: $\|Q(W)x - Wx\|^2$
- **Quantization-aware fine-tuning (QAFT)** minimizes global NLL loss: $\text{NLL}(x|f_{Q(W)})$

These objectives diverge dramatically at low precision:

| Method | Precision | Global NLL (lower is better) |
|--------|-----------|------------------------------|
| QAFT | int2 | ~8.5 |
| GPTQ (local MSE) | int2 | ~11.0 |
| QAFT | int4 | ~6.2 |
| GPTQ (local MSE) | int4 | ~6.5 |

The gap widens as precision decreases, demonstrating that **local error minimization does not preserve global function behavior**.

### 4.2 Loss Landscape Analysis

The loss landscape around pre-trained weights $w \in \mathbb{R}^D$ has a characteristic structure:

1. **Attractive basin**: A region of radius $R(w)$ where loss is near-minimal
2. **Quantization perturbation**: $\|\Delta w\| = \|w_{RTN} - w\|$ pushes weights outside this basin

**Key finding**: When $\|\Delta w\| \gg R(w)$, quantization pushes the model into a different basin entirely. The model can no longer reach its original functional behavior through local corrections.

Binary quantization produces $\|\Delta w\|$ that exceeds $R(w)$ by orders of magnitude, causing complete loss landscape disconnection.

---

## 5. Why Image Classification Succeeded but Language Modeling Failed

### 5.1 Architectural Differences

| Property | CNNs (ImageNet) | Transformers (LLMs) |
|----------|-----------------|---------------------|
| Weight distribution | Approximately Gaussian | Heavy-tailed with outliers |
| Information density | Uniform across weights | Concentrated in outliers |
| Layer coupling | Moderate (hierarchical) | Strong (residual streams) |
| Precision sensitivity | Lower (redundant features) | Higher (token dependencies) |

### 5.2 The Residual Stream Problem

Transformers accumulate information through residual streams:

$$h_{l+1} = h_l + \text{Layer}(h_l)$$

This creates a **compound sensitivity**: errors in early layers propagate and amplify through all subsequent layers. Small quantization errors compound across 32–96 layers.

LLMs also exhibit **error amplification through attention**:
- Attention scores must sum to 1.0 via softmax
- Outlier weights create attention sinks that stabilize this constraint
- Binarizing these weights destabilizes attention computation

### 5.3 The Vocabulary Bottleneck

LLMs must map continuous representations to discrete tokens across vocabularies of 50K–256K tokens. This requires:
- Fine-grained distinctions in high-dimensional space
- Preserving subtle semantic relationships
- Maintaining calibration across temperature-scaled softmax

Binary weights destroy the fine-grained structure necessary for these distinctions.

---

## 6. The Salient Weight Hypothesis

Shang et al. (2023) established that LLM quantization success requires **preserving salient weights**:

### 6.1 Salient Weight Identification

Weights can be ranked by importance using:
- **Magnitude**: $|w_i|$ (simple, effective)
- **Hessian sensitivity**: $w_i^2 / [H^{-1}]_{ii}$ (optimal, expensive)

The top 1–5% of weights by magnitude are disproportionately important.

### 6.2 Partially-Binarized Results

When only non-salient weights are binarized:

| Configuration | Perplexity (C4) | Bits/Weight |
|---------------|-----------------|-------------|
| Full-Precision | 16.07 | 16 |
| 50% Binarized, 50% Salient | 47.82 | ~5 |
| Full Binarization | 367.38 | 1 |

**Preserving even 2% of weights at full precision dramatically improves results**.

### 6.3 Information-Theoretic Interpretation

This suggests an **asymmetric information structure**:
- 95% of weights contribute "bulk" computation requiring ~1–2 bits each
- 5% of weights encode "precision" information requiring ~16 bits each

Binary quantization treats all weights uniformly, destroying both bulk and precision information simultaneously.

---

## 7. Modern Solutions: What Actually Works

### 7.1 BitNet: Training from Scratch with Binary Constraints

Wang et al. (2023, arXiv:2310.11453) and Ma et al. (2024, arXiv:2402.17764) showed that **training from scratch with binary constraints** succeeds where post-hoc binarization fails:

| Model | Hellaswag | Winogrande | Memory |
|-------|-----------|------------|--------|
| LLaMA-7B (FP16) | 76.1% | 70.0% | 14 GB |
| BitNet b1.58-7B | 74.3% | 69.7% | ~2 GB |

**Why it works**:
1. The model learns to distribute information **compatible** with binary constraints
2. No pre-existing outlier structure must be destroyed
3. Training dynamics adapt to the quantization structure

### 7.2 The Ternary Breakthrough

BitNet b1.58 uses ternary weights $\{-1, 0, +1\}$, requiring only $\log_2(3) \approx 1.58$ bits:

$$W_{ternary} = \text{round}\left(\frac{W}{\text{absmean}(W)}\right)$$

The addition of **zero** as a representable value is crucial:
- Enables sparsity (many weights become exactly zero)
- Preserves the ability to "turn off" dimensions
- Matches the distributional structure learned during pre-training

### 7.3 Mixed-Precision Approaches

Successful quantization methods preserve outliers at higher precision:

| Method | Strategy | Overhead |
|--------|----------|----------|
| LLM.int8() | FP16 for outlier channels | ~1 bit/weight |
| SpQR | Isolate outlier weights | ~1 bit/weight |
| ICQuant | Index coding for outliers | ~0.3 bits/weight |

The fact that **0.3 bits of additional overhead enables 2-bit quantization** proves that information is concentrated in a tiny fraction of weights.

---

## 8. Theoretical Synthesis

### 8.1 The Fundamental Theorem of LLM Quantization

The failure of BNNs for LLMs can be formalized as follows:

**Theorem**: Let $W \in \mathbb{R}^{m \times n}$ be a pre-trained LLM weight matrix with heavy-tailed distribution $p(w) \propto |w|^{-\alpha}$ for $|w| > w_0$. Let $Q_1$ be binary quantization and $Q_k$ be $k$-bit uniform quantization. Then:

$$\text{Error}(Q_1(W)) \gg \text{Error}(Q_k(W))$$

for any finite $k > 1$, where Error measures functional degradation (perplexity increase, accuracy drop).

**Intuition**: Binary quantization cannot represent the heavy tail. The error in outlier representation propagates through attention and residual connections, causing exponential error amplification across layers.

### 8.2 Information Density Function

Define the **information density** of weight $w_i$ as:

$$\rho(w_i) = \frac{\partial L}{\partial w_i} \cdot w_i$$

For LLMs:
- $\rho(w)$ is highly non-uniform
- Outliers have $\rho(w) \gg \bar{\rho}$
- Binary quantization treats all $w$ identically, maximizing information loss

### 8.3 The Capacity Gap

The representational capacity gap between binary and full-precision weights is not 16× or 32×—it is **unbounded** for certain function classes:

- **Representable by binary weights**: Linearly separable patterns, simple threshold functions
- **Not representable by binary weights**: Fine-grained semantic distinctions, attention with calibrated softmax, token embeddings in 50K-dimensional space

LLMs require function classes in the second category.

---

## 9. Conclusions

### 9.1 Summary of Failure Mechanisms

| Mechanism | Explanation |
|-----------|-------------|
| Heavy-tailed weights | 5% of weights occupy 50% of range |
| Outlier importance | Outliers carry disproportionate functional information |
| Residual amplification | Errors compound across 32–96 layers |
| Attention destabilization | Binary weights break attention sink structure |
| Local-global misalignment | Layer-wise MSE ≠ global NLL preservation |

### 9.2 Why BNN Research Shifted

The success of BNNs on ImageNet led to incorrect generalization:
- Assumption: All neural networks have redundant, uniformly distributed information
- Reality: LLMs have concentrated, non-uniform information structure

### 9.3 Implications for Future Compression

1. **Information structure matters more than bit count**: 1.58-bit ternary weights can outperform 2-bit uniform quantization when the structure matches the distribution.

2. **Training from scratch is viable**: BitNet demonstrates that extreme quantization is possible when the model learns within constraints.

3. **Outliers are features, not bugs**: The heavy-tailed weight distribution is not inefficiency—it is functional specialization that must be preserved or learned.

---

## Sources

### Primary Papers

1. Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks. ECCV 2016. arXiv:1603.05279

2. Shang, Y., Yuan, Z., Wu, Q., & Dong, Z. (2023). PB-LLM: Partially Binarized Large Language Models. arXiv:2310.00034

3. Xu, Z., Sharify, S., Yazar, W., Webb, T., & Wang, X. (2024). Understanding the Difficulty of Low-Precision Post-Training Quantization for LLMs. arXiv:2410.14570

4. Wang, H., et al. (2023). BitNet: Scaling 1-bit Transformers for Large Language Models. arXiv:2310.11453

5. Ma, S., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764

### Foundational Theory

6. Sharpee, T. (2017). Optimizing Neural Information Capacity through Discretization. PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC5634811/

7. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339

8. Sun, M., et al. (2024). Massive Activations in Large Language Models. arXiv:2402.17762

### Recent Advances

9. Li, X., Hanna, O., Fragouli, C., & Diggavi, S. (2025). ICQuant: Index Coding enables Low-bit LLM Quantization. COLM 2025.

10. Various authors (2025). The Outlier Problem in LLM Quantization. Medium. https://medium.com/@serahabensour/the-outlier-problem-how-a-few-rogue-numbers-are-breaking-llm-quantization-510a25fe2ed4
