# Provenance: BNN-LLM Failure Analysis

## Primary Academic Sources

### 1. XNOR-Net Original Paper
- **ID**: arXiv:1603.05279
- **Title**: XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
- **Authors**: Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
- **Venue**: ECCV 2016
- **URL**: https://arxiv.org/abs/1603.05279
- **AlphaXiv**: https://www.alphaxiv.org/overview/1603.05279
- **Key findings used**: 
  - Binary-Weight-Networks achieve 56.8% Top-1 on ImageNet (comparable to full-precision)
  - XNOR-Networks achieve 44.2% Top-1 (both weights and activations binary)
  - Outperforms BinaryNet by 16.3% absolute improvement
  - Optimal scaling factor: α* = (1/n)||W||_ℓ1

### 2. PB-LLM: Partially Binarized Large Language Models
- **ID**: arXiv:2310.00034
- **Title**: PB-LLM: Partially Binarized Large Language Models
- **Authors**: Yuzhang Shang, Zhihang Yuan, Qiang Wu, Zhen Dong
- **Venue**: 2023
- **URL**: https://arxiv.org/abs/2310.00034
- **AlphaXiv**: https://www.alphaxiv.org/overview/2310.00034
- **Key findings used**:
  - Five binarization methods perform worse than random on LLM benchmarks
  - Salient weights (top 1-5%) must be preserved at higher precision
  - 50% salient + 50% binarized achieves workable perplexity

### 3. Understanding the Difficulty of Low-Precision PTQ
- **ID**: arXiv:2410.14570
- **Title**: Understanding the Difficulty of Low-Precision Post-Training Quantization for LLMs
- **Authors**: Zifei Xu, Sayeh Sharify, Wanzin Yazar, Tristan Webb, Xin Wang
- **Venue**: 2024
- **URL**: https://arxiv.org/abs/2410.14570
- **AlphaXiv**: https://www.alphaxiv.org/overview/2410.14570
- **Key findings used**:
  - Local MSE minimization misaligned with global NLL optimization
  - Loss landscape analysis: quantization pushes weights outside attractive basin
  - QAFT outperforms GPTQ by large margin at low precision

### 4. BitNet: Scaling 1-bit Transformers
- **ID**: arXiv:2310.11453
- **Title**: BitNet: Scaling 1-bit Transformers for Large Language Models
- **Authors**: Hongyu Wang, Shuming Ma, Li Dong, et al.
- **Venue**: 2023
- **URL**: https://arxiv.org/abs/2310.11453
- **AlphaXiv**: https://www.alphaxiv.org/overview/2310.11453
- **Key findings used**:
  - Training from scratch with binary constraints succeeds
  - Maintains ~2-3 point gap from FP16 transformers
  - 8.8x to 38.8x energy efficiency improvement

### 5. BitNet b1.58: The Era of 1-bit LLMs
- **ID**: arXiv:2402.17764
- **Title**: The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
- **Authors**: Shuming Ma, Hongyu Wang, et al.
- **Venue**: 2024
- **URL**: https://arxiv.org/abs/2402.17764
- **AlphaXiv**: https://www.alphaxiv.org/overview/2402.17764
- **Key findings used**:
  - Ternary weights {-1, 0, +1} achieve FP16 performance parity
  - 71.4x arithmetic energy reduction
  - Zero value critical for sparsity and "turning off" dimensions

## Theoretical Foundations

### 6. Optimizing Neural Information Capacity
- **Source**: PMC/Neuron
- **Title**: Optimizing Neural Information Capacity through Discretization
- **Author**: Tatyana O. Sharpee
- **Year**: 2017
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5634811/
- **Key findings used**:
  - Optimal discretization depends on noise level
  - Low noise: distributed coding with staggered thresholds
  - High noise: redundant coding with identical thresholds
  - Binary quantization assumes high-noise regime

### 7. LLM.int8(): Outlier Discovery
- **ID**: arXiv:2208.07339
- **Title**: LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- **Authors**: Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Year**: 2022
- **URL**: https://arxiv.org/abs/2208.07339
- **Key findings used**:
  - Emergent outliers above 6.7B parameters
  - 0.1% of dimensions with 50-100× larger values
  - Outliers appear in consistent positions across all inputs
  - Phase transition behavior in model scale

### 8. Massive Activations in LLMs
- **ID**: arXiv:2402.17762
- **Title**: Massive Activations in Large Language Models
- **Authors**: Mingjie Sun, Xinlei Chen, J Zico Kolter, Zhuang Liu
- **Year**: 2024
- **URL**: https://arxiv.org/abs/2402.17762
- **Key findings used**:
  - Specific neurons with 10,000-100,000× larger activations
  - Fixed positions (BOS, delimiter tokens)
  - Serve as attention sinks
  - Input-agnostic (constant across all inputs)

## Recent Advances

### 9. ICQuant: Index Coding
- **Source**: COLM 2025 (via OpenReview PDF)
- **Title**: ICQuant: Index Coding enables Low-bit LLM Quantization
- **Authors**: Xinlin Li, Osama Hanna, Christina Fragouli, Suhas Diggavi
- **URL**: https://openreview.net/pdf?id=m6nBgFSMTL
- **Key findings used**:
  - 5% outliers occupy 50% of weight range
  - Outlier positions uniformly distributed
  - Index coding enables ~0.3 bits overhead
  - 2.3-bit quantization viable with outlier separation

### 10. LLM Quantization Outlier Analysis
- **Source**: Medium (synthesis of multiple papers)
- **Title**: The Outlier Problem: How a Few Rogue Numbers are Breaking LLM Quantization
- **Author**: Serah Tapia
- **Year**: 2025
- **URL**: https://medium.com/@serahabensour/the-outlier-problem-how-a-few-rogue-numbers-are-breaking-llm-quantization-510a25fe2ed4
- **Key findings used**:
  - Unified framework for outlier types (weight, activation, attention)
  - Softmax as root cause of outlier emergence
  - Context-aware scaling as solution
  - CushionCache approach for existing models

## Supplementary Sources

### Web Sources Used for Context
- Semantic Scholar XNOR-Net overview
- arXiv metadata for paper verification
- Research synthesis from multiple quantization surveys

## Verification Status

| Claim | Source | Verification |
|-------|--------|--------------|
| XNOR-Net ImageNet accuracy | arXiv:1603.05279 | Direct from paper |
| BNN failure on LLMs (worse than random) | arXiv:2310.00034 | Direct from paper Figure 2 |
| Local-global optimization misalignment | arXiv:2410.14570 | Direct from paper Figure 1 |
| 5% outliers = 50% range | ICQuant paper | Direct from paper Figure 1a |
| BitNet b1.58 FP16 parity | arXiv:2402.17764 | Direct from paper Table 2 |
| Massive activations 10,000-100,000× | arXiv:2402.17762 | Direct from paper |
| Phase transition at 6.7B params | arXiv:2208.07339 | Direct from paper |

## Methodology Notes

1. **Paper selection**: Used alpha_search for semantic paper discovery, prioritizing highly-cited foundational works and recent advances.

2. **Cross-validation**: Claims verified against multiple sources where possible (e.g., outlier importance confirmed by both LLM.int8() and PB-LLM).

3. **Theoretical grounding**: Information-theoretic framework derived from Sharpee (2017) PMC article on neural discretization.

4. **Recency balance**: Included both foundational 2016 XNOR-Net work and 2024-2025 advances to capture full historical arc.

## Gaps and Limitations

1. **Missing**: Direct experimental comparison of XNOR-Net applied to modern LLMs (not attempted because failure is well-documented in PB-LLM paper)

2. **Missing**: Formal mathematical proof of the "Fundamental Theorem" (stated as intuition/theorem sketch)

3. **Approximation**: Some quantitative comparisons synthesized from multiple papers with slightly different experimental setups
