# Provenance: BitNet b1.58 Zero Role Literature Review

## Primary Academic Sources

### 1. Trained Ternary Quantization (TTQ)
- **ID**: arXiv:1612.01064
- **Title**: Trained Ternary Quantization
- **Authors**: Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally
- **Venue**: ICLR 2017
- **URL**: https://arxiv.org/abs/1612.01064
- **AlphaXiv**: https://www.alphaxiv.org/overview/1612.01064
- **Key findings used**:
  - Ternary quantization outperforms full-precision AlexNet on ImageNet
  - Optimal sparsity between 30-50% for accuracy
  - Separate scaling factors for positive and negative weights
  - Gradient handling for zero weights differs from ±1 weights

### 2. Ternary Quantization: A Survey
- **ID**: arXiv:2303.01505
- **Title**: Ternary Quantization: A Survey
- **Authors**: Dan Liu, Xue Liu
- **Venue**: McGill University, 2023
- **URL**: https://arxiv.org/abs/2303.01505
- **AlphaXiv**: https://www.alphaxiv.org/overview/2303.01505
- **Key findings used**:
  - Classification of projection functions (direct vs. indirect)
  - Optimization methods from proximal operator perspective
  - Historical development from 1988 to present
  - Gradient behavior in zero zone

### 3. Lightweight Neural Networks
- **ID**: arXiv:1712.05695
- **Title**: Lightweight Neural Networks
- **Author**: Altaf H. Khan
- **Venue**: 2017
- **URL**: https://arxiv.org/abs/1712.05695
- **AlphaXiv**: https://www.alphaxiv.org/overview/1712.05695
- **Key findings used**:
  - Universal approximation with {-1, 0, +1} weights
  - Variable receptive fields emerging from sparsity
  - Receptive field size decreases with network size
  - 80-95% weight pruning during training
  - Biological analogy to excitatory/inhibitory synapses

### 4. Getting Away with More Network Pruning
- **ID**: arXiv:2301.07966
- **Title**: Getting Away with More Network Pruning: From Sparsity to Geometry and Linear Regions
- **Authors**: Junyang Cai, Khai-Nguyen Nguyen, et al.
- **Venue**: Bucknell University, University of Utah, 2023
- **URL**: https://arxiv.org/abs/2301.07966
- **AlphaXiv**: https://www.alphaxiv.org/overview/2301.07966
- **Key findings used**:
  - Upper bound on linear regions for pruned networks
  - Relationship between sparsity and rank
  - Optimal layer-wise sparsity patterns
  - Connection between linear regions and accuracy

### 5. XNOR-Net (Binary Baseline)
- **ID**: arXiv:1603.05279
- **Title**: XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
- **Authors**: Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, Ali Farhadi
- **Venue**: ECCV 2016
- **URL**: https://arxiv.org/abs/1603.05279
- **Key findings used**:
  - Binary-Weight-Networks: 56.8% Top-1 accuracy
  - XNOR-Networks: 44.2% Top-1 accuracy (both weights and activations binary)
  - Optimal scaling factor derivation
  - Success on ImageNet, failure on LLMs (documented in other papers)

### 6. PB-LLM (Binary Failure on LLMs)
- **ID**: arXiv:2310.00034
- **Title**: PB-LLM: Partially Binarized Large Language Models
- **Authors**: Yuzhang Shang, Zhihang Yuan, Qiang Wu, Zhen Dong
- **Venue**: 2023
- **URL**: https://arxiv.org/abs/2310.00034
- **AlphaXiv**: https://www.alphaxiv.org/overview/2310.00034
- **Key findings used**:
  - Binary methods perform worse than random on LLMs
  - Salient weights must be preserved at higher precision
  - 5% outliers occupy 50% of weight range
  - Mixed-precision necessity

### 7. BitNet b1.58
- **ID**: arXiv:2402.17764
- **Title**: The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits
- **Authors**: Shuming Ma, Hongyu Wang, et al.
- **Venue**: 2024
- **URL**: https://arxiv.org/abs/2402.17764
- **AlphaXiv**: https://www.alphaxiv.org/overview/2402.17764
- **Key findings used**:
  - Ternary weights {-1, 0, +1} achieve FP16 parity
  - 71.4x energy reduction for arithmetic operations
  - Performance parity at 3B+ parameters
  - Zero value enables sparsity patterns

### 8. BitNet (Original)
- **ID**: arXiv:2310.11453
- **Title**: BitNet: Scaling 1-bit Transformers for Large Language Models
- **Authors**: Hongyu Wang, Shuming Ma, et al.
- **Venue**: 2023
- **URL**: https://arxiv.org/abs/2310.11453
- **AlphaXiv**: https://www.alphaxiv.org/overview/2310.11453
- **Key findings used**:
  - Training from scratch with binary constraints
  - 2-3 point accuracy gap from FP16
  - Scaling laws hold for binary transformers
  - Energy efficiency scaling with model size

## Supporting Sources

### Information-Theoretic Foundations

9. **Optimizing Neural Information Capacity through Discretization**
   - **Author**: Tatyana O. Sharpee
   - **Source**: PMC/Neuron, 2017
   - **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5634811/
   - **Key findings used**:
     - Optimal discretization depends on noise level
     - Low noise: distributed coding with staggered thresholds
     - High noise: redundant coding with identical thresholds
     - Threshold distribution should match input distribution

### Web Sources

10. **The Outlier Problem in LLM Quantization**
    - **Source**: Medium synthesis
    - **URL**: https://medium.com/@serahabensour/the-outlier-problem-how-a-few-rogue-numbers-are-breaking-llm-quantization-510a25fe2ed4
    - **Key findings used**:
      - Unified framework for outlier types
      - Softmax as root cause of outlier emergence
      - Context-aware scaling solutions

11. **BitNet b1.58 Analysis**
    - **Source**: Multiple synthesis sites
    - **URL**: https://deep-paper.org/en/paper/2402.17764/
    - **Key findings used**:
      - Memory and latency comparisons
      - Energy consumption analysis
      - Scaling law implications

## Search Methodology

### Paper Discovery
1. **Semantic search**: Used alpha_search for papers on "ternary neural networks", "binary quantization", "sparsity feature selection"
2. **Keyword search**: Searched for "Ternary weight networks" directly
3. **Citation tracking**: Followed references from BitNet papers to foundational ternary quantization work

### Query Evolution
- Initial: "BitNet b1.58 architecture zero weight"
- Expanded: "ternary neural networks {-1, 0, 1} quantization sparsity feature selection"
- Deep dive: "binary vs ternary quantization representational capacity theory"
- Specific: "BitNet b1.58 zero weight sparsity analysis"

### Cross-Validation
Claims verified against multiple sources:
- Sparsity-accuracy relationship: TTQ paper + Survey paper
- Gradient behavior: TTQ paper + Survey paper + Lightweight Neural Networks
- Linear regions theory: Pruning paper + Survey paper
- LLM failure modes: PB-LLM + BitNet papers + Web synthesis

## Gaps and Limitations

### Missing Elements
1. **Direct experimental comparison**: No paper directly compares {-1, +1} vs {-1, 0, +1} for LLMs with controlled training
2. **Formal proof**: No theoretical proof that 50% sparsity is optimal
3. **Information-theoretic bounds**: No formal capacity bounds for ternary vs binary

### Approximations
1. **Training differences**: Some papers train from scratch (BitNet), others quantize pre-trained (TTQ, PB-LLM)
2. **Architecture differences**: Comparisons span CNNs, MLPs, and Transformers
3. **Dataset differences**: MNIST, CIFAR, ImageNet, and language modeling corpora

## Verification Status

| Claim | Source | Verification |
|-------|--------|---------------|
| TTQ outperforms FP on ImageNet | arXiv:1612.01064 Table 2 | Direct from paper |
| Optimal sparsity 30-50% | arXiv:1612.01064 Figure 5 | Direct from paper |
| Binary fails on LLMs | arXiv:2310.00034 Figure 2 | Direct from paper |
| Zero gradient scaled by 1 | arXiv:1612.01064 Eq. 8 | Direct from paper |
| Receptive fields vary with size | arXiv:1712.05695 Figure 2 | Direct from paper |
| Linear regions drop with sparsity | arXiv:2301.07966 Figures 1-2 | Direct from paper |
| BitNet b1.58 FP16 parity | arXiv:2402.17764 Tables 2-4 | Direct from paper |

## Date of Review

- **Literature search**: 2026-04-17
- **Paper versions**: Latest available as of search date
- **Field maturity**: Well-established for ternary quantization, emerging for LLM applications
