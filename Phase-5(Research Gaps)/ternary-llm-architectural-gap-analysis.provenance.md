# Provenance: Ternary LLM Architectural Gap Analysis

## Research Session Metadata
- **Date**: 2026-04-17
- **Researcher**: Feynman (AI Research Agent)
- **Topic**: Architectural gaps in running 1.58-bit ternary LLMs on byte-addressable edge hardware

## Primary Sources Consulted

### Academic Papers (via alphaXiv)

1. **T-SAR (arXiv:2511.13676)**
   - Full text retrieved and analyzed
   - Key finding: 75%+ of memory requests are TLUT accesses; 91.6% of execution time is memory R/W
   - Used for: Quantifying the memory bottleneck in CPU-based ternary LLM inference

2. **Bitnet.cpp (arXiv:2502.11880)**
   - AI-generated report retrieved
   - Key finding: Ternary Lookup Table (TL) and Int2 with Scale (I2_S) approaches
   - Used for: Understanding current CPU-based inference optimization approaches

3. **TerEffic (arXiv:2502.16473)**
   - AI-generated report retrieved
   - Key finding: 1.6-bit encoding (5 trits → 8 bits), LUT-based TMat core
   - Used for: Understanding FPGA-specific solutions and their encoding schemes

4. **TeLLMe (arXiv:2504.16266)**
   - AI-generated report retrieved
   - Key finding: Table-lookup MatMul for FPGAs, achieves 9.51 tokens/s at 7W
   - Used for: Understanding prefill optimization and FPGA advantages

5. **FullPack (arXiv:2211.06982)**
   - Full text retrieved
   - Key finding: Stride-based packing for sub-byte SIMD extraction
   - Used for: Understanding software-only optimization approaches for power-of-2 bit widths

### Web Sources

1. **Ternary Packing Blog (compilade.org)**
   - URL: https://compilade.org/blog/ternary-packing
   - Key finding: 5-in-8 encoding with multiplication-based unpacking for SIMD
   - Used for: Practical implementation details of ternary encoding

### Alpha Paper Q&A Sessions

1. **T-SAR Q&A**
   - Question: "What is the fundamental architectural bottleneck that T-SAR identifies?"
   - Answer: Heavy reliance on memory-based LUTs; 75%+ memory requests for TLUTs

2. **TerEffic Q&A**
   - Question: "What is the 1.6-bit encoding scheme and decoding overhead?"
   - Answer: 5 trits → 8 bits; minimal decoding overhead via bitwise operations

## Search Queries Executed

1. `alpha_search`: "ternary LLM 1.58-bit quantization edge deployment hardware architecture"
2. `alpha_search`: "sub-byte quantization memory packing unpacking overhead efficient inference"
3. `web_search`: "BitNet b1.58 ternary weights edge hardware deployment 2024 2025"
4. `web_search`: "sub-byte quantization memory bandwidth unpacking overhead edge AI"
5. `web_search`: "ternary neural network hardware acceleration FPGA ASIC NPU"
6. `alpha_search`: "byte-addressable memory architecture sub-byte access granularity alignment overhead"
7. `web_search`: "base-3 ternary encoding packing efficiency non-power-of-two hardware"
8. `alpha_search`: "cache line padding ternary packing memory controller non-power-of-two alignment"

## Key Findings Synthesis

### The Identified Gap
The academic community has:
- ✅ Optimized encoding schemes (5-in-8, 1.6-bit)
- ✅ Created SIMD unpacking optimizations
- ✅ Developed ISA extensions for LUT operations
- ✅ Built FPGA accelerators with custom datapaths
- ✅ Implemented in-register LUT generation

But has completely overlooked:
- ❌ Ternary-aware memory hierarchies
- ❌ Hardware support for base-3 index arithmetic
- ❌ Ternary-aligned cache line structures
- ❌ Ternary-aware memory allocators
- ❌ Prefetch patterns for ternary access

### The Proposed Solution
A **Ternary-Aware Memory Hierarchy** comprising:
1. Ternary Address Translation Units (eliminate division/modulo for index calculation)
2. Ternary-aligned cache line padding (reduce boundary crossing overhead)
3. Ternary-aware prefetch engines (optimize for non-power-of-2 access patterns)
4. Software memory allocators designed for ternary data

## Confidence Assessment

- **High confidence**: The memory bottleneck exists and is well-documented in T-SAR
- **High confidence**: Current solutions focus on compute/encoding, not memory hierarchy
- **Medium confidence**: The proposed ternary-aware memory hierarchy would provide 20-30% gains (extrapolated from T-SAR's measurements)
- **Medium confidence**: No existing work on ternary-aware memory hierarchies (based on comprehensive search)

## Limitations

1. No hardware simulation was performed to validate proposed solutions
2. The 20-30% improvement estimate is back-of-the-envelope, not empirically validated
3. The search may have missed recent work (late 2025/early 2026) not yet indexed
