Research Progress Report
Bit-Serial Ternary Inference Benchmarking

Version: 1.0

Authors: Team

Status: Ongoing Research

Table of Contents
Project Objective
Motivation
Background
Research Questions
Mathematical Formulation
Weight Representation
Activation Representation
Benchmark Evolution
Benchmark V4
Benchmark V5
Benchmark V6
Experimental Methodology
Correctness Validation
Benchmark Results
Performance Analysis
Why AVX2 Failed
Current Conclusions
Planned V7
ARM NEON Plan
gem5 Plan
Future Work
1. Project Objective

The objective of this research is to investigate whether bit-serial inference using ternary weight representations can provide competitive or superior inference performance compared to conventional dense INT8 matrix multiplication.

Instead of relying on multiplication instructions, the proposed method reformulates matrix-vector multiplication into a sequence of bitwise logical operations and population counts.

The long-term goal is to evaluate both algorithmic and architectural efficiency, ultimately studying the approach on simulated hardware using gem5.

2. Motivation

Transformer inference is dominated by repeated matrix-vector multiplications:

Y=W×X

where:

W is the weight matrix.
X is the activation vector.

Every output neuron performs thousands of multiply-accumulate operations.

Modern LLMs execute billions of these operations during inference.

Although INT8 quantization reduces computation cost, multiplication remains the dominant arithmetic operation.

The motivation behind this work is to investigate whether multiplication can be replaced with cheaper bitwise operations without sacrificing correctness.

3. Core Idea

Instead of storing weights as INT8 values, ternary weights are represented using two packed bitplanes:

Positive bitplane
Negative bitplane

Example:

Weight	Positive	Negative
+1	1	0
0	0	0
-1	0	1

Thus, a row of weights is stored as two bitmaps rather than an array of signed integers.

This significantly changes the computation model.

4. Mathematical Reformulation

Traditional dot product:

y=
i
∑
	​

w
i
	​

x
i
	​


For ternary weights:

w
i
	​

∈{−1,0,+1}

Each activation is decomposed into eight bitplanes.

For every bitplane:

PositiveMatches =
popcount(
PositiveWeights &
ActivationBits
)

NegativeMatches =
popcount(
NegativeWeights &
ActivationBits
)

Contribution:

PositiveMatches
-
NegativeMatches

scaled according to the bit significance.

5. Data Representation
Weight Packing
Dense Weights

+1 0 -1 +1 ...

↓

Positive Bitmap

1011...

↓

Negative Bitmap

0010...

Memory becomes

Positive Bits

Negative Bits

instead of

INT8
INT8
INT8
...
Activation Packing

Each INT8 activation vector is decomposed into eight bitplanes.

Activation

↓

Bit0

Bit1

Bit2

...

Bit7

Each plane is packed into uint32 words.

6. Overall Architecture
Weights
        │
        ▼

Weight Packing

        │

Positive Bitmap

Negative Bitmap

        │

        ▼

Activation

↓

Bitplane Packing

↓

Inference Kernel

↓

Output
7. Benchmark Evolution

The benchmark evolved through several iterations.

Version 4

Compared:

Dense INT8 baseline (A)
Packed scalar implementation (B)
Scalar bit-serial implementation (C)

Problems identified:

Packing overhead included in timing.
Single activation vector.
Cache effects.
Unstable benchmark ordering.
Version 5

Major improvements:

64 activation vectors.
Prepacked activation pool.
Packing performed before benchmarking.
Kernel-only timing.
End-to-end timing.
Randomized execution order.
Correctness verification.

Architecture:

Activation Set

↓

Packing

↓

Prepacked Pool

↓

Kernel Benchmark

This significantly improved measurement reliability.

Version 6

Objective:

Investigate SIMD acceleration.

Added:

Kernel D.

Pipeline:

Load 256 bits

↓

Vector AND

↓

Store

↓

Scalar POPCNT

↓

Accumulator
8. Correctness

Every benchmark compares outputs from:

A
B
C
D

Verification:

A==B==C==D

for all 64 activation vectors.

Every benchmark passed.

This establishes algorithmic correctness independently of implementation.

9. Layers Benchmarked

Three real projection layers were used:

k_proj

640 × 2560

q_proj

2560 × 2560

gate_proj

6912 × 2560

These cover small, medium, and large transformer projection layers.

10. Experimental Methodology

Compiler:

-O3
-std=c++17
-mavx2
-mpopcnt

Measurements collected:

Kernel execution time
End-to-end time
Speedup ratios

Warm-up iterations executed before measurement.

Execution order randomized to reduce cache bias.

11. Experimental Results

Across all three layers:

Performance ordering:

Dense

↓

Scalar Bit Serial

↓

AVX2

↓

Packed Decode

Observations:

Dense remained fastest.
Packed scalar decode incurred significant overhead.
Scalar bit-serial substantially improved over packed decode.
AVX2 implementation did not outperform scalar.
12. Why AVX2 Did Not Improve Performance

The AVX2 kernel vectorized only the logical AND operation.

Pipeline:

Vector AND

↓

Store

↓

Scalar POPCNT

↓

Reduction

Population counting remained scalar.

The expensive portion of the algorithm therefore remained unchanged.

The additional store instructions and vector handling overhead outweighed the benefit of vectorized AND operations.

13. Additional Observation

The dense baseline was compiled using:

-O3
-mavx2

Although written as scalar code, the compiler may automatically vectorize the dense multiply-accumulate loop.

This could partially explain why the dense implementation remained significantly faster.

Future work includes inspecting compiler vectorization reports.

14. Current Conclusions

Current research demonstrates:

The ternary formulation is mathematically correct.
Bitplane representation is functional.
Scalar bit-serial inference significantly improves over naïve packed decoding.
AVX2 logical-operation vectorization alone is insufficient to improve performance.
Population count remains the dominant bottleneck.
15. Planned Version 7

Kernel E will replace scalar population counting with a fully vectorized implementation.

Pipeline:

Vector AND

↓

Nibble Lookup

↓

Vector POPCOUNT

↓

Horizontal Reduction

↓

Accumulator

Implementation will use:

_mm256_shuffle_epi8
_mm256_sad_epu8

The benchmark will compare:

A
B
C
D
E

Metrics:

C/E
D/E
A/E
A/E End-to-End
16. Long-Term Roadmap
V4
│
├── Dense
├── Packed
└── Scalar
│
▼
V5
│
├── Stable Benchmark
├── 64 Activations
└── Correctness
│
▼
V6
│
├── AVX2
└── Performance Analysis
│
▼
V7
│
├── True Vector POPCOUNT
└── LUT-Based SIMD
│
▼
ARM NEON Port
│
▼
gem5 Simulation
│
├── CPI
├── Cache Behaviour
├── Memory Bandwidth
├── Instruction Mix
└── Energy Analysis
│
▼
Research Paper
17. Current Status

The project has successfully established:

A working bit-serial ternary inference framework.
A stable and reproducible benchmarking methodology.
Validation of four independent implementations (A–D).
Identification of the primary bottleneck (scalar population counting).
A clear path toward V7, ARM NEON, and gem5-based architectural evaluation.