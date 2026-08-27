# Bit-Serial Ternary Linear-Layer Experiments

## Project Status and Experiment Wrap-Up

This document records the methodology, implementation structure,
algorithms, benchmark evolution, validation procedure, and experimental
findings completed so far for the packed ternary / bit-serial
linear-layer inference experiments.

The goal of this phase was to establish a **correct and trustworthy
experimental baseline** before proceeding to low-level kernel
optimization and broader paper evaluation.

------------------------------------------------------------------------

## 1. Research Objective

The experiments investigate linear-layer computation where:

-   Weights are ternary: `{-1, 0, +1}`.
-   Ternary weights are stored in a compact bit-packed representation.
-   Activations are signed INT8 values.
-   INT8 activations are decomposed into 8 binary bitplanes.
-   Dot products are evaluated using bitwise AND and population-count
    operations rather than conventional per-element multiplication.

The central engineering question is:

> Can a packed ternary representation combined with bit-serial
> activation processing provide an efficient inference path while
> reducing weight-storage requirements?

At the current stage, the work establishes correctness and a reliable
CPU benchmark methodology. The current scalar bit-serial implementation
is **not yet faster than the dense scalar INT8 baseline**, but it is
substantially faster than naive scalar on-the-fly decoding of packed
ternary weights.

------------------------------------------------------------------------

## 2. Experimental Linear Layers

  Layer                     Matrix Shape   Packed Words per Row
  ----------------------- -------------- ----------------------
  `k_proj_640x2560`           640 x 2560                     80
  `q_proj_2560x2560`         2560 x 2560                     80
  `gate_proj_6912x2560`      6912 x 2560                     80

Since each packed word contains 32 bits, `2560 / 32 = 80 words per row`.

------------------------------------------------------------------------

## 3. High-Level System Structure

``` text
TERNARY WEIGHT MATRIX {-1,0,+1}
            |
            v
+---------------------------+
| Encode weights as masks   |
| +1 -> positive bit mask   |
| -1 -> negative bit mask   |
|  0 -> neither mask set    |
+-------------+-------------+
              |
              v
   PACKED LAYER BINARY FILE
    [header | pos[] | neg[]]
              |
      +-------+--------+
      |                |
      v                v
 BASELINE PATHS    PROPOSED PATH
      |                |
      |          INT8 activation
      |                |
      |                v
      |         8-bitplane packing
      |                |
      |                v
      |         bit-serial kernel
      |         AND + POPCOUNT
      |                |
      +-------+--------+
              |
              v
        OUTPUT CHECKSUM
              |
              v
          A == B == C
```

------------------------------------------------------------------------

## 4. Packed Ternary Weight Representation

Each ternary weight is represented using two logical bit masks:

``` text
Weight       pos bit       neg bit
-----------------------------------
  +1            1             0
   0            0             0
  -1            0             1
```

For a weight at column `c`:

``` text
word_index = c / 32
bit_index  = c % 32

weight =
    ((pos[word_index] >> bit_index) & 1)
  - ((neg[word_index] >> bit_index) & 1)
```

The packed layer structure is:

``` text
PackedLayer
|
+-- rows
+-- words_per_row
+-- original_cols
+-- pos[] : packed positive masks
+-- neg[] : packed negative masks
```

------------------------------------------------------------------------

## 5. INT8 Activation Bitplane Representation

A signed INT8 activation contains 8 bits. The proposed method decomposes
the activation vector into eight bitplanes:

``` text
INT8 activation vector
[x0, x1, x2, ... , xn]
          |
          v
+-----------------------+
| Activation Packing    |
+-----------------------+
          |
          +--> Plane 0 : bit 0 of every activation
          +--> Plane 1 : bit 1 of every activation
          +--> ...
          +--> Plane 7 : sign bit of every activation
```

Each plane is packed into 32-bit words.

The two's-complement significance is:

``` text
bit 0 ->   +1
bit 1 ->   +2
bit 2 ->   +4
bit 3 ->   +8
bit 4 ->  +16
bit 5 ->  +32
bit 6 ->  +64
bit 7 -> -128
```

------------------------------------------------------------------------

## 6. Algorithms Evaluated

### 6.1 Baseline A --- Dense INT8 Ternary MAC

Packed ternary weights are reconstructed into a dense INT8 matrix once
before benchmarking.

``` text
for each row r:
    acc = 0
    for each column c:
        acc += weight[r][c] * activation[c]
    checksum += acc
```

Dense reconstruction is outside the timed region.

### 6.2 Baseline B --- Scalar On-the-Fly Packed Decoding

For every weight inside the arithmetic loop:

1.  Locate the corresponding 32-bit word.
2.  Extract the positive bit.
3.  Extract the negative bit.
4.  Reconstruct the ternary value.
5.  Multiply by the INT8 activation.

``` text
for each row:
    for each column:
        decode ternary weight from pos/neg masks
        acc += decoded_weight * activation
```

This intentionally measures naive scalar decoding overhead.

### 6.3 Proposed Algorithm --- Bit-Serial Ternary Kernel

For each row and activation bitplane:

``` text
positive_count = popcount(pos_mask & activation_plane)
negative_count = popcount(neg_mask & activation_plane)

plane_acc = positive_count - negative_count
acc += plane_acc * bit_scale
```

where:

``` text
bit_scale = 1 << b  for b = 0 ... 6
bit_scale = -128    for b = 7
```

Structural view:

``` text
POS MASK ---- AND with activation plane ---- POPCOUNT ---+
                                                         |
                                                         +-- subtract
                                                         |
NEG MASK ---- AND with activation plane ---- POPCOUNT ---+
                                                         |
                                                         v
                                                   plane result
                                                         |
                                                         v
                                                  bit significance
                                                         |
                                                         v
                                                   accumulate row
```

The current implementation uses scalar `uint32_t` operations and
`POPCOUNT32`.

------------------------------------------------------------------------

## 7. Complete Proposed Inference Flow

``` text
INT8 ACTIVATION VECTOR
        |
        v
Pack into 8 bitplanes
        |
        v
Packed activation bitplanes
        |
        +-----------------------------+
        |                             |
        v                             v
Packed POS masks                Packed NEG masks
        |                             |
        v                             v
AND + POPCOUNT                  AND + POPCOUNT
        |                             |
        +-------------+---------------+
                      |
                      v
             POS count - NEG count
                      |
                      v
          Scale by 1,2,4,...,64,-128
                      |
                      v
                Row accumulator
                      |
                      v
                    Output
```

------------------------------------------------------------------------

## 8. Correctness Validation

For identical weights and activation inputs:

``` text
Same weights + activations
          |
    +-----+-----+
    |     |     |
    v     v     v
    A     B     C
    |     |     |
    +-----+-----+
          |
          v
      A == B == C
```

V5 expanded validation to 64 activation sets.

-   `k_proj`: PASS for all 64 sets; aggregate A = B = C = `-708320`
-   `q_proj`: PASS for all 64 sets; aggregate A = B = C = `840906`
-   `gate_proj`: PASS for all 64 sets; aggregate A = B = C = `39962954`

------------------------------------------------------------------------

## 9. Benchmark Methodology

Final V5 configuration:

``` text
Warm-up rounds : 30
Measured trials: 60
Inner reps     : 10
Activation sets: 64
```

### Warm-Up

All relevant paths are executed before measurement to reduce first-run
effects.

### Inner Repetitions

Each measured operation is repeated 10 times in a timing region, and
average latency per invocation is recorded.

### Balanced Randomized Ordering

Six permutations are used:

``` text
A-B-C
A-C-B
B-A-C
B-C-A
C-A-B
C-B-A
```

Across 60 trials, every permutation occurs exactly 10 times. Orders are
deterministically shuffled with a fixed seed.

### Preallocated Buffers

Activation bitplane buffers are allocated outside timed regions. Packing
reuses existing buffers without vector resize or allocation.

------------------------------------------------------------------------

## 10. Measurements Collected

Each raw trial records:

``` text
layer
trial
order
baseline_a_us
baseline_b_us
activation_pack_us
proposed_kernel_us
proposed_component_sum_us
proposed_direct_e2e_us
e2e_to_component_ratio
```

The internal consistency check is:

``` text
Component Sum = Pack Time + Kernel Time

Direct E2E = directly timed (Pack + Kernel)

Consistency Ratio = Direct E2E / Component Sum
```

A median ratio near 1.0 indicates agreement between the two timing
methods.

------------------------------------------------------------------------

## 11. Benchmark Evolution

### Initial Stage

Established:

-   packed layer loading,
-   dense reconstruction,
-   Baseline A,
-   Baseline B,
-   activation bitplanes,
-   proposed bit-serial kernel,
-   three-way correctness.

### V4

Added:

-   preallocated activation buffers,
-   allocation-free timed packing,
-   30 warmups,
-   60 trials,
-   10 inner repetitions,
-   balanced randomized A/B/C order,
-   separate pack and kernel timing,
-   direct E2E timing,
-   raw CSV output.

However, V4 showed a large inconsistency between Direct E2E and Pack +
Kernel. Median E2E/component ratios were approximately:

``` text
k_proj    : 4.82
q_proj    : 7.70
gate_proj : 8.95
```

Those direct-E2E values were therefore treated as a benchmark artifact
rather than trustworthy performance evidence.

### V5

V5 introduced a pool of 64 activation vectors and prepacked activation
inputs for C-only timing, while preserving multi-input correctness
validation.

Median results:

  ------------------------------------------------------------------------------------------
  Layer       Baseline A  Baseline B      Pack   Proposed   Component Direct E2E       E2E /
                                                   Kernel         Sum              Component
  ----------- ---------- ----------- --------- ---------- ----------- ---------- -----------
  k_proj      249.014 us 2453.614 us 87.420 us 677.090 us  765.764 us 761.442 us       0.997

  q_proj        1025.412   10241.131 87.624 us   2734.138 2821.469 us   2820.093       1.003
                      us          us                   us                     us 

  gate_proj     3068.414   28156.434 87.222 us   7461.314 7548.289 us   7530.032       1.001
                      us          us                   us                     us 
  ------------------------------------------------------------------------------------------

V5 therefore resolves the central V4 timing inconsistency and is the
current experimental baseline.

------------------------------------------------------------------------

## 12. Current Performance Interpretation

Compared with Baseline A using median Direct E2E:

``` text
k_proj:
761.442 / 249.014 ~= 3.06x slower

q_proj:
2820.093 / 1025.412 ~= 2.75x slower

gate_proj:
7530.032 / 3068.414 ~= 2.45x slower
```

The current scalar proposed implementation therefore does **not**
outperform the dense scalar INT8 baseline.

Compared with Baseline B:

``` text
k_proj:
2453.614 / 761.442 ~= 3.22x faster

q_proj:
10241.131 / 2820.093 ~= 3.63x faster

gate_proj:
28156.434 / 7530.032 ~= 3.74x faster
```

Thus, direct bit-serial processing is substantially faster than naive
scalar per-weight decoding in the tested configurations.

These results should not be interpreted as comparisons against optimized
production inference libraries.

------------------------------------------------------------------------

## 13. Key Experimental Observations

1.  All three implementations agree across all 64 tested activation sets
    for all three layers.
2.  V5 Direct E2E and Component Sum agree closely at the median.
3.  Activation packing is approximately 87 us at the median for all
    layers because the input width is 2560 in each case.
4.  Proposed kernel runtime grows with output-row count while activation
    packing remains approximately constant.
5.  Raw trials contain substantial outliers and apparent runtime
    regimes, especially for larger layers.
6.  Median statistics are currently more representative than simple
    means; final reporting should include IQR or percentile-based
    dispersion.

------------------------------------------------------------------------

## 14. Current Experimental Pipeline

``` text
[1] Ternary layer weights
          |
          v
[2] Encode into POS / NEG masks
          |
          v
[3] Save packed binary layer
          |
          v
[4] Load PackedLayer in C++
          |
    +-----+-------------------+
    |                         |
    v                         v
[5A] Reconstruct dense   [5B] Keep packed
     weights for A            weights
    |                         |
    v                         |
Dense scalar MAC              |
                              |
INT8 activation pool ---------+
          |
          +--------------------------+
          |                          |
          v                          v
   Baseline A/B inputs        Pack 8 bitplanes
                                     |
                                     v
                           Bit-serial POPCOUNT C
                                     |
          +--------------------------+
          |
          v
[6] Verify A == B == C for 64 inputs
          |
          v
[7] Warm benchmark paths
          |
          v
[8] Run 60 balanced randomized trials
          |
          v
[9] Measure A, B, Pack, C,
    Component Sum and Direct E2E
          |
          v
[10] Save raw CSV
          |
          v
[11] Aggregate median statistics
          |
          v
[12] Analyze correctness,
     consistency, performance, outliers
```

------------------------------------------------------------------------

## 15. What Has Been Established

-   Packed two-mask ternary weight representation: **DONE**
-   Dense reconstruction reference path: **DONE**
-   Scalar packed-decoding reference path: **DONE**
-   INT8 activation bitplane packing: **DONE**
-   Scalar bit-serial popcount kernel: **DONE**
-   Three-way A/B/C correctness: **DONE**
-   64-input correctness validation: **DONE**
-   Balanced benchmark ordering: **DONE**
-   Preallocated timed buffers: **DONE**
-   Direct E2E validation: **DONE**
-   V5 internally consistent benchmark baseline: **DONE**

------------------------------------------------------------------------

## 16. Next Experimental Phase

The benchmark infrastructure can now serve as the reference point for
optimization.

### SIMD and Vectorization

Investigate:

-   compiler auto-vectorization,
-   AVX2,
-   AVX-512 where available,
-   vectorized population-count strategies,
-   wider packed operations where appropriate.

### Loop and Memory Optimization

Investigate:

-   loop reordering,
-   repeated address-calculation reduction,
-   cache behavior,
-   weight-mask traversal,
-   activation-bitplane locality,
-   unrolling.

### Compiler Analysis

Inspect generated assembly and optimization reports to identify
bottlenecks and determine what transformations the compiler already
performs.

### Stronger Baselines

Baseline A is currently a scalar dense INT8 reference.
Publication-quality evaluation should eventually include appropriate
optimized CPU inference/GEMV or GEMM baselines and clearly distinguish
scalar reference comparisons from optimized production comparisons.

### Statistical Reporting

Final experiments should report:

-   median,
-   IQR,
-   relevant percentiles,
-   number of trials,
-   warm-up policy,
-   hardware,
-   compiler and flags,
-   thread count,
-   layer dimensions,
-   activation-set count,
-   CPU frequency/governor conditions where controllable.

------------------------------------------------------------------------

## 17. Current Research Position

The experiments have progressed from a representation and kernel
prototype to a reproducible benchmark with multi-input correctness
validation and internally consistent E2E measurements.

The current evidence supports the statement that packed ternary weights
can be consumed directly by a bit-serial popcount-based kernel without
scalar per-weight reconstruction while maintaining exact agreement with
the tested reference implementations.

The current evidence also shows that the scalar bit-serial
implementation does not yet outperform the dense scalar INT8 baseline,
although it substantially outperforms naive scalar decoding of the
packed ternary representation.

The next central research question is:

> Can architecture-aware optimization of the packed bit-serial kernel
> convert its representation-level advantages into competitive or
> superior end-to-end inference performance?

------------------------------------------------------------------------

## 18. Experiment Status Diagram

``` text
Packed ternary representation       [DONE]
        |
        v
Dense reconstruction baseline       [DONE]
        |
        v
Scalar packed decode baseline       [DONE]
        |
        v
INT8 activation bitplanes           [DONE]
        |
        v
Scalar bit-serial kernel            [DONE]
        |
        v
Three-way correctness               [DONE]
        |
        v
64-input correctness validation     [DONE]
        |
        v
Balanced benchmark ordering         [DONE]
        |
        v
Preallocated timed buffers          [DONE]
        |
        v
Direct E2E validation               [DONE]
        |
        v
V5 stable benchmark baseline        [DONE]
        |
        v
SIMD / architecture optimization    [NEXT]
        |
        v
Optimized baseline comparison       [FUTURE]
        |
        v
Full statistical evaluation         [FUTURE]
        |
        v
Paper-ready performance analysis    [FUTURE]
```

------------------------------------------------------------------------

## 19. Summary

The completed phase produced a validated experimental framework for
evaluating packed ternary linear layers with INT8 activations.

The core algorithm represents ternary weights through positive and
negative bit masks and decomposes signed INT8 activations into eight
packed bitplanes. Dot products are computed using bitwise intersections
and population counts, with each bitplane weighted according to
two's-complement significance.

Three implementations were evaluated:

-   **A:** dense scalar INT8 ternary multiply-accumulate.
-   **B:** packed ternary weights with scalar on-the-fly decoding.
-   **C:** proposed packed bit-serial popcount kernel.

V5 verifies exact agreement between A, B, and C across 64 activation
sets for all three tested layer dimensions. It also resolves the
direct-E2E timing inconsistency observed in V4.

The resulting benchmark should now be treated as the reference baseline
for the next phase: low-level optimization of the proposed kernel,
followed by comparison with stronger optimized baselines and a broader
publication-quality evaluation.
