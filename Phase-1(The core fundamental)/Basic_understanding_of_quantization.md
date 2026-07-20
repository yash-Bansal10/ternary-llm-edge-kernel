Lecture: Quantization and the Edge Device Memory Wall

 Adjusts glasses. Points at the board.

 Sit down. We're going to talk about why your "efficient" neural network running on a smartphone is actually a disaster waiting to happen.

 ────────────────────────────────────────────────────────────────────────────────

 Part I: The FP32 Elephant in the Room

 ### What You're Actually Storing

 A single FP32 weight is 32 bits = 4 bytes. That's the IEEE 754 standard:

 ```
   ┌─────────────────────────────────────────────┐
   │  Sign │  Exponent (8 bits) │ Mantissa (23) │
   │   1    │        8           │      23       │
   └─────────────────────────────────────────────┘
      bit 0    bits 1-8           bits 9-31
 ```

 ### The Math That Should Terrify You

 Consider a "small" model like ResNet-50:
```
 ┌─────────────┬──────────────┬───────────┐
 │ Component   │ Parameters   │ FP32 Size │
 ├─────────────┼──────────────┼───────────┤
 │ ResNet-50   │ 25.6 million │ 102.4 MB  │
 ├─────────────┼──────────────┼───────────┤
 │ BERT-Base   │ 110 million  │ 440 MB    │
 ├─────────────┼──────────────┼───────────┤
 │ GPT-2 Small │ 124 million  │ 496 MB    │
 └─────────────┴──────────────┴───────────┘
```
 Now let me ask you a simple question: What's the typical RAM on a mid-range smartphone?

 Waits. No one answers.

 4 GB. Shared between OS, apps, and your model.

 So your "small" ResNet-50 wants 100 MB just for weights. Add activations, gradients (if you're doing on-device training), and the operating system's memory footprint. You've just consumed 5-10% of available
 RAM before inference even starts.

 ### The Battery Problem: It's Not Just Storage

 Here's what most students misunderstand: the problem isn't just storing weights. It's moving them.

 Every time you run inference:

 ```
   Memory Access Energy (approximate, 45nm process):
   ─────────────────────────────────────────────────
   DRAM Read:        ~640 pJ per 32-bit word
   SRAM Read:        ~5   pJ per 32-bit word
   Integer Multiply: ~3.1 pJ
   FP Multiply:      ~3.7 pJ
 ```

 Memory access costs 170× more than the actual multiply operation.

 A single inference pass on ResNet-50 in FP32:
 - Reads ~100 MB of weights
 - At 640 pJ per 32-bit access = ~20 million accesses = ~12.8 mJ per inference
 - At 30 FPS = 384 mJ/second = 1.4 W just for memory reads

 Your phone battery is ~10-15 Wh. You just drained it in hours, not days.

 ────────────────────────────────────────────────────────────────────────────────

 Part II: What Quantization Actually Is

 ### The Core Operation

 Quantization maps a continuous floating-point range to a discrete integer grid:

 $$\mathbf{W}{int8} = \text{round}\left(\frac{\mathbf{W}{fp32}}{s}\right) + z$$

 Where:
 - $s$ = scale factor (step size between representable values)
 - $z$ = zero-point (integer value representing 0.0)

 Dequantization reverses this:

 $$\mathbf{W}{fp32} \approx s \times (\mathbf{W}{int8} - z)$$

 ### Concrete Example

 Original FP32 weights: [-0.72, -0.15, 0.0, 0.38, 0.91]

 With INT8 quantization (scale $s = 0.0078$, zero-point $z = 0$):

 ```
   FP32:    -0.72   -0.15    0.00    0.38    0.91
             ↓       ↓       ↓       ↓       ↓
   Scale:  -92.3   -19.2    0.0    48.7   116.7
             ↓       ↓       ↓       ↓       ↓
   Round:    -92     -19      0      49     117
             ↓       ↓       ↓       ↓       ↓
   INT8:    -92     -19      0      49     117  (stored as uint8 + offset)
             ↓       ↓       ↓       ↓       ↓
   Dequant: -0.72  -0.15   0.00   0.38   0.91  (reconstructed)
 ```

 The error? $|0.38 - 0.3822| = 0.0022$. Negligible for inference.

 ### What You Gain
```
 ┌───────────────────┬────────┬────────────────────┬─────────────────┐
 │ Metric            │ FP32   │ INT8               │ Improvement     │
 ├───────────────────┼────────┼────────────────────┼─────────────────┤
 │ Bits per weight   │ 32     │ 8                  │ 4× compression  │
 ├───────────────────┼────────┼────────────────────┼─────────────────┤
 │ Memory bandwidth  │ 100 MB │ 25 MB              │ 4× reduction    │
 ├───────────────────┼────────┼────────────────────┼─────────────────┤
 │ Energy per access │ 640 pJ │ 160 pJ (effective) │ 4× savings      │
 ├───────────────────┼────────┼────────────────────┼─────────────────┤
 │ Integer ops       │ N/A    │ ~3.1 pJ            │ Cheaper than FP │
 └───────────────────┴────────┴────────────────────┴─────────────────┘

 ────────────────────────────────────────────────────────────────────────────────
```
 Part III: The Physical Analogy

 Draws on board.

 ### The Parking Lot Problem

 Imagine you run a valet parking service:

 FP32 = Full-sized parking spaces
 - Each space: 32 feet wide
 - You can park any car: compact cars, trucks, tanks, spacecraft
 - Precision: You can position a car to within 1 millimeter
 - Cost: 100 parking spaces = 3,200 linear feet of real estate

 INT8 = Compact-only spaces
 - Each space: 8 feet wide
 - You can only park compact cars (values must fit in range)
 - Precision: You can position to within ~4 inches (quantization error)
 - Cost: 100 parking spaces = 800 linear feet of real estate

 ### The Trade-off

 If your customers all drive compact cars (neural network weights with bounded range):

 - FP32: Wasting 75% of your real estate on empty space between cars
 - INT8: Perfectly adequate, saves 75% land, costs less to maintain

 But if a customer shows up with a tank (outlier weight value):
 - FP32: No problem, fits easily
 - INT8: You have to turn it away or clip it (quantization error → accuracy loss)

 ### The Energy Connection

 Every time you need to retrieve a car (read a weight during inference):
 - FP32 lot: Your valet walks 32 feet per car → tired valet (high energy)
 - INT8 lot: Your valet walks 8 feet per car → efficient valet (low energy)

 Multiply by millions of weights accessed per inference. Now you understand the battery drain.

 ────────────────────────────────────────────────────────────────────────────────

 Part IV: The Accuracy Trade-off

 ### Why Bits Matter: The Information Content

 An $n$-bit integer can represent $2^n$ distinct values:
```
 ┌───────────┬─────────────────┬─────────────────────────┐
 │ Bit-width │ Distinct Values │ Dynamic Range           │
 ├───────────┼─────────────────┼─────────────────────────┤
 │ FP32      │ ~4.3 billion    │ $10^{-38}$ to $10^{38}$ │
 ├───────────┼─────────────────┼─────────────────────────┤
 │ INT16     │ 65,536          │ $-32,768$ to $32,767$   │
 ├───────────┼─────────────────┼─────────────────────────┤
 │ INT8      │ 256             │ $-128$ to $127$         │
 ├───────────┼─────────────────┼─────────────────────────┤
 │ INT4      │ 16              │ $-8$ to $7$             │
 ├───────────┼─────────────────┼─────────────────────────┤
 │ Binary    │ 2               │ $-1$ or $+1$            │
 └───────────┴─────────────────┴─────────────────────────┘
```
 ### The Precision-Loss Mechanism

 Consider a weight distribution in a trained network:

 ```
   Weight histogram (simplified):
        │
     ▂▃▅███████████▅▃▂
     ───────────────────► weight value
    -1.0   0.0    1.0
 ```

 With INT8 (256 levels over range [-1, 1]):
 - Resolution: $\frac{2}{256} = 0.0078$ per step
 - Weights differing by < 0.0078 get collapsed to same value

 This creates two problems:

 1. Clipping Error (Outliers)

 ```
   Original:  [0.98, 0.99, 1.02, 1.05]  (values > 1.0)
   Quantized: [0.99, 0.99, 1.00, 1.00]  (clipped to max)
                        ↑       ↑
                     Information lost
 ```

 2. Round-off Error (Precision)

 ```
   Original:  [0.1234, 0.1278, 0.1312]
   Quantized: [0.125,  0.125,  0.125 ]  (all map to same bin)
                 ↑        ↑
              Distinguishability lost
 ```

 ### Empirical Reality
```
 ┌───────────────┬──────────────────────────────┐
 │ Quantization  │ Typical Accuracy Drop        │
 ├───────────────┼──────────────────────────────┤
 │ FP32 → FP16   │ ~0% (near-lossless)          │
 ├───────────────┼──────────────────────────────┤
 │ FP32 → INT8   │ 0.5-2% (usually acceptable)  │
 ├───────────────┼──────────────────────────────┤
 │ FP32 → INT4   │ 5-15% (requires fine-tuning) │
 ├───────────────┼──────────────────────────────┤
 │ FP32 → Binary │ 10-30% (requires retraining) │
 └───────────────┴──────────────────────────────┘
```
 ────────────────────────────────────────────────────────────────────────────────

 Part V: The Complete Picture

 ### Memory Bandwidth: The Real Bottleneck

 Draws a diagram.

 ```
   ┌─────────────────────────────────────────────────────────┐
   │                    EDGE DEVICE                          │
   │  ┌──────────┐         ┌──────────┐                     │
   │  │  DRAM    │ ──────► │  SRAM/   │ ────► [Compute]    │
   │  │ (Slow,   │  Bus    │  Cache   │                     │
   │  │  Large)  │ Width   │ (Fast,   │                     │
   │  │          │         │  Small)  │                     │
   │  └──────────┘         └──────────┘                     │
   │       │                                                    │
   │       │ 100 MB weights (FP32)                             │
   │       │ vs 25 MB weights (INT8)                           │
   │       │                                                    │
   │       ▼                                                    │
   │  [Battery drains at 4× rate with FP32]                    │
   └─────────────────────────────────────────────────────────┘
 ```

 The bus between DRAM and cache has finite bandwidth. In mobile SoCs:
 - Typical memory bandwidth: 10-30 GB/s
 - FP32 ResNet-50 inference: Must transfer 100+ MB
 - At 20 GB/s: 5 ms just to load weights (before compute!)
 - INT8: 1.25 ms to load same model

 That's 3.75 ms saved per inference. At 30 FPS, that's 112 ms/second = 11% more time for actual computation.

 ────────────────────────────────────────────────────────────────────────────────

 Summary

 Erases board. Writes in large letters.

 Quantization is the controlled sacrifice of numerical precision for memory and energy efficiency.

 The physics:

 1. Memory dominates energy: Moving data costs 100× more than computing on it
 2. Bit-width = bandwidth: 4× fewer bits = 4× less data movement
 3. Precision has diminishing returns: Neural networks are surprisingly tolerant of quantization noise because they're trained to be robust to input variations
 4. The trade-off is real: Every bit removed increases representation error. The art is knowing when the model still works

 When someone tells you their model is "efficient," ask them: "In what precision?" A 100 MB model in FP32 is a 25 MB model in INT8. That's not optimization. That's arithmetic.

 Dismisses class.

 ────────────────────────────────────────────────────────────────────────────────

 Sources for deeper reading:
 - Han et al., "Deep Compression" (ICLR 2016) — pruning + quantization
 - Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018) — INT8 training
 - Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference" (arXiv:1806.08342)