 Mathematical Operations: FP32 vs Binary Neural Networks

 ### Standard FP32 Neural Network Layer

 For a fully-connected layer or convolutional layer, the fundamental operation is the Multiply-Accumulate (MAC):

 $$y = \sum_{i=1}^{n} w_i \cdot x_i + b$$

 Hardware operations per output element:
```
 ┌───────────────────────┬──────────┬─────────────────────┐
 │ Operation             │ Count    │ Description         │
 ├───────────────────────┼──────────┼─────────────────────┤
 │ FP32 Multiplication   │ $n$      │ $w_i \times x_i$    │
 ├───────────────────────┼──────────┼─────────────────────┤
 │ FP32 Addition         │ $n$      │ Accumulate products │
 ├───────────────────────┼──────────┼─────────────────────┤
 │ FP32 Add (bias)       │ 1        │ Add bias term       │
 ├───────────────────────┼──────────┼─────────────────────┤
 │ Total FP32 operations │ $2n + 1$ │ ~$n$ MACs + 1 add   │
 └───────────────────────┴──────────┴─────────────────────┘
```
 For a convolutional layer with $C_{out}$ output channels, $C_{in}$ input channels, and $k \times k$ kernel:
 - Operations per output pixel: $C_{in} \times k^2$ MACs
 - Operations per output channel: $H_{out} \times W_{out} \times C_{in} \times k^2$ MACs

 ────────────────────────────────────────────────────────────────────────────────

 ### 1-Bit Binary Neural Network Layer

 When weights are constrained to ${-1, +1}$, the multiplication transforms:

 $$y = \sum_{i=1}^{n} w_i \cdot x_i + b \quad \text{where } w_i \in {-1, +1}$$

 Key insight: Since $w_i \in {-1, +1}$:

 $$w_i \cdot x_i = \begin{cases} +x_i & \text{if } w_i = +1 \ -x_i & \text{if } w_i = -1 \end{cases}$$

 This means multiplication becomes conditional negation:
```
 ┌────────────────────┬──────────┬──────────────────────────────┐
 │ Hardware Operation │ Count    │ Description                  │
 ├────────────────────┼──────────┼──────────────────────────────┤
 │ Negation/Add       │ $n$      │ Conditional: $x_i$ or $-x_i$ │
 ├────────────────────┼──────────┼──────────────────────────────┤
 │ Addition           │ $n$      │ Accumulate                   │
 ├────────────────────┼──────────┼──────────────────────────────┤
 │ FP32 Add (bias)    │ 1        │ Add bias                     │
 ├────────────────────┼──────────┼──────────────────────────────┤
 │ Total              │ $2n + 1$ │ But no multipliers           │
 └────────────────────┴──────────┴──────────────────────────────┘
```
 Result: MAC units replaced by adder-subtractor units.

 ────────────────────────────────────────────────────────────────────────────────

 ### Fully Binarized (Weights + Activations Both Binary)

 When both weights $w_i \in {-1, +1}$ AND activations $x_i \in {-1, +1}$:

 $$w_i \cdot x_i = \begin{cases} +1 & \text{if } w_i = x_i \ -1 & \text{if } w_i \neq x_i \end{cases}$$

 This is exactly the XNOR operation (logical equivalence):

 $$w_i \odot x_i = \text{XNOR}(w_i, x_i)$$

 The sum becomes a population count (popcount):

 $$y = \sum_{i=1}^{n} \text{XNOR}(w_i, x_i) = 2 \cdot \text{popcount}(\text{XNOR}(\mathbf{w}, \mathbf{x})) - n$$

 Hardware operations:
```
 ┌─────────────────┬─────────┬─────────────────────────────┐
 │ Operation       │ Count   │ Description                 │
 ├─────────────────┼─────────┼─────────────────────────────┤
 │ XNOR (1-bit)    │ $n$     │ Bitwise equivalence         │
 ├─────────────────┼─────────┼─────────────────────────────┤
 │ Popcount        │ 1       │ Count 1s in result          │
 ├─────────────────┼─────────┼─────────────────────────────┤
 │ Integer scaling │ 1       │ $2 \times \text{count} - n$ │
 ├─────────────────┼─────────┼─────────────────────────────┤
 │ Total           │ $n + 2$ │ All 1-bit or integer ops    │
 └─────────────────┴─────────┴─────────────────────────────┘
```
 ────────────────────────────────────────────────────────────────────────────────

 How Binary Weights Eliminate MAC Units

 ### The Hardware Cost Hierarchy
```
 ┌────────────────────┬───────────────────┬──────────────────┐
 │ Operation          │ Energy (relative) │ Area (relative)  │
 ├────────────────────┼───────────────────┼──────────────────┤
 │ 32-bit FP Multiply │ ~3.7 pJ           │ ~200 FPGA slices │
 ├────────────────────┼───────────────────┼──────────────────┤
 │ 32-bit FP Add      │ ~0.9 pJ           │ ~50 FPGA slices  │
 ├────────────────────┼───────────────────┼──────────────────┤
 │ 32-bit Int Add     │ ~0.1 pJ           │ ~10 FPGA slices  │
 ├────────────────────┼───────────────────┼──────────────────┤
 │ 1-bit XNOR         │ ~0.001 pJ         │ ~1 FPGA slice    │
 ├────────────────────┼───────────────────┼──────────────────┤
 │ Popcount (32-bit)  │ ~0.02 pJ          │ ~5 FPGA slices   │
 └────────────────────┴───────────────────┴──────────────────┘
```
 Source: XNOR-Net paper; BinaryNet paper; energy estimates from Horowitz (2014)

 ### Why MAC Units Disappear

 A MAC unit contains:
 1. A multiplier (the dominant cost)
 2. An accumulator (adder with register)
 3. Control logic

 Binary weights eliminate the multiplier because:

 $$\text{sign}(w) \times x = \begin{cases} +x & \text{no operation needed} \ -x & \text{2's complement negate} \end{cases}$$

 The "multiplication" reduces to:
 - 0 gates when $w = +1$ (pass-through)
 - NOT gates + adder when $w = -1$ (negation)

 In practice, this is implemented as a conditional negate or XOR with sign bit followed by accumulation.

 ────────────────────────────────────────────────────────────────────────────────

 ### Fully Binary Case: XNOR-Popcount Architecture

 From the XNORBIN paper (ETH Zurich, 65nm CMOS):

 │ "Binary neural networks (BNNs) use bipolar binarization (+1/-1) for both the model weights and feature maps, reducing overall memory size and bandwidth constraints by ≈ 32×, and simplifying the costly
 │ sum-of-products computation to mere XNOR-and-popcount operations."

 The hardware transforms:

 ```
   Traditional MAC:    [FP32 Multiplier] → [FP32 Adder] → [Accumulator]
                                           ↓
   Binary MAC:          [XNOR Gate] → [Popcount] → [Integer Adder]
 ```

 32-bit packed operation: Process 32 binary values simultaneously:

 $$\text{XNOR}{\text{packed}}(\mathbf{w}{32}, \mathbf{x}_{32}) \xrightarrow{\text{popcount}} \text{count of matches}$$

 This gives a theoretical 32× speedup for the same silicon area.

 ────────────────────────────────────────────────────────────────────────────────

 Concrete Example: 3×3 Convolution

 ### FP32 Convolution

 For one output pixel with $C_{in}$ input channels:

 ```python
   # Pseudocode
   output = 0.0
   for c in range(C_in):
       for i in range(3):
           for j in range(3):
               output += weight[c,i,j] * input[c, y+i, x+j]  # 9×C_in MACs
   output += bias
 ```

 Operations: $9 \times C_{in}$ FP32 multiplications + $9 \times C_{in}$ FP32 additions

 ### Binary Convolution

 ```python
   # Pseudocode (weights and inputs both binary)
   matches = 0
   for c in range(C_in):
       for i in range(3):
           for j in range(3):
               matches += XNOR(weight[c,i,j], input[c, y+i, x+j])
   output = 2 * matches - 9*C_in  # Convert count to sum in {-9, ..., +9}
 ```

 Operations: $9 \times C_{in}$ XNOR gates + 1 popcount + 1 integer scale

 Or packed (32 values at once):

 ```python
   # 32 binary inputs/weights per XNOR operation
   matches = popcount(XNOR(weight_packed, input_packed))
 ```

 ────────────────────────────────────────────────────────────────────────────────

 Summary Table
```
 ┌────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────────┐
 │ Aspect             │ FP32 Layer               │ Binary Weights Only      │ Binary Weights + Activations │
 ├────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
 │ Weight storage     │ 32 bits                  │ 1 bit                    │ 1 bit                        │
 ├────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
 │ Activation storage │ 32 bits                  │ 32 bits                  │ 1 bit                        │
 ├────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
 │ Memory reduction   │ —                        │ 32× (weights)            │ 32× (both)                   │
 ├────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
 │ Core arithmetic    │ FP32 multiply-add        │ Conditional add/subtract │ XNOR + popcount              │
 ├────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
 │ Hardware unit      │ MAC (multiplier + adder) │ Adder-subtractor         │ Logic gates + counter        │
 ├────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
 │ Energy/op          │ ~4-10 pJ                 │ ~0.1-0.5 pJ              │ ~0.01-0.05 pJ                │
 ├────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────┤
 │ Speedup potential  │ 1×                       │ ~2×                      │ ~32-64×                      │
 └────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────────┘
```
 ────────────────────────────────────────────────────────────────────────────────

 Sources

 1. XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks (Rastegari et al., 2016). arXiv:1603.05279. https://arxiv.org/abs/1603.05279
 2. Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or −1 (Courbariaux et al., 2016). arXiv:1602.02830. https://arxiv.org/abs/1602.02830
 3. XNORBIN: A 95 TOp/s/W Hardware Accelerator for Binary Convolutional Neural Networks (Al Bahou et al., ETH Zurich, 2018). arXiv:1803.05849. https://arxiv.org/abs/1803.05849

 ────────────────────────────────────────────────────────────────────────────────

 Key insight: The multiplier—the most expensive component in a MAC unit—is eliminated because multiplying by $\pm 1$ requires only sign manipulation (XOR) or pass-through, not actual multiplication. In the
 fully binary case, this extends to replacing the entire MAC with XNOR-popcount logic, achieving 50-100× energy efficiency gains.