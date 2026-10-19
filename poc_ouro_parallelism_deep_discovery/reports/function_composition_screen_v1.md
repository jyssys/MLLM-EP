# Function composition versus state readiness

## Question

Can a balanced composition tree bypass the recurrent input chain without hidden-state prediction?
The [CPU control](../analysis/function_composition_screen_v1.json) separates representation construction from state evaluation.
It introduces no new candidate card.

## Results

| Control | Actual result | Scope |
|---|---|---|
| Four exact rational affine maps, width2 | Balanced composition has two construction levels. The final value matches serial evaluation. | Positive compact-closure control, not BF16 model parity. |
| Four opaque nonlinear callbacks | Two construction levels still produce four ordered state evaluations. | This callback representation, not every possible algorithm. |
| Naive dense affine arithmetic | Serial uses 16 multiplications and 16 additions. Folding uses 40 and 28. | Scalar counts, not physical latency. |
| Four compositions of a squaring map | Degree grows from2 to16. | Counterexample to fixed-degree closure, not an Ouro theorem. |
| Complete width2048 lookup over 16-bit input patterns | This representation requires 2^32780 output bytes, excluding history. | Enumeration size, not a lower bound for all representations. |

The [scan reference](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda) requires an associative operator.
Its balanced-tree analysis counts operator applications.
It does not make an arbitrary function composition cheap.
No performance number from that older hardware becomes an H100 estimate.

## Transfer to unchanged Ouro

The inspected native path includes SiLU, normalization, residual updates, and depth-indexed attention state.
Shared weights do not make the complete state transition identical across depth-indexed histories.
The affine positive control therefore supplies no direct Ouro substitution.
Even real-arithmetic closure would still need the original BF16 numerical contract.

This screen supplies no model-output, GPU-latency, or new headroom result.
It does not disprove another compact representation, certified reformulation, or approximate algorithm.
A useful proposal must name that representation and charge construction, storage, evaluation, and state handling.
The [source audit](../artifacts/function_composition_source_audit_v1.json) retains read extents and the native file hash.
Eight CPU tests cover the controls and input guards.
