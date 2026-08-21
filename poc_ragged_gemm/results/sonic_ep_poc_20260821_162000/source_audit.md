# Source and prior audit

## SonicMoE / QuACK execution

The external SonicMoE checkout is commit
`7396f3e604827d8186c2e16e64b28ee33d3defd0`. Its forward path calls QuACK's
variable-M grouped GEMMs with cumulative per-expert row offsets. The checked
SM90 configurations use a single launch-wide `tile_m` (normally 128, with a
256 option); a launch does not select a different M tile for each expert.
QuACK schedules each group with `ceil_div(M, tile_m)` work units and masks
out-of-bounds accesses. Sonic's benchmark token-rounding function fixes
`Mtile=128` and changes expert counts using nearest/up/down rounding.

The source establishes ceiling-quantized scheduling and predicated addressing,
but source inspection alone does **not** prove that a complete tensor-core MMA
cost is exposed for every invalid row. The iso-N/G timing is therefore the
deciding evidence: increasing arithmetic tile count from 32 to 48 did not
increase median kernel latency for either tested shape, and the boundary sweep
did not produce a reproducible staircase.

The current Sonic checkout declares QuACK >=0.6.4. A Python 3.12 isolated
environment with QuACK 0.6.4 and CUTLASS DSL 4.6.2 imported successfully but
failed during first-kernel compilation with a CUTLASS DSL kwargs-wrapper ABI
error. No package in the existing vLLM environment was changed. The measured
Sonic compatibility baseline consequently uses the already validated QuACK
0.5.0/CUTLASS DSL 4.5.3 stack; this is a material limitation.

## vLLM Qwen3 path

The downloaded checkpoint is Qwen3MoeForCausalLM, BF16, 48 layers, hidden 2048,
MoE intermediate 768, 128 experts, top-8, and normalized top-k probabilities.
The validated live topology is TP4/DP1/EP4/PP1 with linear placement and 32
disjoint local experts per rank. vLLM selected `FlashInferExperts` and generated
the H100 CUTLASS unquantized grouped-MoE M128 kernel family. The hook times only
the selected local expert backend, after dispatch and before combine, and
derives exact local histograms from unchanged top-k IDs plus `expert_map`.

## Prior overlap and novelty risk

* SonicMoE changes token/expert assignment counts to reach tile alignment; it
  is the routing-edit counterfactual, not the proposed exact-routing method.
* TEMPO and DA-MoE already motivate shape-aware EP/runtime decisions. A better
  straggler predictor is not novel here.
* TMA-Adaptive FP8 Grouped GEMM already executes exact heterogeneous group
  lengths without padding and reports adaptive tile scheduling. This is direct
  mechanism overlap, although its published scope is FP8.
* NVIDIA CUTLASS also documents a newer Blackwell ragged contiguous grouped
  GEMM example. It is not the measured H100/BF16 path, but it creates a second
  direct-prior risk for a generic “ragged grouped GEMM” claim.
* MegaBlocks and ScatterMoE are older sparse/grouped execution context, not a
  basis for claiming tile-waste novelty.

`NOVELTY_RISK: HIGH` — any future claim would need to be narrowed to a concrete
H100 BF16 EP inference mechanism and compared directly with the exact-routing
adaptive/ragged kernels above.
