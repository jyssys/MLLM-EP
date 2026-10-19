# Native FA3 envelope audit for the nonzero-tail certificate

Status: `NATIVE_ERROR_ENVELOPE_NOT_ESTABLISHED`.

## Outcome

The Round 9 real-arithmetic bound remains valid under its declared control contract.
This audit does not qualify that bound against the installed FA3 backend.
It assigns no zero native error.
It adds no candidate, kernel, model result, or quality claim.

The first unresolved numerical operation is the selected BF16 WGMMA dot product.
A sound bound must cover QK and the later probability-times-V reduction.
The public instruction contract does not specify their accumulation order, rounding, or subnormal handling.
This does not prove that a useful bound is impossible.
It prevents this audit from importing a sequential FP32-FMA error model as a documented native guarantee.

## Backend identity and actual observed extent

The installed package reports `0.26.0+cu129` and vLLM commit `568afb3a13806beb53bb2e6bd518269357b237c0`.
Three installed Python files match that commit byte-for-byte.
They select and call `torch.ops._vllm_fa3_C.fwd`.
The upstream [build declaration](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/cmake/external_projects/vllm_flash_attn.cmake#L24) pins flash-attention revision `caaa4eb59845388a20b1f435ecaafb4bd9517ad8`.
It also permits a local source override.

The [existing graph inventory](native_graph_argument_inventory_v1.md) contains 96 FA3 forward templates and 96 combine templates.
This audit reads its frozen kernel names, not new GPU observations.
The forward type uses BF16 operands, FP32 accumulators, SM90, and a 64×128×128 tile.
Its template enables the split path.
A combine node's presence does not establish its runtime split count.
The combine source returns early when that count is at most one.

The native TP2 ownership record lists BF16, head width 128, page size 16, and `FlashAttentionImpl`.
It does not identify every selected template in every campaign configuration.
The JSON records the installed extension's SHA256.
No build receipt here proves exact source/compiler equivalence for that binary.

The [matrix](../artifacts/native_fa3_tail_envelope_v1.json) records exact inspected ranges and hashes.
This audit reads selected kernel passages and the complete 191-line softmax helper.
It does not certify a full FA3 repository or full-paper read.

## Operations that the envelope must cover

All kernel links below use the pinned flash-attention revision.

| Stage | Observed source behavior | Required bound |
|---|---|---|
| QK | BF16 WGMMA with FP32 accumulator type | Signed dot-product error, overflow domain, and subnormal behavior |
| Score scale | Rounded scale/log2(e) coefficient and scaled score difference | Coefficient error and actual fused or separate arithmetic |
| Online softmax | FP32 maxima, `exp2f`, partial row sums, repeated rescaling | Exponential error, denominator lower bound, and each reduction/rescale |
| Probability conversion | FP32 exponential weights become BF16 before PV | BF16 conversion error before value accumulation |
| PV | BF16 probability/V WGMMA, FP32 state, output rescaling | Dot-product error and rescale error, including cancellation |
| Split combine | FP32 partial outputs/LSE, exponential weights, sum, reciprocal, weighted merge | A separate merge envelope when multiple splits execute |
| Final output | FP32-to-BF16 conversion | The exact rounding cell, ties, zero sign, overflow, and NaN policy |

The source shows these stages in [mainloop lines 1212–1280 and 1350–1442](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1212).
The [softmax helper](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/softmax.h#L63) computes the denominator from FP32 weights.
PV consumes the converted weights, not that identical FP32 array.
Thus, one exact normalized weighted sum does not describe the complete rounded computation.

The [combine helper](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/flash_fwd_combine_kernel.h#L585) adds another normalization and merge.
The [epilogue](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/epilogue_fwd.hpp#L244) uses the shared conversion helper.
Its pinned CUTLASS dependency implements the relevant [pair conversion](https://github.com/NVIDIA/cutlass/blob/62750a2b75c802660e4894434dc55e839f322277/include/cutlass/numeric_conversion.h#L1074) with `cvt.rn.bf16x2.f32`.

The intended [FA3 build flags](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/CMakeLists.txt#L120) include `--use_fast_math`.
[NVCC 12.9](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) associates that flag with FTZ, approximate division, and FMA contraction.
The observed binary's exact compile command remains unverified.

## What primary numerical sources establish

The [FA3 paper, Section 4.3 and Table 3](https://tridao.me/publications/flash3/flash3.pdf) compares numerical error with FP64 on constructed input distributions.
Its reported FP16/FP8 RMSE values are measurements, not per-coordinate worst-case BF16 bounds.
They cannot serve as the certificate's error radius.
The pinned [varlen test](https://github.com/vllm-project/flash-attention/blob/caaa4eb59845388a20b1f435ecaafb4bd9517ad8/hopper/test_flash_attn.py#L523) accepts output errors relative to a PyTorch reference error.
That empirical test is not a rounding-cell theorem.

[CUDA 12.9.1 PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma) specifies at least single-precision accumulation for BF16 WGMMA.
It leaves accumulation order, rounding, and subnormal behavior unspecified.
A type named FP32 does not resolve those details.
The same document gives `ex2.approx.f32` a 2-ulp bound.
Its FTZ variant also needs an absolute allowance for flushed values.

The [CUDA 12.9 mathematical-function tables](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#mathematical-functions) give separate function error limits.
Those limits do not combine themselves into an attention-output bound.
The actual lowered instructions, division mode, operation count, and range conditions still matter.

## Required conditional certificate

Let `a_D` and `a_S` denote real dense and kept-set outputs over the declared stored operands.
Let `z_D` and `z_S` denote the respective native values before their final BF16 casts.
A sufficient scalar check remains:

```text
abs(a_D - a_S) <= E
abs(z_D - a_D) <= eta_D
abs(z_S - a_S) <= eta_S
[z_S - (E + eta_D + eta_S), z_S + (E + eta_D + eta_S)]
    lies strictly inside the BF16 rounding cell of RN_BF16(z_S).
```

These are requirements, not measured native values.
Changing the kept set can change tile order, maxima, splits, and rescaling.
The dense and kept computations therefore need separate envelopes.
A zero-valued omitted V entry still changes the denominator.
A tiny exponential alone does not justify ignoring intermediate rounding.

Even page-score bounds need outward arithmetic.
They must account for the actual passed scale and rounded QK, not only an ideal `1/sqrt(128)`.
The page summary must also preserve request, depth, layer, head, generation, and committed-range identity.

### Positive mathematical control remains intact

The prior rational control uses kept output 1, tail mass `1/4096`, and tail value 2.
Its dense result is `4098/4097`.
Both final one-rounding BF16 words are 16256.
Its conservative tail radius is `3/4096`.
The strict symmetric cell radius around 1 is `1/512`.
Thus, an additional `eta_D + eta_S < 5/4096` would still satisfy this sufficient check.

The CPU calculation verifies that remaining budget exactly.
It does not assert that FA3 satisfies it.
The positive example uses constructed rational masses, not observed native logits.

For normal positive weights, a simple BF16 probability-cast model can contribute up to `2^-8 * max(abs(V))`.
This statement assumes an otherwise exact numerator and denominator.
It is not an FA3 envelope.
A coarse bound can consume a rounding-cell budget even when the actual constructed cast error is zero.
Its failure would not disprove a tighter, operand-specific certificate.

## Exceptional values and fail-closed conditions

A practical certificate needs a restricted, checked input domain.
Finite BF16 inputs alone do not exclude FP32 product or accumulation overflow.
Relative error formulas also need an absolute underflow/FTZ term.

NaN, infinity, signed zero, fully masked rows, and zero denominators need explicit policies.
The helper handles all-masked maxima and zero/NaN sums through specific branches.
These branches are not ordinary positive-mass normalization.
Skipping a nonfinite V can change a dense `0*Inf` or NaN path.
The certificate cannot assume that masking always removes this issue before arithmetic.

These checks require metadata or actual operand validation.
They cannot silently assume exceptional values absent from an unopened page.
No such native domain check executes in this audit.

## Smallest next prerequisite

First, bind one existing captured specialization to its binary build receipt and dynamic split/scheduler values.
The current inventory resolves the template family, not those argument values.

The first missing numerical contract is then a sound, range-qualified BF16 WGMMA dot-product bound for that specialization.
It must cover both QK and PV.
A vendor contract or independently checked instruction model could supply it.
A few successful GPU examples could not establish a universal bound.

Only after that contract exists should CPU interval propagation cover softmax, BF16 probability casts, rescaling, and split merge.
An archived complete Q/K/V case could then test whether the resulting certificate admits any page.
This audit neither requests nor performs that model/GPU step.

The disposition remains `GENERIC_CERTIFICATE_WITH_UNQUALIFIED_NATIVE_BOUND`.
This is not a proof against all exact selective-attention algorithms.

## Time and scope

Work started at 2026-10-19T17:52:12Z.
Source inspection ended at 2026-10-19T17:57:32Z.
The accounting cutoff is 2026-10-19T18:00:18Z.
The recorded active segment is 486 seconds, or 8 minutes 6 seconds.
The ledger excludes later timestamp stamping.
No pause or deliberate wait occurred.
The segment overlaps coordinator wall time.
It does not add independent campaign elapsed hours.

No GPU job, research model, Torch import, compilation, or installed-code mutation occurred.
CPU work reads source, hashes artifacts, demangles frozen names, and checks rational arithmetic.
Earlier reports, scripts, and raw numerical controls remain unchanged.
