# FlashVEP Phase 1 Baseline Profile

> Current TP4/effective-EP4 result: `phase1_profile_tp4.md`. The remainder of
> this file preserves the earlier TP7 blocker report for provenance.

Date: 2026-08-03 (Asia/Seoul)
Decision: **NO-GO / HOLD for Phase 2 under the required seven-GPU layout**

## Outcome

The smallest unchanged baseline entry point was attempted with only physical
GPUs 1-7 visible. vLLM rejected TP=7 during model configuration because Qwen3-
VL has 32 attention heads. The failure occurred before weight loading, worker
launch, warm-up, token generation, or any profiling range.

```text
ValueError: Total number of attention heads (32) must be divisible by tensor
parallel size (7).
```

The reproducer is `poc_flashvep/baseline_command.sh`; structured failure
evidence is in `poc_flashvep/results/baseline/smoke_failure.json`.

## Protocol Completion

| Item | Required | Completed |
|---|---:|---:|
| Fixed batch-size-1 smoke | 1 valid run | 0 |
| Warm-up | >=5 | 0 |
| Measured iterations | >=20 | 0 |
| Fixed input | Yes | Defined, not executed |
| `max_new_tokens` | Prefer 1 | Configured as 1, not executed |
| Per-rank/per-layer timing | Required | None |
| Profiling overhead comparison | Required | Not measurable |

## Stage Measurements

No duration was recorded for QKV projection, attention core, attention output
projection, residual + RMSNorm, router, dispatch, local experts, combine, or
the full MoE layer. `summary.csv` and `layer_breakdown.csv` deliberately leave
all unavailable measurement fields empty. Empty fields do not mean zero.

Installed source inspection established potential future boundaries:

- attention projections/core and decoder residual/RMSNorm in
  `vllm/model_executor/models/qwen3_moe.py`;
- internal router and FusedMoE orchestration in
  `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`;
- prepare/finalize in `fused_moe/prepare_finalize/naive_dp_ep.py`;
- all-gather dispatch and reduce-scatter combine in
  `distributed/device_communicators/all2all.py`.

No vLLM package was patched because the exact local expert backend is selected
at model load time and model load never occurred. Instrumentation could not be
validated against an unchanged baseline.

## Required Analysis Values

| Value | Result |
|---|---|
| `T_layer` | Not measured |
| `T_attention` | Not measured |
| `T_norm_router` | Not measured |
| `T_dispatch` | Not measured |
| `T_expert_max` | Not measured |
| `T_combine` | Not measured |
| `critical_rank` | Not measured |
| `expert_fraction` | Not measured |
| `exposed_nonexpert_fraction` | Not measured |
| `T_optimistic` | Not computable |
| Oracle layer speedup | Not computable |

There is also no first-tile latency measurement or estimate. The archived TP=8
FusedMoE CUDA-event totals are prior-system evidence only: they use eight
devices and do not split stages or measure whole-layer wall-clock latency.

## Vision Metadata

Current model config exactly identifies image token 151655 and video token
151656, plus vision boundary IDs 151652/151653. However, no current request was
executed, so current vision/text counts and token index ranges are unavailable.
Historical smoke counts were not copied into current results.

## Why Other Seven-GPU Layouts Were Not Substituted

vLLM 0.20 derives effective EP from TP x DP x PCP rather than PP. DP=7 changes
the offline single-request semantics by partitioning requests and filling idle
ranks with placeholders; PCP=7 is unsupported by the installed attention
backend; PP=7 leaves EP=1. Using four allowed GPUs or adding forbidden GPU 0
would change an explicit constraint.

## Phase 2 Decision

The acceptance criteria require both >=1.15x oracle layer speedup and >=15%
exposed non-expert time for a representative vision-heavy case. Neither value
exists, so the criteria are not met by evidence. The correct decision is
**NO-GO / HOLD** for the requested configuration. This is an infrastructure/
parallel-layout blocker, not a measured negative result about FlashVEP.

Debate options are to re-authorize TP=8 using GPU 0, approve TP=4/EP=4 on an
allowed subset, or separately approve a runtime change that supports exact
seven-rank EP. Phase 2 remains unstarted.
