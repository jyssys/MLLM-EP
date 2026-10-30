# Failures, negative results, and unmeasured items

## Instrumentation issue

Version 1 counted `aten::mm` and related operators.
Torch inference mode exposes `aten::linear` before decomposition.
The first operand counters therefore remained empty.
That result did not mean that the model ran no GEMMs.
Version 2 records `aten::linear` and includes selected real CUDA kernel traces.
The original Version 1 evidence remains unchanged.

The Torch profiler can emit nested duplicate Linear annotations under this dispatch mode.
Selected GPU-kernel counts avoid that annotation duplication.
No annotation count is presented as a physical kernel count.

## Negative measured results

Huginn Native Mixed does not improve cohort makespan over A_sep in the quiet probe.
Its speedup is 0.99× with a paired 95% interval of 0.96×–1.02×.
Ouro A_sep is slower than PD_G1 in the same small regime.
Its speedup is 0.90× with an interval of 0.84×–0.96×.

Ouro Native Mixed improves mean completion but has higher ITL P99 than PD_G1.
Huginn Native Mixed has higher ITL P95 than A_sep.
Neither result establishes decode SLO protection.

The fixed-cut fused execution is already present in A plus Native Mixed.
No incremental execution mechanism beyond that existing path was identified.
This is a novelty limitation, not a failure to execute combined tensors.

## Runtime warnings

Some UCX agents print CUDA-context errors during process teardown.
All 36 experimental runs return complete status and valid output records.
Their work counts and applicable byte checks pass.
The warnings remain in the logs.
They are not hidden or treated as proof of a runtime failure during measured work.

## Unmeasured axes

| Item | Status | Reason |
|---|---|---|
| Conditional admission | NOT_RUN | No additional fixed-cut fusion mechanism beyond existing A/Mixed |
| W1 long-output controlled workload | NOT_RUN | Stage-1 discovery stop |
| W1 Natural EOS representative workload | NOT_RUN | Stage-1 discovery stop |
| W2/W3/W4 | NOT_RUN | Stage-1 discovery stop |
| 512-token and multi-chunk continuation | NOT_RUN | No justification for a larger migration implementation after the novelty audit |
| DP and DP_Mixed equal-budget serving | NOT_RUN | No architecture-advantage or capacity claim is made |
| SLO calibration and goodput | NOT_MEASURED | No later serving phase |
| 4GPU / 8GPU RCC | NOT_RUN | No Stage-2 GO |
| GSM8K / MATH accuracy | NOT_MEASURED | Mechanism-only discovery scope after gated stop |
| Original-HF equivalence | NOT_ESTABLISHED | Predecessor Huginn triage remains 5/8 |
| Exclusive full-E2E GPU operation shares | NOT_MEASURED | One-core heavy profiles do not cover clean E2E |
| Full GPU idle time | NOT_MEASURED | NVML utilization is not exact kernel busy time |
| Graph workspace breakdown | NOT_MEASURED | Peak memory does not isolate graph workspace |
| Same-operand separate/fused core latency | NOT_MEASURED | No isolated paired operator microbenchmark |

Huginn's 4096-token context prevents an unmodified 4K-prompt plus 512-output condition.
The study does not override this checkpoint limit.
It also does not replace either required model with a proxy.
