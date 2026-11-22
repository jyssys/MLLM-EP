# Operator cost checkpoint

Date: 2026-11-21 KST. This file records an intermediate result.
It does not replace the final serving decision.

## Measurement contract

The model is Qwen3-8B with original BF16 and BI0.
The runtime uses the actual serving Linear, Attention and Norm APIs.
The controlled protocol uses P1 / D-suffix1 / D-decode1 and source credit4.
All workers use B8192/S1024 and KV block128.

The workload contains 16 original prompts per input bucket.
The output is fixed at8 tokens for this calibration.
The Token suffix allocation exactly matches aggregate GEMM token-layer rows.
The input prompts do not use filler or padding.

Each axis has three predeclared clean/profile/clean triplets.
The warmup uses all measured prompt shapes. This foreknowledge is diagnostic, not an online policy.
The GPU collector correlates kernels with actual operator calls.
It also exports already-collected CUDA API timestamps.
The collector does not enable GRAPH_TRACE.

## Observer gate

The gate compares profile latency with each triplet's mean clean latency.
The paired start-level 95% CI must stay within0.95–1.05.
The gate checks TTFT p95, mean TTFT, ITL p99, TPOT p95 and mean TBT.
It also requires complete kernel attribution and zero dropped or invalid records.

| Input bucket | Cut | Layer | Token |
|---|---:|---|---|
| 3K–6K | Last2 | `DIAGNOSTIC` | `DIAGNOSTIC` |
| 3K–6K | Last4 | `DIAGNOSTIC` | `DIAGNOSTIC` |
| 20K–36K | Last4 | `ISOLATED_KERNEL_MEASUREMENT` | `ISOLATED_KERNEL_MEASUREMENT` |

The short-input measurements fail the predeclared observer gate.
They remain in the raw evidence. They do not establish primary operator speedups.
The long-input measurements pass for both axes and both GPU roles.
These gates do not qualify continuous online profiling.

## Qualified long-input Last4 result

Values below are means across three independent profile starts.
Each start first averages the 16 real requests.
The measured scope contains prompt prefill or continuation kernels only.
It excludes subsequent decode kernels from these prompt totals.

| GPU role and axis | Kernel union per request | Attention kernels | Projection GEMM kernels |
|---|---:|---:|---:|
| P, Layer | 791.907ms | 287.453ms | 432.476ms |
| P, Token | 763.103ms | 258.186ms | 432.905ms |
| D, Layer | 94.329ms | 32.926ms | 50.534ms |
| D, Token | 126.338ms | 65.107ms | 52.069ms |

Projection GEMM includes QKV, output projection, MLP Gate/Up and MLP Down.
The table excludes LM-head kernels from that GEMM subtotal.
The kernel-union column also contains other measured prompt kernels.

Layer adds28.804ms of P kernel work in this control.
Its P Attention subtotal adds29.267ms.
The P projection GEMM subtotals differ by−0.429ms.

Layer removes32.009ms of D continuation kernel work in this control.
Its D Attention subtotal removes32.181ms.
Its D projection GEMM subtotal removes1.535ms.
Other operator differences offset part of those reductions.

These observations support an Attention-placement explanation at long input.
They do not show a large GEMM-only advantage at this token dimension.
They do not establish each operator's causal share of natural online TTFT.
P and D work can overlap. Their kernel costs are not added to predict TTFT.

## Remaining tests

1. Reduce short-input observer interference without changing the serving operators.
2. Test admission-only changes against the unchanged ITL failure fixture.
3. Preserve the default compiled full-model path in partial-prefill adapters.
4. Validate each runtime change before any clean serving claim.

SM utilization and HBM traffic are `NOT_MEASURED`.
CUDA API duration remains host time.
Kernel gaps that overlap host APIs do not identify a causal bottleneck by themselves.

## Evidence

The frozen contract is `dense_ar_longbench400_2d_20261121/configs/operator_repeat_observer_contract_v3.json`.
The analysis is `dense_ar_longbench400_2d_20261121/analysis/operator_repeat_observer_analysis_v3.json`.
Each profile retains its source capsule, raw kernels, API records and clean neighbors.
