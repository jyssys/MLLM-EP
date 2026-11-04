# Huginn recovery: independently repeated operational gate

Evidence: measured serving results, actual CUPTI kernel intervals, and exact internal-runtime comparisons.

Huginn R32 passed the declared same-window operational gate in three independent 512-request starts.
Each start uses one predeclared approximately 32-second GPU capture within that same serving execution.
The result does not establish physical peak-FLOP saturation or indefinite sustainable capacity.

## Workload and contract

- Model: Huginn-0125, original BF16, fixed R32.
- Input/output: 2048/16 tokens. Generation uses a controlled output cap and ignores EOS.
- Dataset: 128 preserved ShareGPT/LongBench controlled prefixes, explicitly reused cyclically for 512 requests.
- Nominal arrival rate: 1.971 requests/s. Independent Poisson seeds vary between starts.
- P/D slots: 2/8. Full-prompt P execution retains the qualified native NIXL path.
- Additional work: twelve independent D-local 16-token, full-R32 Prefill requests per run.
- Probe admission defers new local probes while more than two background requests occupy the recurrent Decode phase.
- These probes are not P-to-D continuations. They do not reduce the background Prefill workload.

## Same-run gates

| Start | P union % | Service proximity % | Device-qualified queue growth | D TPOT p95 ms | D ITL p99 ms | Minimum D free blocks | Profile wall change % | Complete probes in interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 95.513 | 106.934 | 1.015625 | 121.747 | 188.025 | 3520 | 0.295 | 5 |
| 2 | 95.512 | 107.175 | 14.656250 | 127.562 | 190.541 | 3520 | 0.809 | 5 |
| 3 | 95.785 | 109.193 | 11.093750 | 123.788 | 190.243 | 7648 | -0.085 | 5 |

The unchanged thresholds are 80% kernel union, 85% service proximity, and at most 5% matched-wall profiler perturbation.
The D TPOT gate uses 167.914355 ms. The frozen token-ITL p99 limit is 192.821254 ms.
Service proximity uses the strongest recorded matched finite reference, 1.516134218 requests/s.
A value above 100% does not establish a universal capacity bound.

| Start | Whole-run ITL p99 ms | Whole-run maximum ITL ms |
|---|---:|---:|
| 1 | 125.038 | 200.196 |
| 2 | 204.464 | 1165.952 |
| 3 | 185.823 | 465.105 |

The ITL contract is a percentile contract. It does not require every token interval to remain below the p99 limit.
The table retains maximum outliers. The capture interval does not replace whole-run latency reporting.
Start 2 fails the whole-run ITL p99 limit: 204.464 ms exceeds 192.821254 ms.
Thus three same-window passes do not establish three whole-run Decode-SLO passes.
Large method-performance expansion is deferred pending matched-clean tail analysis and safe mitigation.

## Exactness and measurement scope

All three starts match the no-probe native output reference for their 512 background requests.
Each start completes all twelve retained local probes with exact inputs, outputs, and depth against the native probe reference.
Each start records 33,806,336 recurrent token rows. This equals the full background work plus the probe work.
The probes add 6,144 recurrent token rows. The background Prefill contains 33,554,432 recurrent token rows.
Thus probe rows equal 0.018310546875% of background Prefill rows. This structural ratio is not a measured FLOP or speedup ratio.
The probe does not establish that D can accept a useful fraction of all serving Prefill work.
The separate eight-request state qualification checks recorded prompt KV and hidden states.
The 512-request runs do not hash every intermediate tensor. Original-HF fidelity and downstream accuracy remain separate limitations.
No measured model-graph captures occurred in the three confirmed runs.

## Queue audit

The inherited phase-only counter reports zero ready requests during synchronous submission.
Its result remains in the raw evidence and is not relabeled as a pass.
The new audit identifies D-credit-clear P-slot waiting and verifies actual P kernel activity at each queue timestamp.
It excludes timestamp uncertainty and does not substitute a host service span for GPU activity.
The unchanged queue thresholds require positive half-window growth and a maximum of at least two ready requests.
The independent repeats use this audit. Phase-only, host-ownership, and ticket-only counters remain separate sensitivity results.

## Profiling recovery and unresolved cause

Native Nsight tracing repeatedly stalled inside NIXL/UCX CUDA IPC with CUPTI in the native stack.
Disabling the supported NIXL progress thread did not resolve that stall.
A late-initialized 32-second activity-only capture failed with a CUDA launch error. Its data is not accepted.
The accepted path initializes CUPTI before model CUDA contexts and preallocates sixteen activity buffers.
A 400,000-kernel eager/graph stress test records every expected kernel without dropped or invalid records.
The three model captures also report zero dropped and invalid records.
Early initialization and the increased buffer count are initially confounded. A separate three-buffer control tests that distinction.
The exact driver lock cycle remains ROOT_CAUSE_INCONCLUSIVE. The repeated result establishes a usable measurement path, not a vendor-level root cause.

## Next gate

Both models may enter small exact Token/Loop migration tests.
The local-probe result does not qualify a whole-prompt recurrent continuation or any online speedup.
Method expansion still requires exact connector and native-reference checks.

Raw qualification ledger: `state/admitted_buffer_recovery.json`.
Each referenced run retains configuration, source hashes, output IDs, device traces, and failure-independent evidence.
