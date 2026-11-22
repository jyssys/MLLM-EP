# Decode-tail intervention contract

Status: `FROZEN_DESIGN`. Date: 2026-11-21 KST.

The user removed the wall-clock deadline. Correctness and safety gates remain unchanged.
The campaign will complete the declared scope or document a scientific stop gate.
This change does not authorize unsafe insertion or weaker baselines.

## Observed failure

Evidence: `MEASURED_ONLINE`, one discovery start per condition.
The workload is Discovery200 at QPS1.8 with seed2026112291.
The matched paths use BI0, B8192/S1024, source credit4 and Mode0.

| Method | Client ITL p99 | SLO-goodput |
|---|---:|---:|
| Native PD | 16.656ms | 55.005 output tokens/s |
| Layer Last4 | 35.587ms | 31.205 output tokens/s |
| Token Last4-equivalent | 35.196ms | 34.920 output tokens/s |
| Hybrid Last4 / suffix0.125 | 48.587ms | 21.926 output tokens/s |

These offloading paths fail decode protection. Their TTFT reductions do not establish serving success.
These values are finite-cohort results. They do not measure sustainable capacity.

## Competing explanations

1. Mixed prefill/decode steps delay active decode requests.
2. Import, hidden reads or host synchronization delay active decode requests.
3. Compile or graph changes increase execution cost.

The default-compiled stock Token path also shows decode tails.
This result rejects compile alone as a sufficient explanation.
Token has no hidden-state handoff. Its tails reject hidden transfer alone as a sufficient explanation.
Neither result identifies the remaining cause by itself.

## Admission-only intervention

P keeps its continuous scheduler and source credit4.
D admits one whole request at a time.
D requires prior request completion and a device/NIXL idle certificate before admission.
The intervention applies to Native, Layer, Token and Hybrid controls.

This protocol changes actual D batching. It is a causal control, not unrestricted continuous serving.
It does not predict a future decode-safe window.
It can reduce TTFT or throughput. Those regressions remain valid outcomes.

The strong comparison retains unrestricted, default-compiled Native PD.
The same-guard Native control isolates the admission change.
Client timestamps, output generation and SLO thresholds remain unchanged.
The system does not buffer tokens to alter measured ITL.

## Validation sequence

1. N1 strict KV, hidden and greedy-output comparison.
2. N8 strict comparison with same-geometry original-model teachers.
3. Natural32 coverage, import, lifetime and task-quality checks.
4. Natural400 state, coverage, lifetime and task-quality checks.
5. Same-trace clean online measurement without state observers.
6. Independent paired-start confirmation of eligible conditions.

Native-relative ITL p99 and TPOT p95 require a 95% CI upper bound of at most1.05.
Absolute SLO thresholds remain TTFT5000ms, every ITL34ms and TPOT33ms.
Correctness does not establish decode safety.

## GPU measurement

A separate collector records concurrent kernels and optional copies.
It does not record CUDA APIs, external correlation or graph traces.
The collector must pass clean/profile/clean checks before its results become primary evidence.
The checks cover the full trace and matched requests exposed to the ROI.
An observed kernel gap does not establish a future executable window.
