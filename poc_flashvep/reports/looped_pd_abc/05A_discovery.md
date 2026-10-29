# A: recurrence-split discovery

Status: exact state transfer passed. Repeated serving evaluation is in progress.

## Execution contract

The source executes the complete prompt through recurrent loop `r`.
It then suspends that request. The destination executes the remaining prefill loops.
The destination executes every later decode token at the complete fixed depth.
This implementation does not divide autoregressive decode loops across GPUs.

`split.py:SplitConnector` implements the transfer boundary.
The source transfers valid depth-indexed KV and the complete prompt's recurrent state.
It also transfers request RNG state when the model uses that state.
The destination restores cache validity, position frontiers, and the prefill phase.
It does not recompute earlier prompt loops.
Under A, the upstream `prefill_complete` notification means the source assignment completed.
It does not mean the entire fixed-depth model prefill completed.
The destination must finish the remaining loops before it samples the first output token.

The initial implementation accepts a complete prompt in one chunk.
It rejects a prompt that exceeds that chunk.
It preserves a fixed carrier for Huginn and Dory input reinjection.
It uses a conservative destination copy barrier before resumed execution.
The serving measurements include that barrier.

## Measured transfer identity

The following values describe the 512-token correctness probe.
Each case passed three imports, including warmup.
Evidence: `MEASURED_NIXL_BYTE_IDENTITY_HEAVY_CORRECTNESS_ONLY`.

| Model | Cut | Valid KV bytes | Registered hidden bytes | Exact imports |
|---|---:|---:|---:|---:|
| Ouro R4 | 2 | 201,326,592 | 2,097,152 | 3/3 |
| Huginn R32 | 16 | 713,687,040 | 10,813,440 | 3/3 |
| Dory R2 | 1 | 58,720,256 | 5,242,880 | 3/3 |

The registered hidden buffer can contain padding beyond the useful prompt state.
The source zeros that padding before transfer.
The measured transfer payload includes the padding. It is not a zero-cost hidden vector.
These byte counts exclude host control messages, page-table metadata, and serialized RNG messages.
Their latency still contributes to measured request wall time.

The earlier operator probes also compared resumed KV and logits against uninterrupted execution.
Ouro passed 10 cases. Huginn passed 10 cases. Dory passed six cases.
Those probes use bounded generation and do not establish benchmark accuracy.
Their per-request memory registration is not the optimized serving path.

## Discovery controls

The runtime sweeps Ouro cuts 1/2/3 and Huginn cuts 1/8/16/24/31.
Dory has one intermediate cut, 1.
The discovery workloads use 512- and 2048-token prompts, concurrency 4, and output cap 8.
Controls include two colocated replicas, decode-local prefill, ordinary PD, and matched staged PD.
Decode-local prefill uses only the destination GPU within the two-GPU resource budget.
It is not evidence of two-GPU compute parallelism.

Repeated serving uses a separate output-cap-32 workload.
The selected intermediate cut comes from the completed cap-8 discovery set.
The selection reads no future outputs during serving.
The matched A control uses the same full-prompt chunk and staged G1 execution.

The original selected cuts were frozen before the evidence-I/O correction:

| Model | P128/P512 serving cut | P2048 serving cut |
|---|---:|---:|
| Ouro R4 | 3 | 1 |
| Huginn R32 | 1 | 16 |
| Dory R2 | 1 | 1 |

Each newly selected cut passed a separate 100-prompt bounded output comparison.
Those new cuts also passed nine additional byte-identity imports in total, including warmup.
The original qualified cuts retain their earlier evidence.
`configs/chosen_cuts.json` records the tuning and qualification sources.
The cap-128 serving regime remains separate from cap-32 measurement.

The recorder correction invalidates the original sweep's clean timing interpretation.
Its output and state checks remain correctness evidence.
`clean_cut_sweep.py` repeats all 42 tuning jobs with the nonblocking recorder.
It preserves the original choices and writes qualified new choices to `configs/clean_chosen_cuts.json`.
Canonical A timing confirmations use those new choices.
No old cut is described as optimal under the corrected protocol before that sweep completes.

## Structural limit and novelty

An individual request still has a sequential recurrence dependency.
A removes no prescribed model computation.
Its possible benefit is queue redistribution between P and D.
It adds a prompt-sized state transfer and destination prefill interference.

A generic logical-depth migration can reproduce this split.
Shared weights allow complete replicas without an unrolled parameter copy.
That deployment property does not prove a new scheduling algorithm.
No loop-specific novelty or accepted serving gain is established yet.

The state DAG is available in `dependency_contract.md`.
Final decisions require the strongest equal-budget baseline, not only full-prefill PD.
