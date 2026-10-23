# Correctness evidence

## Isolated pretrained GPU qualification

All eight physical GPUs completed the registered numerical queue.
Each GPU used one independent model process.
Even GPUs tested P128 and P512.
Odd GPUs tested P1536 and P2048.
Every condition compared Atomic, fixed G2, fixed G3, and scripted legal C/G switching.

The queue contains 16 input/device conditions and 64 execution records.
Atomic supplies one reference per condition.
All 48 non-reference comparisons matched the reference output token IDs exactly.
Hidden states, logits, and all 96 depth/layer KV namespaces also matched exactly.
The observed maximum relative L2 was 0 for each tensor category.
The preregistered tolerance remains 0.03.

Every prompt recorded exactly `4 × P` token-loop work.
All eight generated output tokens in each execution had exit depth four.
The scripted path completed C512/G2 and its remaining G2 before opening a new C1024/G4 chunk.
The native G3 path also completed its legal G1 remainder.
No prompt span changed size while staged work remained unfinished.

The observer recorded 3,056 native submissions across these probes.
Every submission had a nonnegative service interval.
The maximum recorded same-GPU clock-alignment bound was 0.066094 ms.
This bound describes the probe's anchor alignment, not inter-GPU clock synchronization.

These inputs are isolated numerical probes with eight bounded output tokens.
They do not prove full-answer accuracy or universal batched output parity.
Matched clean/event controls and batched fairness comparisons are separate queues.
The subsequent full-answer quality pilot is reported in `quality.md`.
It does not establish uniform equivalence across all arms.

## Software fixtures

Seventeen local CPU tests passed.
They cover completion debt, request identity, fixed/variable frontiers, enumeration, cost-path labels, feedback, and budget admission.
Native integration tests verify prompt work, final outputs, released KV blocks, and released state slots.
The integration models are tiny software fixtures.
They are not pretrained-model quality or GPU performance evidence.

## Scope limits

Native preemption is disabled in the registered configuration.
Native event dependencies remain unchanged.
No unfinished CUDA tensor is read for a scheduling decision.
The new method does not change weights, BF16 precision, masks, RoPE, or recurrent depth.
The previous batched BF16 geometry mismatches remain preserved historical evidence.
The isolated passes do not remove those earlier limitations.

## Subsequent natural generation

All 798 quality generations recorded Full R4 output depths.
J1 and G1 matched correctness labels on all 114 benchmark questions.
They matched token sequences on 113/114 questions.
C1A/C1S had observed GSM8K regressions versus stock Atomic.
The short numerical probes do not explain or erase those regressions.
The exact batching/numerical cause remains unresolved.
No blanket batched-output or all-arm accuracy PASS is claimed.
