# Online Dynamic Cut: Gate Status and Unexecuted Work

Status: NOT_RUN. No online Dynamic Cut serving result exists in this study.

## Source and execution contract

The numerical release is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
The study starts from RCC release `b53a8fed36a1706ac74754b53753f20edb8b913e`.
Implementation checkpoint `28b1cbf75b0b26a758e98e26c60af9860e7c498e` preserves the initial migration implementation.
Code and raw-evidence release: `71453683a69d600a7cabc667386df4dd1b622f7a`.
Later run archives preserve the exact process-local overlays.
The final summary identifies the publication and implementation revisions.

The models are Ouro Base 1.4B R4 and Huginn-0125 R32.
Both retain BF16 parameters, BF16 KV, the original checkpoints, and full recurrent work.
Native Mixed is off. The executor rejects combined Prefill/Decode recurrent batches.
The private RLT source remains unchanged.

## What the implementation does

The harness can commit different fixed cuts for different requests.
It preserves the complete prompt state across multiple token chunks.
It supports Cut 0 and Full Cut R as explicit endpoints.
It also supports ordinary native chunk-streaming PD_G1 as a separate control.

This infrastructure is not an adaptive scheduler.
The experiment configuration supplies each cut before P computation starts.
No component observes SLO slack and chooses a cut online.
The source rejects a cut change after reservation.
That check prevents an invalid receiver frontier or an inconsistent transfer protocol.

## Why the study does not report an online speedup

The frozen fixed-regime comparison combines completed homogeneous mechanism experiments.
It does not switch cuts within one live serving trace.
Its regime labels come from the experiment design.
The loaded-D profiles also initialize two background requests through Full Cut R.
Those background requests hold the initial Decode state constant across target-cut comparisons.
This setup is not a global static policy applied to every historical request.

No validated contention-aware Dynamic Cut simulator exists in this namespace.
No request-by-request ideal minima are used as a serving oracle.
No maximum stable SLO-constrained goodput is inferred from cohort makespan.
The primary expansion gate therefore remains unqualified.

## Admission and fallback requirements

An actual online policy would need a tentative cut and a transfer-time admission check.
It would need current Decode progress, queue state, KV capacity, and a calibrated SLO.
It must distinguish a ready Decode token from an imported Prefill continuation.
The coordinator's activation message alone does not make that distinction.

Before export, the policy must retain a valid source state.
If D rejects the continuation, P must resume the remaining loops safely.
The destination must acknowledge any revised cut before it accepts a different payload.
The source must not queue incompatible NIXL fragments before that acknowledgement.
The current fixed-cut connector does not implement this renegotiation.

These requirements describe possible future work. They are not tested online features.
No reinforcement learning or future-arrival information is used.

## Required nine-arm matrix

| Arm | Current evidence | W1 long-generation stable SLO-goodput |
|---|---|---|
| DP | NOT_RUN | NOT_MEASURED |
| DP + separate-phase Chunked Prefill | NOT_RUN | NOT_MEASURED |
| PD_G1 | Finite native streaming control, not a capacity campaign | NOT_MEASURED |
| Loop-Static | Finite fixed-target-cut maps | NOT_MEASURED |
| Loop-Dynamic | NOT_IMPLEMENTED | NOT_MEASURED |
| Token-Static | NOT_IMPLEMENTED | NOT_MEASURED |
| Token-Dynamic | NOT_IMPLEMENTED | NOT_MEASURED |
| Layer-Static | Intra-loop migration NOT_IMPLEMENTED | NOT_MEASURED |
| Layer-Dynamic | NOT_IMPLEMENTED | NOT_MEASURED |

The same missing-result status applies independently to both models.
No empty cell represents zero latency or zero throughput.

## GPU budgets and workloads

Actual mechanism experiments use 1P+1D pairs.
The runs cross GPU0/1, GPU4/5, and GPU6/7 where their configurations specify those pairs.
The initial Huginn correctness probes also use GPU2/3.
The worker maps record physical UUIDs and visible-device indices.
Clean timing uses one pair without other GPU experiments or owned burners on the shared fabric.
The study does not claim complete CPU-host exclusivity.

Four-GPU and eight-GPU pooled PD serving are NOT_RUN.
The prepared W1 calibration and test manifests each contain 200 full-text prompts.
Their controlled output targets average 448 tokens.
That number is not an observed generated output mean.
W1-LG, W1-Natural, W2, W3, and W4 serving-capacity campaigns are NOT_RUN.

The completed maps use diagnostic ShareGPT token prefixes.
They generate actual greedy tokens with explicit output caps and `ignore_eos=True`.
They are controlled generation, not Natural-EOS completion and not frozen-token replay.

## Limits and reproduction

Original-HF equivalence and downstream benchmark accuracy remain separate from internal migration parity.
This study does not establish lossless original-model equivalence.
It does not establish an online SLO guarantee.

The complete measured metric tables reside in `adaptive_recurrence_cut/`.
They include queue observations, TTFT, TPOT, ITL, E2E, memory, and transport accounting.
The static-map report explains the observed cut trade-offs.
The oracle-headroom report distinguishes fixed-regime composition from a dynamic oracle.
The token/loop/layer report states the logical-layer inclusion and equivalence limits.

See `adaptive_recurrence_cut/reproduce.md` for exact commands.
The raw run configurations and source archives are the authoritative execution record.
