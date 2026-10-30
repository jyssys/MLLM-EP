# Token, Loop, and Logical-Layer Migration

Status: finite Loop migration verified. Token and intra-loop Layer migration are NOT_IMPLEMENTED.

## Execution contract

The study uses Ouro BF16 R4 and Huginn BF16 R32. All Decode tokens retain the complete recurrent depth.
The qualified numerical release is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
The experiment base is RCC release `b53a8fed36a1706ac74754b53753f20edb8b913e`.
Code and raw-evidence release: `71453683a69d600a7cabc667386df4dd1b622f7a`.
The private RLT source remains read-only. The source lock records its hashes.

Native Mixed and cross-depth Prefill/Decode GEMM co-batching are disabled.
All primary workers use separate-phase `refill` scheduling.
An execution guard rejects a batch that contains both Decode recurrent rows and Prefill recurrent rows.
This setup does not represent the complete optimized Sarathi-Serve serving path.

## Structural comparison

These entries describe state requirements. They are not measured speedups.

| Axis | P computation | D continuation | Required migration state | Implementation status |
|---|---|---|---|---|
| Token | Full-depth prompt prefix | Full-depth prompt suffix | Valid prefix KV, positions, RNG, and model-specific state | NOT_IMPLEMENTED |
| Loop | All prompt tokens through cut n | Remaining recurrent loops | Valid depth-indexed KV and all intermediate prompt rows | Finite GPU qualification and fixed-cut timing complete |
| Logical layer | All prompt tokens through an unfolded layer boundary | Remaining logical layers | Valid per-layer KV, intermediate rows, residual state, and injection state | Intra-loop migration NOT_IMPLEMENTED |

The previous A connector rejects prompts longer than one configured chunk.
Removing this check would not create valid multi-chunk migration.
The original wavefront runner reuses a bounded ring of prompt-state rows.
Those rows cannot retain every token at one recurrent frontier when the prompt exceeds the ring.

The new process-local overlay retains absolute prompt-state rows.
It keeps the original causal token chunks and per-depth progress frontiers.
A P-side cut barrier prevents work above the committed cut.
The connector transfers only consumer-valid KV and the required hidden rows.
The source state remains leased until transfer acknowledgement.
New GPU checks establish finite correctness independently of the previous A release.
They do not establish universal original-model losslessness.

The matched Full Cut R control sends one completed-prompt snapshot.
The existing PD_G1 connector can stream completed full-depth token chunks.
These controls are not identical transfer protocols.
The study retains native streaming PD_G1 as a separate measured control.
A snapshot-versus-streaming difference is not a recurrent-partitioning benefit.

## Logical-layer inclusion and equivalence

Ouro has 24 physical core layers and four recurrent iterations.
Its unfolded recurrent core has 96 logical transformer layers.
Each Ouro iteration also includes its final RMSNorm and the gate required by the qualified path.
The matching recurrent boundary follows those transition operations.
A raw output after physical layer 24, before that norm, is not the same boundary.
Huginn has two Prelude layers, four recurrent core layers, and two Coda layers.
Its R32 path has 128 logical recurrent core layers, plus Prelude and Coda.
Huginn also carries reinjection state. A valid layer cut must preserve that state.
Its recurrent boundary includes the carrier with both state and injection.
An internal core boundary must not apply the reinjection adapter twice.

Let L be the set of valid logical-layer cuts.
Let R be the recurrent-loop boundary subset.
Then R is a subset of L when both methods permit the same boundary states and costs.
The exact Layer oracle cannot be worse than the exact Loop oracle under identical constraints.
An implementation that exposes only recurrent boundaries is not a complete Layer-Dynamic baseline.

A logical-layer cut at the same recurrent boundary uses the same computation and state-transfer protocol.
This case is `EQUIVALENT_MECHANISM`.
Renaming that case does not establish Loop-specific novelty.
Both workers retain shared physical weights in the matched same-model comparison.
Duplicating weights across logical layers would create an artificial disadvantage for the generic baseline.

Measured weight storage is 2,869,305,346 bytes per Ouro worker.
It is 7,129,953,600 bytes per Huginn worker.
These values remain unchanged across the measured cuts.
The current implementation does not demonstrate a cut-dependent weight-residency saving.
The generic same-model comparator must also retain physical weight aliases.

Token-prefix migration needs a separate correctness protocol.
Huginn requires valid prefix Prelude and Coda KV, not only recurrent-core KV.
The consumer must also preserve RNG progression for suffix initialization.
An internal Layer cut needs per-layer validity and an unfinished-loop cursor.
It must preserve the residual and injection state required by the next operation.
These protocols are not implemented by relabeling the existing Loop connector.

Possible practical differences include serialization complexity and online search overhead.
They also include state-frontier management and memory requirements.
These differences require measurements. They are not established benefits.

## Existing A and Mixed

A was a previous implementation in this project. It was not a native external PD feature.
Native RLT Mixed already packs Prefill and Decode recurrent rows into shared-weight execution.
RCC Stage 1 established that A plus Native Mixed uses this existing capability.
The earlier finite Ouro speedup does not prove a Dynamic Cut benefit.
The earlier Huginn result did not show a repeatable makespan benefit.
Those frozen results and their novelty verdict remain unchanged.

## Related work

| Work | Relevant mechanism | Difference from this study |
|---|---|---|
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | Separate Prefill and Decode resources under latency objectives | It is not evidence of recurrent partial-Prefill migration |
| [Splitwise](https://arxiv.org/abs/2311.18677) | Separate inference phases and transfer request state | Phase separation alone does not establish a Loop-specific contribution |
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Chunked Prefill and mixed serving schedules | The current primary comparison explicitly excludes fused Mixed execution |
| [Layered Prefill](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html) | Layer-group scheduling for MoE serving | In-worker scheduling differs from cross-GPU Dense Prefill-compute migration |
| [Load-aware Prefill Deflection](https://arxiv.org/abs/2607.02043) | Move full Prefill to a Decode worker when queue and TBT conditions permit | The Cut-0 endpoint and generic load-aware fallback have strong prior work |
| [DuetServe](https://proceedings.mlr.press/v306/gao26m.html) | Adaptive SM partitioning for Prefill and Decode within one GPU | It does not establish cross-GPU recurrent-state migration |

Adaptive load balancing and Prefill borrowing are not new by themselves.
Any contribution must survive tuned Token-Dynamic and Layer-Dynamic comparisons.

## Current limits

No Token-Dynamic or intra-loop Layer-Dynamic GPU result exists in this namespace yet.
No structural argument in this report is a measured latency improvement.
Stage 2 must justify those implementations before the experiment scope expands.
The final report must distinguish a blocked comparison from a measured negative result.

The same-boundary structural verdict is `EQUIVALENT_LAYER`.
Claiming a new partition axis from that equivalent boundary is `NO_GO_LOOP_SPECIFIC_NOVELTY`.
This statement does not prove that every practical Loop-aware implementation lacks value.
Practical search, packing, serialization, and implementation-cost advantages remain NOT_MEASURED against a complete generic baseline.
