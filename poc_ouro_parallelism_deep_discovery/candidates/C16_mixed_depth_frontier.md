# C16: Mixed-depth rows across GPU owners

## 1. Identity

Revision 1. Family: ready-frontier batching and distributed ownership.
Current verdict: `GENERIC_ONLY`.
The source and CPU audit is [here](../reports/mixed_depth_frontier_redteam.md).

## 2. Mechanism

Pool ready rows from different owners and recurrent depths into shared dense execution, while retaining each row's depth-indexed KV and original gate.

## 3. Relevant Ouro property

The same physical layer weights serve several logical depths.
The Ouro adapter runs that physical stack for every selected row.
KV and original gate state remain token-local and depth-aware.
Repeated weights permit this union, but generic alias-aware execution permits the same union.

## 4. Target scope

Both Base sizes, fixed R4, and the unchanged original adaptive gate.
The primary stage is decode at system C=1/2/4/8 and contexts 1K/4K/8K.
P2/P4 need later same-resource measurement.
P8 is projection only.
Prefill is outside this ready-decode-row screen.

## 5. Baseline and exposed cost

Use competent same-P TP plus full-KV continuous batching as the principal baseline.
Also include independent DP and generic attention/FFN disaggregation.
Give every baseline the same ready rows and physical weight aliases.
The intended saving is dense weight-load amortization or communication, not eliminated model work.

## 6. Operation sequence

AS-IS: each owner dispatches its ready rows, preserves KV, completes a recurrent pass, and consumes the original gate.
TO-BE: select compatible ready rows, transfer required inputs, execute a pooled union, return outputs, and consume the same original gates.
A generic per-row KV union is the required intermediate control.

## 7. DAG and ownership

`actual hidden -> QKV/RoPE -> matching KV write -> original attention -> O/residual -> original MLP -> loop norm -> original gate`

Every physical layer retains its actual four norm and residual operations.
The diagram omits their internal nodes only for space.
The next recurrent pass waits for the previous output and its required continuation decision.

| State | Owner and readiness | Lifetime |
|---|---|---|
| Physical weights | Operator workers after load | Immutable model revision |
| Historical KV | Request/depth/layer owner after committed write or permitted propagation | Original context lifetime |
| Current KV | Same logical owner after actual QKV/RoPE | Original context lifetime |
| Hidden row | Request owner after preceding output | Until next actual consumer |
| Pool scratch | Shared worker after explicit input transfer | Until outputs return |
| Survival/CDF | Token owner after actual gate signal | Until original exit decision |

## 8. Source of concurrency

Different active requests supply already-ready rows.
Constructed C=1/2/4/8 frontiers contain 1/2/4/8 rows, not 4/8/16/32 rows.
No later recurrent input or next-token input becomes ready merely because weights repeat.
C=1 gains no cross-request axis.
Fixed synchronous requests already supply a C-row conventional batch.

## 9. Exactness and quality

The intended class preserves the original graph, gate timing, and KV identities.
Batch-shape or backend changes do not guarantee identical BF16 arithmetic.
Future tests must compare gates, exit depths, committed KV, and free-running outputs.
Both-domain quality validation remains necessary if an implementation survives.
All model and quality tests are `NOT_RUN`.

## 10. Memory

Depth banks remain distinct.
Mixed-depth batching removes no KV slot or physical weight by itself.
Full-core migration needs history transfers, remote attention, or replicated KV.
MLP-only pooling needs input/output scratch, intermediate activations, metadata, and resident MLP weights.
A per-row page table can avoid WaveFront-style historical packing without merging KV values.

## 11. Communication

A remote MLP row sends and returns D BF16 values.
At D=2048, the logical round-trip payload is 8192 bytes per physical layer.
Also charge headers, synchronization, packing, queue delay, and return routing.
Full-core pooling needs an additional explicit attention/KV communication plan.
This screen assumes no topology bandwidth or P8 latency.

## 12. Critical path and costs

The pool path includes send, assembly, queued dense work, return, and dependent owner continuation.
Larger batches can amortize weight reads but can also idle other workers.
Compare with the best same-P generic union, not only small independent depth buckets.
No measured critical-path reduction follows from this screen.

## 13. Nearest prior works

WaveFront implements full-depth mixed-row unions and explicit per-row depth slots.
CDB implements continuous depth scheduling but restricts the inspected full-KV path to homogeneous slots.
NanoFlow implements operation-level nano-batching with explicit DAG dependencies and overlap.
Generic paged attention already supports per-row historical addresses.
The report contains pinned primary links and scope limits.

## 14. Prior evidence and changed assumption

The new assumption is that several ready owner rows can share one larger dense batch.
This differs from a same-batch remote MLP component test.
It does not remove remote transfer costs or establish a new axis.
No previous GPU test repeats in this block.
The full-KV union control prevents a local CDB addressing limitation from inflating novelty.

## 15. Smallest falsifier

Compare identical online ready traces under depth buckets, generic full-KV union, and cross-owner pooling.
Preserve original gate availability and charge every transfer and waiting interval.
Stop if the benefit disappears against generic union.
That outcome permits an implementation improvement, not an architectural claim.

## 16. Results and uncertainty

Eight constructed CPU frontier cases pass their legality checks.
Four controls reject missing inputs, unavailable gates, duplicate active request rows, or continuation after exit.
The address control preserves 512 identities and checks 112 historical reads.
The generic union equals the candidate union in every constructed case.
Latency, quality, and ownership-specific benefit remain `NOT_RUN`.

## 17. Compatibility matrix

| Scope | Result |
|---|---|
| 1.4B fixed R4 | Legal multi-request union, no new width beyond generic control |
| 1.4B original adaptive | Requires actual preceding gate decisions, no quality test |
| 2.6B fixed R4 | Same dependency result, more physical layers |
| 2.6B original adaptive | Same gate-readiness condition, no quality test |
| P2 | Possible ownership pooling, no incremental result |
| P4 | Same condition, greater small-C resource-idling risk |
| P8 | Projection only, no hardware performance claim |

## 18. Loop-specificity controls

An aliased logical unrolling obtains the same weight-compatible union.
An R1 control also amortizes dense weights across requests.
Materializing four weight copies changes memory but is not the strongest conventional control.
The inspected WaveFront implementation already uses repeated-weight unions with separate depth KV.
No recurrence-specific incremental benefit survives these controls here.

## 19. Complexity and next PoC

Do not build an engine from this screen.
The smallest next action is the paired online ready-trace comparison in field 15.
A later implementation would need cache addressing, graph buckets, gate routing, and owner queues.
No supported implementation-time estimate exists before a surviving cost argument.

## 20. Verdict

Do not shortlist C16 as a new execution axis.
Retain the full-KV mixed-row union as a generic implementation control.
Reopen only with an ownership-specific saving that survives that control and same-resource TP/DP/disaggregation.
This verdict does not prove that every recurrence-aware ownership scheme must fail.
