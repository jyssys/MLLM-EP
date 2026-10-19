# Depth-sharded verification: direct-neighbor audit

Revision 1. This is a source and cost screen, not a candidate card.
Asynchronous speculation, staged verification, private KV, rollback, and distributed cancellation already have direct prior implementations.
The proposed Ouro placement still needs a matched cost test.
No source result establishes its advantage over TP on the existing mixed-depth frontier.
No GPU, model, or engine execution occurs here.

## Verified identities and versions

| Source | Latest observed arXiv version | Official code pin |
|---|---|---|
| PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decoding | [2505.01572v1](https://arxiv.org/abs/2505.01572), 2025-05-02 20:29:31 UTC | [9a64774f](https://github.com/BradMcDanel/PipeSpec/tree/9a64774ff9589278bcc16f4a07f2c3974c929a52) |
| PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation | [2407.11798v2](https://arxiv.org/abs/2407.11798), 2024-11-16 23:19:51 UTC | [main 123a824d](https://github.com/AutonomicPerfectionist/PipeInfer/tree/123a824d9b53e56efa2a7645894d30a9a1135ed2) |
| Speculative Pipeline Decoding: Higher-Accuracy Drafting with Hidden Latency via Pipeline Parallelism | [2605.30852v3](https://arxiv.org/abs/2605.30852), 2026-07-27 01:27:11 UTC | [186f3667](https://github.com/yuyijiong/speculative_pipeline_decoding/tree/186f366764b213f5157fa0d15bce9fb59c0182db) |

PipeSpec also has a later published [ACL Findings 2025 version](https://aclanthology.org/2025.findings-acl.669.pdf).
Its experiment values differ from arXiv v1.
This audit uses the published version for experiment scope and retains both identities.
The ACL record links the official code.
The primary SPD HTML and PipeInfer author's announcement also identify their repositories.
Current repository pins are not necessarily the exact revisions behind every paper datapoint.

## PipeSpec: asynchronous hierarchy of complete models

PipeSpec connects independent pretrained models as asynchronous producers and consumers.
Intermediate models verify earlier drafts while drafting for later models.
A rejection propagates toward earlier stages and truncates their speculative suffixes.
It requires no additional training for this execution scheme.
These models are not successive Ouro depth blocks. [Published paper, §§2.3,3.1–3.2](https://aclanthology.org/2025.findings-acl.669.pdf)

The inspected code provides a concrete greedy protocol:

- `step` and `perform_verification` use argmax.
- `compute_verified_ids` accepts the matching contiguous prefix.
- `perform_verification` crops the verifier's cache after that prefix.
- `build_draft_inputs` finds divergence and crops the earlier model's cache.
- `_chain_process` propagates sync requests and waits for acknowledgements.

Each worker owns its model and its `DynamicCache`.
Shared CPU tensors carry token histories and synchronization counters, not a common hidden-state pipeline.
The last model's output buffer supplies the final output.
The inspected path does not implement general sampling rejection. [Code:33–150,476–489,513–665,698–739](https://github.com/BradMcDanel/PipeSpec/blob/9a64774ff9589278bcc16f4a07f2c3974c929a52/decoding.py#L33)

Published experiments use four A100-40GB GPUs with NVLink.
The 70B model uses 4-bit weights across two GPUs.
All experiments use greedy decoding with a 512-token maximum.
The strongest directly matched control uses the same model chain synchronously.
Some other-method comparisons use published numbers, not common-runtime reruns.
The throughput theorem counts tokens per modeled step.
It does not prove a same-resource wall-clock advantage with communication and batching costs. [Published §§3.3,4.1–4.2](https://aclanthology.org/2025.findings-acl.669.pdf)

The code times the parent's completion wait after signaling workers to start.
Workers process the prompt after that signal.
This is not the campaign's decode-only timing contract. [Timing and worker entry](https://github.com/BradMcDanel/PipeSpec/blob/9a64774ff9589278bcc16f4a07f2c3974c929a52/decoding.py#L698)

## PipeInfer: direct distributed KV and cancellation prior

PipeInfer runs separate target and draft pipelines.
The target can process canonical single-token inference while speculative microbatches remain in flight.
Each target run gets a private sequence range.
Accepted state joins canonical sequence zero.
Ordered transactions propagate cache changes through the target stages.
Invalid runs receive cancellation signals keyed by run ID.
Empty payloads preserve message order after cancellation. [Paper, §§IV-A–IV-D](https://arxiv.org/html/2407.11798v2)

The code confirms that publication need not transfer historical KV tensors between stage owners.
`llama_kv_cache_seq_cp` inserts destination sequence IDs into existing cache-cell metadata.
Its distributed wrapper sends source/destination IDs and position bounds through ordered MPI transactions.
The speculative driver removes rejected suffixes, publishes accepted entries to sequence zero, and shares that accepted interval with other sequences.
`check_for_cancel` compares run lineage and position bounds before invoking `llama_cancel_run`.
The latter propagates IDs backward through the pipeline. [Cache metadata and transactions](https://github.com/AutonomicPerfectionist/PipeInfer/blob/123a824d9b53e56efa2a7645894d30a9a1135ed2/llama.cpp#L1699), [publication and cancellation](https://github.com/AutonomicPerfectionist/PipeInfer/blob/123a824d9b53e56efa2a7645894d30a9a1135ed2/examples/speculative/speculative.cpp#L585), [cancel transport](https://github.com/AutonomicPerfectionist/PipeInfer/blob/123a824d9b53e56efa2a7645894d30a9a1135ed2/llama.cpp#L9981)

The model's physical layers occupy different nodes.
Their KV remains on the corresponding owner while activations cross stage boundaries.
The inspected source does not establish exact allocated-cache capacity per rank.
Canonical runs cannot simply stop midway because later cache-publication commands rely on their valid entries. [Author's implementation description](https://github.com/ggml-org/llama.cpp/discussions/10344), [paper §IV-D.3](https://arxiv.org/html/2407.11798v2)

The method needs no new training, but uses a separate pretrained drafter.
Its principal experiments use quantized models on CPU clusters with up to 32 nodes.
The strongest in-runtime controls are pipeline-parallel iterative and standard speculative inference.
The paper reports identical greedy outputs for those tests.
Its distribution argument requires correct target distributions and isolated cache histories.
It does not establish Ouro BF16 or original-gate parity. [Paper, §§IV-E,V](https://arxiv.org/html/2407.11798v2)

Mixed CPU/GPU results use MI60, P40, Titan V, and RTX3090 devices across four nodes.
They use another backend revision and are not directly comparable with the CPU results.
The author identifies unintended CPU execution of stage-boundary layers in that GPU backend.
The public GPU branch pin is `05fbad5b078fc2b04645dc44bd7029c728c0b51a`.
This audit does not inspect that entire branch or treat it as an H100 BF16 baseline. [Paper §VI](https://arxiv.org/html/2407.11798v2), [author's backend limitation](https://github.com/ggml-org/llama.cpp/discussions/10344)

## SPD: target-depth pipeline plus a trained drafter

SPD pipelines different tokens through one target's disjoint layer stages.
Its PDM uses already available target features and runs concurrently with the stage forward.
Streaming target verification accepts the next token or flushes the speculative suffix.
The paper trains the PDM and its feature projections while freezing the target.
Its results therefore do not supply a permitted training-free Ouro drafter. [Paper, §§3.1–4.1](https://arxiv.org/html/2605.30852v3)

The official distributed code has explicit stage-local cache shards.
Rank zero retains proposal distributions, intermediate snapshots, and pipeline positions.
On rejection, the controller removes rejected snapshots and restarts from the correction token.
Workers wait for pending transport, crop their local cache, and clear buffered activations.
Invalid downstream messages retain an explicit invalid flag. [Cache shards](https://github.com/yuyijiong/speculative_pipeline_decoding/blob/186f366764b213f5157fa0d15bce9fb59c0182db/distributed_inference/cache.py#L34), [controller:437–487](https://github.com/yuyijiong/speculative_pipeline_decoding/blob/186f366764b213f5157fa0d15bce9fb59c0182db/distributed_inference/rank0_controller.py#L437), [worker:150–165,323–353](https://github.com/yuyijiong/speculative_pipeline_decoding/blob/186f366764b213f5157fa0d15bce9fb59c0182db/distributed_inference/stage_worker.py#L150)

Published wall-clock experiments use Qwen3.5-4B/9B on H20 servers.
They allocate one GPU per target stage plus a dedicated PDM GPU: `n+1` ranks.
AR and EAGLE-3 use one H20.
EAGLE-3 uses SGLang, while SPD/PPSD use the authors' implementation.
These controls do not answer the same-P TP question.
Branched-tree wall-clock execution remains future work in §4.4. [Paper, §§4.1–4.4,6](https://arxiv.org/html/2605.30852v3)

Current code calls itself v11.
Its README labels archived v10 as the paper implementation and says their checkpoints are not interchangeable.
The distributed README requires v11 checkpoints.
The precise connection between each published timing and this code pin therefore needs clarification.
Do not assign current-code behavior to every paper experiment. [Version statement](https://github.com/yuyijiong/speculative_pipeline_decoding/blob/186f366764b213f5157fa0d15bce9fb59c0182db/README.md#L47), [archive differences](https://github.com/yuyijiong/speculative_pipeline_decoding/blob/186f366764b213f5157fa0d15bce9fb59c0182db/old_version_v10/README.md#L3)

### Sampling guarantee limit in current code

Greedy verification requires target argmax equality.
Sampling usually accepts with the proposal/target ratio and corrects with normalized `max(p-q,0)`.
However, the current helper samples from p when residual mass is at most `eps*vocab_size`.
That fallback is not universally distribution-preserving. [Sampling helper:103–154](https://github.com/yuyijiong/speculative_pipeline_decoding/blob/186f366764b213f5157fa0d15bce9fb59c0182db/pipeline_model.py#L103)

A minimal analytic counterexample uses two supported tokens in a 100,000-token vocabulary:

```text
p = (.500, .500, 0, ...)
q = (.505, .495, 0, ...)
residual mass = .005
FP32 eps * 100000 = .011920928955078125
fallback output distribution = (.5025, .4975, 0, ...)
correct residual output       = (.5000, .5000, 0, ...)
```

The rejection probability is `.005`.
Sampling p after rejection adds `.0025` to the first token's accepted mass.
The resulting total-variation distance is `.0025`.
A CPU rational-arithmetic check confirms this calculation.
This is a helper-level counterexample, not an observed model-quality regression.
It does not invalidate the separate greedy branch or the mathematical rejection identity.

## Transfer to the proposed Ouro ownership

The hypothesis assigns recurrent depths, not independent branches, to stationary owners.
Each owner retains the full common physical core and the KV for its assigned depths.
At P2, an owner can still co-batch rows from multiple assigned depths.
At P4, one owner per depth is a possible fixed-R4 mapping.
No shallow state substitutes for an unavailable target state.

This placement has a concrete distinction from branch-owned replicas.
All branches' depth-r ancestor KV can remain on the same depth owner.
That can avoid the tentative-KV delivery required when parent and child branches occupy separate replicas.
It does not save that delivery against TP, which already keeps each head's KV locally.
Do not count nonexistent remote KV reads in the TP denominator.

Unlike ordinary disjoint-layer PP, this mapping replicates the common core weights across owners.
Its cluster core payload is approximately `P*W_core`, not `W_core`.
Its balanced depth-KV partition does not automatically use less cluster KV than balanced head-sharded TP.
Transporting a normalized BF16 H=2048 boundary costs 4,096 payload bytes per row.
A fused hidden/residual pair can require 8,192 bytes instead.
Neither payload count measures transfer latency.

The transfer boundary overlaps existing work as follows:

| Mechanism | Direct prior | What remains specific to the hypothesis |
|---|---|---|
| Concurrent shallow proposals and later verification | LoopSpec, WaveFront, PipeSpec, SPD | Rank placement of ready Ouro depth states. |
| Stationary stage KV with activation transport | PipeInfer, SPD | Repeated common-core replicas instead of disjoint layer weights. |
| Private speculative histories and accepted-prefix publication | LoopSpec, PipeInfer, SPD | Depth-indexed remote lifetime and epoch protocol. |
| Rollback and cancellation | All three audited neighbors | Exact original-gate and `last_exited` behavior across depth owners. |

LoopSpec already batches mixed-depth lanes and shares same-depth ancestor mappings.
WaveFront already addresses per-row depths and groups compatible same-slot queries.
The missing TP path in those checkouts does not prove a new parallel axis.
It defines a required implementation control, not a measured baseline in this audit. [Existing pinned audit](literature_nearest_audit.md), [ownership contract](branch_ownership_cost_contract.md)

For fixed R4, depth KV can remain stationary without remote historical reads.
For original adaptive execution, early exit changes that obligation.
Deeper owners need the exact terminal-copy entries for each halted historical token before dependent attention.
One complete BF16 terminal-KV bundle contains 196,608 bytes for Base1.4B or 393,216 bytes for Base2.6B.
Charge each receiving owner only for its missing exact version.
Local depth aliases can avoid duplicate payloads if their provenance and lifetime remain valid.
An alias or remote-read design needs its own valid lifetime and transport contract.
Full-R4 verification cannot certify a different original-gate adaptive target.

## Charged necessary inequality

Use the same P, target, proposal policy, accepted output count N, and gate/KV contract.
Use the same branch DAG, including rejected work.
Allow TP to batch the existing mixed-depth frontier and use generic shared-prefix attention.
Allow each depth owner to batch all compatible local ready rows.

For a synchronized-frontier comparison, let `Q_s` be the ready rows in round s.
Let `Q_sj` be the rows assigned to depth owner j.
Define these compute-only critical terms:

```text
M_D  = sum_s max_j T_full_core(Q_sj, local_depth_KV)
M_TP = sum_s T_TP_core_without_exposed_collectives(Q_s, head_sharded_KV)
```

Define A as the TP collective time exposed on the critical timeline.
Define X as exposed depth-boundary transport and required state publication.
Define G as exposed draft, verification, gate, control, and cancellation costs.
Define S as fill/drain, dependency waits, and rollback/refill stalls not already counted.
Use subscripts D and TP for placement-specific terms.
Include discarded compute in M, not again in G or S.
Only disjoint critical-time terms may enter these sums.
Do not sum overlapping worker timers, message service times, or per-GPU maxima twice.

```text
T_D  = M_D  + X_D + G_D  + S_D
T_TP = M_TP + A   + G_TP + S_TP

Necessary for a strict same-N decode win:
A > (M_D-M_TP) + X_D + (G_D-G_TP) + (S_D-S_TP)
```

This inequality does not assume that the compute difference is positive.
That difference includes smaller batches, duplicated weight reads, different kernel shapes, and actual owner parallelism.
Avoided reductions must cover the remaining charged difference.
No source supplies those terms for this Ouro/H100 operating point.
For E2E timing, also include the difference in prefill and state-installation time.

A CPU rational check confirms lose, tie, and win examples for this rearrangement.
The inputs are arbitrary time units, not campaign estimates.
For fully asynchronous schedules, replace round sums with their actual event-DAG makespans.
The synchronized expression cannot reject an optimum that changes legal ready-frontier timing.
TP must receive the same scheduling opportunity.

## Decision and time

These neighbors remove novelty claims based only on pipelining, stationary KV, or rollback messages.
They do not prove that every depth placement loses.
A remaining claim needs a distinct ownership transformation and a charged same-P advantage.
This audit adopts no candidate, trained component, new engine, or source performance result.
Original-gate distributed speculative correctness remains `UNVALIDATED`.
Depth-owned performance remains `UNTESTED`.

Worker start: 2026-10-19 14:20:56 UTC.
Primary retrieval cutoff: 2026-10-19 14:30:41 UTC.
CPU algebra checkpoint: 2026-10-19 14:26:27 UTC.
Final verification cutoff: 2026-10-19 14:30:41 UTC.
The worker segment spans 585 seconds through that cutoff.
Concurrent worker time is not additional campaign elapsed time.
