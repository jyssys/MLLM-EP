# C14 reopening: physical-layer cyclic prefill

## Changed assumption and decision

The changed assumption concerns physical weight storage, not causal dependencies.
The installed default pipeline IR duplicates active tied parameters across logical stages.
An aliased graph does not establish that the stock runtime preserves those aliases.
The [CPU storage audit](pipeline_alias_storage_audit_v1.md) establishes this difference for its toy only.

C14 remains `GENERIC_ONLY` for the mechanism examined here.
This decision does not reject every cyclic placement advantage over native TP4.
It rejects a distinct architecture claim based only on physical layer owners, retained aliases, causal chunks, and depth-indexed KV.
A tied-aware generic cyclic pipeline can use all four features with the same physical execution trace.
No GPU execution or runner change occurs in this audit.

## Primary implementation checks

### Generic tied ownership: Fast-LLM

The inspected official code is [Fast-LLM](https://github.com/ServiceNow/Fast-LLM/tree/7d61a1ba2047814955f8acc1cde7bc10a68fa198), commit `7d61a1ba2047814955f8acc1cde7bc10a68fa198`.
Its commit time is 2026-07-20 22:11:16 UTC.

The [stage builder](https://github.com/ServiceNow/Fast-LLM/blob/7d61a1ba2047814955f8acc1cde7bc10a68fa198/fast_llm/engine/multi_stage/multi_stage.py#L71) maps stage indices cyclically across ranks.
Lines 328–346 supply tied logical uses with the main stage's parameter buffer.
The [parameter binding](https://github.com/ServiceNow/Fast-LLM/blob/7d61a1ba2047814955f8acc1cde7bc10a68fa198/fast_llm/engine/multi_stage/stage_base.py#L142) assigns that buffer to each consuming module.
This is actual buffer binding, not only a shared-name annotation.
The tied-group builder at `multi_stage.py:618–633` creates no cross-rank tie group when all uses belong to one rank.

The [runner](https://github.com/ServiceNow/Fast-LLM/blob/7d61a1ba2047814955f8acc1cde7bc10a68fa198/fast_llm/engine/schedule/runner.py#L207) executes the supplied stage schedule.
Its nontraining path returns before tied-gradient reduction at lines 230–243.
These inspected functions supply generic placement and alias primitives.
They do not supply a tested Ouro prefill server, depth-KV adapter, or latency result.
FSDP buffer restoration, configuration limits, and schedule integration still need implementation-specific checks.

### Tied names do not imply one physical copy

The inspected [DeepSpeed implementation](https://github.com/microsoft/DeepSpeed/blob/0eb3a8bcc6d9a8775478e5377482c4bf2dd23c1c/deepspeed/runtime/pipe/module.py#L257) is commit `0eb3a8bcc6d9a8775478e5377482c4bf2dd23c1c`.
Its commit time is 2026-10-19 13:31:24 UTC.
`_build()` reuses a keyed module for repeated local uses at lines 257–269.
Cross-stage owners instead receive local copies, an initialization broadcast, and training gradient synchronization at lines 457–527.
The [official tutorial](https://www.deepspeed.ai/tutorials/pipeline/#tied-layers) explicitly describes replication across owning stages.
This is not a free remote pointer or an Ouro inference implementation.

The existing [PiPPy/PyTorch audit](pipeline_alias_storage_audit_v1.md) already pins both transmission and replication policies.
Its installed public conversion takes the replication path.
Correcting this generic default cannot itself establish architectural novelty.

### Causal chunks: TeraPipe

[TeraPipe](https://arxiv.org/html/2102.07988v2) remains arXiv:2102.07988v2, dated 2021-09-28.
Sections 3.2–3.4 expose token-chunk causal pipelining and its combination with operation partitioning.
Section 3.3 charges increasing prefix cost when selecting chunk sizes.
Its measured scope is synchronous training, not recurrent inference.

The official repository HEAD is `1a17aa26798331f5986bff4974a6c9447ba866ea`, dated 2021-05-04.
In [the forward code](https://github.com/zhuohan123/terapipe/blob/1a17aa26798331f5986bff4974a6c9447ba866ea/terapipe.py#L238), each batch slice retains cache state across sequence chunks.
Lines 251–264 update the cache length after each chunk.
[Attention code](https://github.com/zhuohan123/terapipe/blob/1a17aa26798331f5986bff4974a6c9447ba866ea/transformer_models.py#L214) appends current KV and masks future positions.
This code is not a tied recurrent implementation.
It directly supports the generic causal-chunk mechanism that C14 must beat.

The previous [breadth-first audit](parallel_track_breadth_first_transfer_v1.md) already checks generic cyclic stage placement.
That training schedule does not provide four independent inputs from four dependent recurrent passes.
No second broad search of that paper occurs here.

### Inference KV placement: MOCAP

[MOCAP](https://arxiv.org/html/2606.22968v1) has one listed arXiv version, v1, dated 2026-06-22.
Sections 2.1 and 4 describe prefill-only inference, cross-stage KV spill, and nonuniform chunk partitioning.
Its fixed cross-half pairing avoids repeated rematching.
Remote KV service can block the owner and extend the critical path.
Section 5.1 evaluates an ASTRA-sim-based event simulator with modeled wafer-scale hardware.
It does not measure Ouro on four H100 GPUs.
No verified public code link appeared in the inspected paper or bounded code searches.
The code status is `NOT_LOCATED`, not proof of absence.

This prior strengthens the generic KV-placement comparison.
Its prefill-only cache lifetime does not match a prompt whose KV must remain available during later decoding.
No remote-read saving transfers to native TP4, whose local head shards already retain stationary KV.

## Exact physical-layer-stationary construction

Use P=4 owners and R=4 visits.
Owner j retains one contiguous physical layer partition across all visits.
Base 1.4B assigns six layers per owner.
Base 2.6B assigns twelve layers per owner.
This differs from C07's hypothetical full-core replica per depth owner.

Each cache address retains `(request, depth, physical_layer, position, version)`.
An owner keeps all depth banks for its own layers.
Equal weights do not make those banks equal.
No cross-depth KV alias follows from recurrent weight sharing.

The prefill contract permits causal chunks because all prompt token IDs already exist.
It does not permit unavailable hidden states or unavailable prefix KV.
For each `(request, depth, layer, chunk)`, attention waits for the required same-depth earlier-position KV writes.
The recurrent successor waits for its actual previous-pass output and the loop boundary norm.
The generic control must retain these exact edges and state versions.

Known prompt chunks can supply parallel work even at system C=1.
For example, stage 1 can execute chunk 0 while stage 0 executes chunk 1 in the first pass.
Thus the earlier `min(P,C)` current-token decode bound must not reject chunked prefill.
That bound still applies to its stated non-speculative one-current-token decode control.

Full-R4 prefill also applies to the original adaptive reference.
This prefill schedule does not reveal future decode exit decisions.
Later adaptive execution still needs the original gate and `last_exited` publication semantics.
The [frontier contract](exact_frontier_contract.md) defines those requirements.

## Strongest generic comparison: physical trace identity

Give the generic cyclic baseline the following identical inputs:

1. The physical layer owners and immutable weight buffers.
2. The chunk DAG and every loop norm dependency.
3. The depth-indexed KV addresses, capacities, and publication rules.
4. The kernel implementations and boundary tensor representation.
5. The stream, event, communication, and legal scheduling choices.

Map each proposed operation to the same generic operation with a logical depth label.
Map each allocation to the same owner-local allocation.
Map each send, receive, and synchronization event identically.
Choose the same schedule.
The resulting device work, messages, storage lifetimes, and resource contention are identical under this construction.

This comparison is stronger than DAG representability alone.
It explicitly matches physical ownership and the state protocol.
It is a constructed matched baseline, not evidence that stock Fast-LLM or PyTorch already serves Ouro efficiently.
Missing Ouro integration can justify engineering work.
It does not establish a separate architecture when that integration adds no different physical mechanism.

## Capacity and communication versus native TP4

The following quantities are structural accounting, not measured allocations.
Dense core W excludes embeddings, head, norms, gate, workspace, and allocator overhead.
Both layouts retain one aggregate W.
Balanced physical PP divides layers.
Native TP4 divides matrix dimensions and attention heads.

| Base model | W per owner | Full-R4 KV bytes per prompt token per owner | KV at C8,N4096 per owner |
|---|---:|---:|---:|
| 1.4B | 588 MiB | 196608 | 6144 MiB |
| 2.6B | 1176 MiB | 393216 | 12288 MiB |

These core and KV terms are identical for balanced physical PP4 and native TP4.
There is no dense-core or final retained-KV admission gain from this placement alone.
The controlled native TP4 jobs already complete these maximum protocol cells with the stated 52 GiB KV reservation per rank.
This observation does not settle transient scratch, non-core tensors, or other workloads.

Each chunk crosses `RP−1=15` core boundaries before final output processing.
At H=2048, one BF16 hidden row contains 4096 bytes.
Fifteen such payloads give 60 KiB per prompt token as a simple lower-payload accounting case.
This is not a measured native message representation.

The installed `vllm/model_executor/models/ouro.py:260–281` returns both hidden and residual tensors between physical layers.
Lines 367–374 combine them at the loop norm.
Preserving that representation can require two BF16 rows at internal cuts.
Materializing their sum can change rounding and must not silently replace the native numerical contract.
Embedding/head routing, metadata, message startup, staging, synchronization, and link contention also count.

Native TP4 pays its actual reductions instead of these pipeline boundaries.
It does not transfer all historical KV to each query.
Neither collective count nor hidden-message count establishes a latency advantage.
Full local matrices also lose TP's simultaneous use of four devices for one projection.
Chunking can reduce GEMM efficiency and change attention cost.
Reentrant visits contend with other ready chunks for the same owner.

## Charged latency condition and cheapest falsifiers

For a fixed resource-constrained schedule, define `T_C0` and `T_T0` as zero-network-cost makespans for cyclic PP and native TP4.
Retain actual compute costs, chunk shapes, kernel ordering, KV dependencies, local control, fill/drain, and buffer restrictions.
Let `D_C=T_C−T_C0` and `D_T=T_T−T_T0` denote each layout's added network cost under that model.
Do not obtain either value by summing overlapping communication durations.

The cyclic layout wins only if:

```text
D_T > (T_C0 - T_T0) + D_C
```

This is an accounting identity, not a performance prediction.
The network terms include exposed message latency and charged contention effects.
This audit measures neither value.
Kernel slowdowns caused by communication require additional modeled or measured costs.
A zero-cost network counterfactual must not silently remove unrelated compute work.

The capacity falsifier is already cheap: compare both balanced W and retained KV, then inspect actual admission.
The current protocol shows no such admission boundary.
A different transient-memory claim must name the exact extra buffer and its lifetime in both layouts.
The generic baseline receives the same buffer policy.

The cheapest useful future latency falsifier is a component lower-bound screen, not a complete new engine.
Measure the required full-matrix partition kernels at real chunk and prefix shapes with correct depth KV.
Use the slowest-owner service bound and the longest dependency-path bound separately.
Do not add those overlapping bounds.
A component measurement is not a universal lower bound for every kernel implementation.
The screen must state its kernel, grouping, fusion, and scheduling restrictions.
If their maximum already exceeds matched TP4 prefill time, positive communication costs cannot rescue that modeled layout.
Otherwise, add actual boundary payload/startup measurements and evaluate the complete resource-constrained schedule.
Passing the lower bound would not establish a speedup.
Current cohort E2E controls include 128 decode tokens and do not isolate this prefill time.

## Scope and evidence preserved

No new ownership distinction survives the granted matched generic baseline.
A generic cyclic pipeline could still outperform native TP4 in a different measured regime.
That possibility does not reopen C14 as a new architecture under the present mechanism.
The audit adds no candidate card and changes no coordinator status.
Earlier decode-frontier, alias, and causal-DAG evidence remains intact within its stated scope.

The [source ledger](../artifacts/c14_cyclic_prefill_sources_v1.jsonl) preserves pins, hashes, short excerpts, and inspected locators.
The [search ledger](../artifacts/c14_cyclic_prefill_searches_v1.jsonl) records new queries and bounded negative lookups.
The [screen artifact](../artifacts/c14_cyclic_prefill_screen_v1.json) separates derived quantities from untested costs.

Worker start: 2026-10-19 14:48:41 UTC.
Verification cutoff: 2026-10-19 14:57:39 UTC.
This worker segment spans 538 seconds through that cutoff.
Metadata publication follows the cutoff.
Concurrent worker intervals do not add campaign elapsed time.
