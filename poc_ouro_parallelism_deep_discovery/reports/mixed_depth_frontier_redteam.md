# C16: Mixed-depth rows across GPU owners

Worker: `/root/independent_axes`.
Segment began: 2026-10-19 11:07:23 UTC.
Evidence: primary-source inspection, symbolic accounting, and deterministic CPU controls.
This report contains no GPU measurement or model-quality result.

## Outcome

Mixed-depth row batching is legal when each row retains its logical-depth KV identity.
The inspected CDB implementation restricts full-depth KV batches to one depth slot.
Removing that restriction could improve this implementation.
It does not establish a new execution axis.

WaveFront already implements a union of different requests, positions, and recurrent depths with separate depth-indexed KV.
Its implementation copies selected historical KV into packed attention inputs.
A generic per-row page table can instead address separate banks in one arena.
That change concerns cache layout and kernel integration.

Cross-owner pooling can increase dense batch size relative to independent small replicas.
The same ready rows remain available to generic TP and generic operator disaggregation.
No ownership-dependent advantage survives this source and dependency screen.
The verdict is `GENERIC_ONLY`, not a claim that every possible hybrid must fail.
The complete card is [C16](../candidates/C16_mixed_depth_frontier.md).

## 1. Candidate and changed assumption

The hypothesis pools ready rows from different GPU owners into one shared physical-core execution.
The rows can occupy different recurrent depths because the physical weights repeat.
Each row keeps its request, position, physical layer, recurrent depth, and KV version.
The hypothesis does not use predicted hidden states or future gate decisions.

This differs from a same-row remote MLP test.
The proposed saving comes from a larger pooled batch and fewer separate weight streams.
It does not come from faster execution of the same single-row remote call.
This block performs no new GPU test and does not repeat the prior remote test.

It also differs from assuming that the local CDB restriction is fundamental.
The first control changes only KV addressing and permits an ordinary mixed-depth union.
An ownership-specific benefit must survive that control before an engine merits implementation.

## 2. Actual implementation overlap

### CDB

The local CDB source is revision `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The working files also receive SHA256 identities below.
`model_runner.py:756` selects one integer `kv_slot` for the whole recurrent batch.
It raises an error when another row needs a different slot.
`scheduler.py:322` therefore defers rows with other slots.
The mixed-depth path in `continuous_api.py:1075` explicitly describes shared-KV execution.

The Ouro adapter uses the same physical layers for all rows.
It applies loop norm before the original gate.
The original hazard policy updates token-local survival and CDF after the current signal exists.
This screen retains that timing.
It does not use the trained lookahead gate or recorded future exit depths.
The [author repository](https://github.com/kschwethelm/continuous-depth-batching) contains these interfaces.

The CDB paper supplies the direct scheduling prior.
It does not establish a production multi-GPU implementation for this candidate.
Its shared-KV choices cannot silently replace the required full-depth identities.
See the [paper](https://arxiv.org/html/2608.09444v2), sections on continuous depth batching and implementation limitations.

### WaveFront

The author revision is `434520932897fcc1e26898891c7e2feaa6ab099f`.
`OuroBlocksBatch.advance_union()` stacks all selected hidden rows before shared dense operations.
It derives a separate depth/layer slot for every row.
With `kv_budget_s=None`, it allocates all recurrent-depth slots.
Thus, mixed-depth batching does not require shared KV.
See [the actual union implementation](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/ouro_blocks_batch.py#L154).

Its cache writes use `(slot, sequence, position)` triples.
`gather_pack()` copies the selected historical prefixes with `index_select` before varlen attention.
See [the cache implementation](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/batch_kv_cache.py#L54).
The inspected fixed-batch scheduler includes an emulated-acceptance experiment path.
That path is not evidence for unchanged original adaptive gates.
See [its declared protocol](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/scheduler_batch.py#L1).

The paper evaluates single-GPU self-speculation and discusses adaptive integration as future work.
It does not report this multi-GPU, unchanged-gate comparison.
That missing evaluation does not remove its direct mixed-depth union overlap.
See the [WaveFront paper](https://arxiv.org/html/2609.23033), implementation and discussion sections.

### NanoFlow and generic paged attention

NanoFlow's author revision is `f179a907828b87e042126e585880301ff5b2c62a`.
Its split function copies operators for different row subsets and adds explicit dependency links.
Its executor constructs a topological DAG and uses CUDA events or graph replay.
This directly overlaps operation-level row partitioning and compute/network overlap.
See [the split function](https://github.com/efeslab/Nanoflow/blob/f179a907828b87e042126e585880301ff5b2c62a/core/nanobatchSplit.py)
and [the executor](https://github.com/efeslab/Nanoflow/blob/f179a907828b87e042126e585880301ff5b2c62a/core/executor.py).

The NanoFlow paper studies throughput with resource-aware operation overlap.
Its large-batch results do not establish a gain at Ouro system C=1/2/4/8.
That regime difference requires a new performance test, not a new scheduling claim.
See the [NanoFlow paper](https://arxiv.org/html/2408.12757), sections 4 and 6.

FlashInfer's batch decode API accepts per-row page indices, index pointers, and final-page lengths.
Separate depth banks can therefore remain separate addresses inside one compatible allocation.
This is an interface-level construction, not a verified Ouro integration.
See the [official paged-decode API](https://docs.flashinfer.ai/api/attention.html#flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper).
The same page records graph-shape restrictions and planning costs.

## 3. Explicit useful ready frontiers

Here, r=1 means the first recurrent pass.
All rows below wait at the entrance to the same physical layer.
Their earlier passes completed, and their required historical KV versions exist.
An online arrival pattern can produce these snapshots without future-token speculation.
The examples do not establish how often these snapshots occur.

| System C | Constructed next-depth vector | Depth-bucket sizes | Union size | New width beyond generic union |
|---:|---|---|---:|---:|
| 1 | [1] | [1] | 1 | 0 |
| 2 | [1,2] | [1,1] | 2 | 0 |
| 4 | [1,2,3,4] | [1,1,1,1] | 4 | 0 |
| 8 | [1,2,3,4,1,2,3,4] | [2,2,2,2] | 8 | 0 |

For fixed R4, staggered arrivals can produce these depths.
Synchronous arrivals with equal work instead remain at one common depth.
They already give the baseline one C-row batch.
Delaying those requests only to create depth diversity is not a valid benefit.

For adaptive execution, the CPU construction uses actual previous hazards of 1/16 and a threshold of 1/2.
Previous CDF values are 0, 0.0625, 0.12109375, and 0.176025390625.
They permit the shown continuation rows after the previous decisions complete.
The CPU control constructs these hazards rather than using model outputs or recorded exits.
The construction does not reveal the current or future hazard.

One non-speculative decode request supplies at most one active token row at a time.
Its future recurrent inputs depend on its current recurrent output.
Its next token depends on the current selected token.
Therefore, the ready frontier has at most C such rows, not R*C rows.
The comparison excludes prefill and speculative branches that require separate correctness checks.

An alias-aware logical unrolling groups rows by physical weight identity and compatible operator signature.
It obtains the same union.
It need not materialize four copies of the weights.
An R1 control also benefits from cross-request dense batching.
These controls defeat the proposed loop-specificity claim at this abstraction level.

## 4. KV ownership and readiness

The legal identity is `(request, token position, r, physical layer, version)`.
Shared weights do not make two such KV values interchangeable.
Original adaptive propagation must preserve the baseline's selected exit-state semantics.
The policy named `last_exited` copies or references the actual selected state in deeper slots.
It does not remove those slots or authorize an uncomputed source state.

One arena per physical layer can contain separate depth banks.
For R=4, C requests, and S positions, a scalar row address can use:

`offset(r,q,j)=((r-1)*C+q)*S+j`.

The physical layer selects its arena.
This injective address transformation changes neither KV values nor their readiness.
Paged blocks can use the same separation with additional allocation metadata.
The CPU control checks 512 distinct identities and 112 selected historical reads.
A deliberately wrong-depth read gives 10010 instead of 20010.

| Object | Ready condition | Owner and lifetime |
|---|---|---|
| Weight shard | Model load or completed transfer | Physical operator worker, immutable |
| Current hidden row | Actual preceding layer/pass completed | Request owner until transfer |
| Historical KV | Original write or permitted exit propagation completed | Request/depth/layer cache owner |
| Current KV | Actual QKV projection and RoPE completed | Matching cache owner |
| Gate state | Original signal consumed after loop norm | Token-local survival/CDF owner |
| Packed rows and metadata | All selected inputs ready | Pool scratch until matching outputs return |

Moving a row to a worker does not move its historical KV for free.
A full-core pool needs local KV, remote attention, or explicit history migration.
Each choice creates different messages and memory pressure.
The cheaper MLP-only pool keeps attention and KV at the original owner.
It is generic attention/FFN disaggregation rather than a new recurrent dependency.

## 5. Cross-owner cost screen

For one remote MLP row, input and output each contain D BF16 values.
At D=2048, the logical round-trip payload is `2*D*2=8192 bytes` per physical layer.
This count excludes metadata, messages, packing, synchronization, queue delay, and scratch.
It does not model topology or link latency.

The table balances request owners by `request_id modulo P` and pools at owner 0.
It describes one possible ownership choice, not a lower bound for every pool.

| P | C=1 remote bytes | C=2 remote bytes | C=4 remote bytes | C=8 remote bytes |
|---:|---:|---:|---:|---:|
| 2 | 0 | 8192 | 16384 | 32768 |
| 4 | 0 | 8192 | 24576 | 49152 |
| 8 | 0 | 8192 | 24576 | 57344 |

These are per-layer MLP payloads, not measured network traffic.
Splitting the pool across workers changes both payload and dense batch sizes.
Forcing one worker to handle C rows can idle otherwise useful GPUs.
The same-resource baseline must include independent replicas and generic TP, not only depth-bucketed DP.
Independent replicas must not receive an artificial cross-request barrier.

Let F be the legally ready rows at the same operator.
Let `T_union(F,P)` describe the best matched generic execution of that union.
The candidate must beat that cost after every owner transfer and waiting interval.
Reducing several depth buckets to one union only compares with a weaker baseline.
Writing `sum(bucket times)` as `max(bucket times)` also requires real resource overlap.

The intended exposed saving is dense weight-load amortization or reduced TP communication.
The added path is owner send, pool assembly, queued compute, output send, and owner continuation.
The original loop norm and gate still follow the completed recurrent output.
No numerical latency estimate follows from this symbolic screen.
P8 remains a projection only.

## 6. CPU controls and exactness limits

Run from the campaign root:

`python scripts/mixed_depth_frontier_cpu.py`

The saved [artifact](../artifacts/mixed_depth_frontier_cpu.json) records eight legal frontier cases.
It also records four rejected controls: duplicate active request, missing recurrent input, uncomputed gate, and continuation after actual exit.
The address test preserves all 512 identities.
An eight-row integer dense control matches separate-row and pooled-row results.
Generic alias-aware batching produces the same union in every constructed frontier.

These CPU controls use constructed algebra and addresses.
They use no model weights, fitted predictor, GPU, or learned component.
The DAG excerpt checks one attention entrance, not a complete resource-constrained model schedule.
Its input and historical-state readiness are explicit snapshot preconditions.
The payload table is structural accounting, not measured traffic.

The intended exactness class preserves the model graph and state identities.
It does not promise identical BF16 results across different batch shapes or kernels.
Different arithmetic can change a gate close to its threshold.
Any later implementation needs matched backend controls, gate/exit/KV comparisons, and free-running quality checks.
All model numerics, quality, latency, and incremental speedup remain `NOT_RUN`.

## 7. Smallest discriminator and stop criterion

This screen already rejects mixed-depth union as an independent new axis.
It does not recommend a full engine.
The next informative action is a paired ready-frontier trace comparison with three executors:

1. Full-KV CDB with homogeneous depth buckets.
2. Generic full-KV union with per-row addressing and the same physical weight aliases.
3. Cross-owner pooling with every transfer and wait interval charged.

Use the same arrivals, completed-state events, and original online gate availability.
Do not use recorded future exits to choose the next dispatch.
Do not add batch-wait barriers only to the baseline.
Test C=1/2/4/8 separately for fixed R4 and original adaptive execution.

Stop this candidate if its benefit disappears against executor 2.
Such a result supports a CDB implementation improvement, not a qualifying new axis.
A surviving ownership benefit needs same-P TP, DP, disaggregation, and NanoFlow-style overlap controls.
Only then would operator calibration or an engine estimate become informative.

## 8. Search and source ledger

The following searches occurred in this segment:

- `NanoFlow serving official github nanobatch pipeline tensor parallel`
- `WaveFront Decoding Parallelized Self Speculative Decoding Looped Language Models github 2609.23033`
- `NanoFlow official github NanoFlow serving Berkeley`
- `WaveFront Decoding Hyeongju Ha Jae Joon Kim github`

One result concerned an unrelated 2020 normalizing-flow project.
This report does not use that result.
The inspected author repositories and revisions appear in section 2.
Paper reading preceded code inspection, followed by targeted dependency and addressing checks.
No search establishes that no other prior exists.

Local CDB SHA256 identities:

| File | SHA256 |
|---|---|
| model_runner.py | 7a5e62c355f516233b432d1184da86e41b9880ad422c7aef94c0dab2e4572386 |
| scheduler.py | 3a3f32144990bcf9ed906f1f94387469b7c230644c59caa1906f599f6eddcc2d |
| exit_policy.py | 46340237b2ba399fb38b814e0c7f6d0b65e915c554d404b148520f238860d948 |

The CPU script SHA256 is `97d483059a21e6cca961962199ceb262f1bea764625bd7865ad842c09329e71e`.
The first CPU run occurred at 11:14:24 UTC.
The final script separates the generic logical-unrolling control from direct physical grouping.
Its actual execution ran from 11:18:03.738422 to 11:18:03.739305 UTC on 2026-10-19.
These times describe this tiny CPU script, not inference performance.
The writing skill and its required reference informed new prose only.
Earlier reports and raw evidence remain unchanged.

## 9. Time ledger

The segment began at 11:07:23 UTC on 2026-10-19.
The first source inspection block reached the 11:12:35 UTC checkpoint.
CPU construction and execution reached the 11:14:24 UTC checkpoint.
Targeted source checks and report synthesis followed.
The final evidence checkpoint was 11:18:04 UTC.
The active worker interval to that checkpoint was 641 seconds.
This interval includes source inspection, derivation, CPU controls, and report synthesis.
Final publication and prose checks followed that checkpoint.
Count this worker interval once within the campaign's concurrent wall time.
Do not add it again to another worker's overlapping interval.
