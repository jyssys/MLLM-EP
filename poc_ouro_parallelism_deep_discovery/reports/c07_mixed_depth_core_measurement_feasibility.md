# C07 four-row common-core service screen

## Decision

The stock native worker has no one-pass API with a different original depth-cache tensor for each row.
Pinned CDB also rejects multiple physical KV slots in one recurrent launch.
A bounded standalone physical screen is possible in principle through explicit fixture preparation.
It requires actual selected-depth KV, a fixed one-pass body, and valid native attention metadata.
Those implementation steps remain untested.
This audit changes no runner and executes no GPU work.

The proposed comparison is TP4 with four co-batched rows versus four full-weight TP1 owners with one row each.
Every row advances through one complete physical core pass.
This is conditional operator service, not closed-loop speculation, task quality, or an end-to-end speedup.
The [metadata ledger](../artifacts/c07_mixed_depth_core_measurement_feasibility.json) retains source hashes, layouts, CPU address checks, and actual timestamps.

## What the inspected code supports

Native [OuroAttention.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:198) selects `self.attn[current_ut]` for the whole tensor.
Its `current_ut` argument is scalar.
Each Attention object owns one cache tensor.
The native [attention context lookup](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:731) selects metadata and cache by one layer name.
A per-row write-slot vector does not select a different cache allocation for each row.

The [native core forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:355) performs all configured recurrent passes.
It does not expose a pass-input hidden-state argument plus a per-row depth vector.
`inputs_embeds` can supply hidden values, but the normal core forward still executes its configured loop count.

CDB's [Ouro adapter](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/adapters/ouro.py:214) exposes one recurrent stage and a `recurrent_steps` vector.
That vector does not imply per-row cache selection.
The [runner](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/model_runner.py:756) raises an error when items select different KV slots.
The [scheduler](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/scheduler.py:322) defers items from other slots.
The [cache contract](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_batching/cache.py:195) requires slot-homogeneous batches for multi-slot policies.
Selecting a single-slot policy would change the required depth-KV semantics.
It is not an acceptable shortcut.

The CDB adapter also expects a decoder-level `rotary_emb` hook.
Native vLLM Ouro keeps rotary modules inside its physical attention blocks.
The adapter is therefore not a direct wrapper for the loaded native model.
See its [support check](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/adapters/ouro.py:139).

## Exact work unit and native body

Each fixture row needs a branch ID, state version, input position, logical depth, pass-input hidden state, and selected-depth prefix KV.
Its predecessor hidden state must already exist.
Every required prefix KV version must also exist.
The four rows must form a ready frontier rather than four dependent, uncomputed passes of one token.
If the fixture assumes readiness, label that assumption.
It does not measure natural speculative frontier availability.

Retain actual checkpoint weights and the complete model allocation in both arms.
TP4 uses four local query/KV heads per rank.
Each TP1 owner uses all 16 heads for its one row.
Both use head dimension 128, BF16, native RoPE, FA3, native packed projections, and native SwiGLU.
Batch and shard shapes differ by design.
Do not replace one arm with a different attention or MLP implementation.

The timed body must retain all L native layer bodies and the pass-boundary norm:

```python
h = fixture_hidden  # [4, 2048] for TP4, [1, 2048] for each TP1 owner
residual = None
for layer in core.layers[core.start_layer:core.end_layer]:
    h, residual = layer(positions, h, current_ut=0, residual=residual)
h, _ = core.norm(h, residual)
```

Here, physical slot 0 receives the remapped fixture pages described below.
It does not replace the logical per-row depth with depth zero.
The original [layer body](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:260) retains four normalization modules and its residual operations.
The [RMSNorm implementation](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/layernorm.py:74) can update the residual in place.
Keep an immutable hidden-state fixture and a private working copy.
Restore the working copy before every real replay.
The first-layer residual can alias that working input.
Do not assume that a previous replay leaves it unchanged.
Do not omit or algebraically replace these normalization and BF16 boundaries.

The [MLP](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:68) retains its native merged gate/up projection, activation, and down projection.
Native [row-parallel projections](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/linear.py:1676) retain their TP reductions.
At source level, one pass has two such reduction sites per layer.
That is 48 sites for Base1.4B or 96 for Base2.6B.
These source counts are not measured kernel counts after compiler fusion.
Embedding, gate, LM head, and sampler remain outside this core-service interval.
Their weights remain allocated.

## Preserving selected-depth KV through a bounded fixture remap

Native FA3 accepts different block-table rows within one cache allocation.
It does not accept four unrelated cache base pointers in one call.
Use that existing paged interface rather than inventing a per-row pointer kernel.

For each physical layer, copy each row's true selected-depth prefix into disjoint destination pages of one native cache tensor.
Retain an explicit mapping from `(branch, version, logical depth, prefix position)` to each destination page.
The native cache layout is `[num_blocks, local_heads, 16, 256]`.
K and V occupy the two 128-element halves of the last dimension.
Copy their BF16 words without conversion.
Keep every unselected original depth state and every full model allocation in the residency ledger.
This is a fixture copy, not free zero-copy native mixed-depth execution.

The four block-table rows select those disjoint destination pages.
The write-slot vector selects each row's fresh current-token position.
Only prefix positions before the current input position belong to the imported history.
The original QKV projection must compute the current K/V row.
The native cache update must complete before attention reads it.
The [native dependency](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:774) preserves that update-to-attention order.

For block size 16, the write slot is `block_table[i, position[i]//16] * 16 + position[i]%16`.
Preserve every row's true position and sequence length.
Do not substitute an R1 prefix for another depth's prefix.
Do not preload a future current-token KV result and omit its computation.

The existing slot-0 allocation can serve as the fixture arena only if sufficient disjoint pages exist.
The engine must be quiescent, with no pending reader or allocator reuse of those pages.
Otherwise, use a separate arena and count its full capacity.
The generic [bind_kv_cache helper](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/utils.py:482) is not a safe repeated post-load shortcut.
It asserts an empty runner cache list at line 505.
Page ownership and lifetime remain implementation requirements, not solved facts in this audit.

## Obtaining a valid ForwardContext and FA3 metadata

`execute_model` can provide a real context during an ordinary cached-decode preparation request.
It [builds slot mappings](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu_model_runner.py:4027), constructs attention metadata, and [enters set_forward_context](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu_model_runner.py:4376).
It then calls [GPUModelRunner._model_forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu_model_runner.py:3824).
That method is an explicit inspection or override boundary.
An opt-in standalone extension can inspect `get_forward_context()` there without changing an existing runner file.

Accept a metadata fixture only when it has these properties:

- Four actual rows for TP4, or one actual row for each TP1 owner.
- One query token per row and no speculative decode metadata.
- Ordered request identities and positions that match the fixture.
- `num_actual_tokens=4/1` and `max_query_len=1`.
- `query_start_loc=[0,1,2,3,4]` or `[0,1]`, before any declared padding.
- Sequence lengths equal each current position plus one.
- Valid block tables and write-slot mappings for the selected native cache.
- Causal full attention, the matching local head count, and native FA3 scheduling state.

Do not accept `attn_metadata=None` from a dummy profile call.
Do not assume that submitted C4 yields a four-row cached-decode invocation.
Fail with `NO_SUITABLE_METADATA` if the preparation request supplies no suitable invocation.

A saved ForwardContext reference is not a frozen fixture.
The native input batch reuses its device buffers.
Clone or rebuild every referenced metadata tensor before the engine can overwrite it.
Preserve dtype, shape, stride, padding, and lifetime.
Retain the original Attention module objects.
Do not deep-copy the model or `ForwardContext.no_compile_layers`.
The [FA3 metadata schema](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:241) includes scheduler metadata and split counts beyond block tables.
The [native builder](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:504) derives scheduling from query lengths, sequence lengths, heads, page size, and split policy.
Changing row count, head sharding, or lengths requires native metadata regeneration.
A pure page-ID remap leaves those schedule inputs unchanged, but still needs private block and slot tensors.

Actual pass-input hidden states are another fixture requirement.
A context snapshot alone does not provide them.
Use an explicit untimed native preparation path with matching predecessor passes and prefix versions.
A Python hook inside core.forward does not observe the internals of an already cached CUDA Graph replay.
Do not claim that a callback captured actual R2/R3/R4 inputs unless its execution and state versions support that claim.

## Temporary loop binding and graph safety

A standalone one-pass binding can execute inside the context that `execute_model` provides.
It must consume the prepared hidden states rather than recompute token embeddings.
However, `execute_model` also performs logits and other runner work.
Its complete wall time is not the isolated core interval.

Do not set the checkpoint's `total_ut_steps` override to one before model construction.
That change creates fewer Attention slots and erases required allocation.
Do not change `core.total_ut_steps` after R4 compilation and assume an existing graph now performs one pass.
Use a dedicated fixed one-pass body with fresh compilation and capture provenance.
Retain the original R4-created Attention objects and all four cache slots.
The simpler boundary uses untimed context preparation and a standalone helper under its own `set_forward_context`.
No serving scheduler needs to advance during the measured helper.

Native TP graph capture uses [distributed graph_capture](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/parallel_state.py:1430).
Its [group context](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/parallel_state.py:598) includes custom-All-Reduce capture handling.
A bare `torch.cuda.graph` call does not establish native TP collective registration.
FA3 split counts must also satisfy its captured-graph contract.
Use the same helper, kernel policy, and compile/capture policy in both placements.
If only eager execution works, label it an eager physical screen rather than the strongest compiled native TP control.

## Preparation cost and measured scope

Let `B = 2 * L * 16 * 128 * 2` bytes per token and pass across all physical layers.
The selected-prefix copy payload is `B * sum(prefix_length[i])`.
One device-to-device materialization reads and writes at least that payload once.
This is a byte-accounting model, not measured HBM transactions.
CPU staging adds its actual D2H and H2D traffic and time.

| Selected four-row prefixes | Base1.4B | Base2.6B |
|---|---:|---:|
| N1024 | 0.75 GiB | 1.5 GiB |
| N4096 | 3 GiB | 6 GiB |

Private append pages and 16-token rounding add capacity beyond these prefix payloads.
The metadata records those exact values.
Its target-page capacity counts private destination pages only.
A separate arena must also count its null or padding pages.
If the source histories and destination pages coexist, count both.
If destination pages occupy spare preallocated pool capacity, count their occupancy and retain the entire pool allocation.
Retain fixture preparation time separately from the warm service measurement.
No result may treat history replication, state extraction, or destination copies as a free serving operation.

For warm service, release all four devices after readiness and time one complete four-row core advance.
Input restoration must complete or precede the timed graph on the same ordered stream.
Use an explicit event dependency for restoration on another stream.
Retain the parent makespan and each actual replay interval separately.
Do not align independent GPU clocks by assumption.
Do not multiply one pass or the last replay to invent generation time.
Check normalized hidden outputs and newly written KV against native scalar references before timing conclusions.
Retain same-arm repeats to distinguish numerical variation from a placement effect.

This screen cannot establish branch creation, acceptance, correction, commit, prune, rollback, or future-state transfer cost.
It cannot establish that a real speculative scheduler supplies four ready rows continuously.
It cannot establish original-gate adaptive correctness or natural task quality.
It supplies only conditional physical service evidence for the declared fixture.

If fixture state, metadata lifetime, free-page ownership, or capture registration cannot remain bounded, stop at a lower-level screen.
A homogeneous-depth full-pass screen is then defensible, but it does not test mixed-depth cache selection.
Grouped native attention calls also preserve true depth caches, but their launch cost does not represent one co-batched FA3 call.
Neither fallback establishes the strongest C07 control or a new architecture.

## CPU evidence

The CPU checks passed 310 AST, logical-address, and capacity assertions with zero failures.
They include repeated depth labels without destination-page aliasing.
They do not check real CUDA addresses, numerical parity, or kernel execution.
The source segment began at 2026-10-19 14:33:23 UTC.
The metadata records its actual final timestamp and elapsed seconds.
