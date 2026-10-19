# KV indirection red-team audit

## Decision

Mixed-depth KV addressing is legal when each row retains its required request, token, depth, and physical-layer identity.
It does not require identical KV across depths.
The existing FA3 interface already accepts a separate page table for each batch row.
The current CDB wrapper, rather than the attention equation, imposes one recurrent slot per launch.

No-copy `last_exited` is a separate change.
It can alias only the deeper logical rows that the selected policy explicitly fills with terminal-depth KV.
It cannot generally alias the distinct computed states of fixed R4.
Thus copy removal alone is `MODE_SPECIFIC` to adaptive execution.

The proposed address protocol supplies no new distributed parallel axis by itself.
It is a conditional recurrence-aware cache representation built from generic paging and immutable aliases.
WaveFront already uses per-row recurrent-depth addressing.
A stronger architecture claim requires a separate distributed ownership mechanism and a same-P benefit.
This audit measures no such benefit.

## Inspected interfaces

CDB remains pinned to `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
Its model runner passes one scalar `kv_slot` into a recurrent stage.
The graph key also contains that scalar at `model_runner.py:901–907`.
`_ensure_recurrent_cache_policy_kwargs:756–771` rejects rows from different slots.
`OuroCDBAdapter.recurrent_step:248–264` otherwise applies the same physical weights to every row.

The installed Transformers wrapper selects one cache tensor through `module.layer_idx`.
It then passes a per-row page table to FA3.
See `transformers/integrations/flash_paged.py::_paged_decode_forward:105–134`.
The exact inspected file has SHA-256 `6a83037a637cff837addad17e0568e33246b7c09fcc73b777700d9bb9e24dcfa`.
Its absolute path appears in the address contract.

The inspected FA3 Python API documents arbitrary page sizes, including 1.
It also documents in-place K/V append at the supplied cache length.
See `flash_attn_interface.py::flash_attn_with_kvcache:971–979,1013–1034`.
The file has SHA-256 `61d7c0bca45eb39e8ed1bdc1479b8dca380b7230454a57546bd8fbc9f5dfa233`.
This is interface evidence. It is not a completed kernel compatibility test.

The default CDB page size is 16.
Its config permits other positive sizes with FA3 at `continuous_depth_batching/config.py:55–56,146–153`.
FA2 requires a multiple of 256 in the same validation path.
This audit makes no FA2 compatibility claim for token-sized pages.

## Change A: mixed depth without KV aliasing

For physical layer l, allocate one key arena and one value arena containing all recurrent slots.
Keep each logical depth's storage distinct.
Let M denote the number of physical blocks reserved for one slot in that arena.

```text
arena_block(request, depth, token_block) = depth * M + request_block(token_block)
page_table[batch_row, token_block] = arena_block(request_for_row, depth_for_row, token_block)
```

The physical layer selects the arena. Each batch row selects its depth through block IDs.
The same weight projection can process rows at different depths because their physical-layer weights are identical.
Each row still consumes its own predecessor hidden state.
Every new K/V append must target a private writable location.

This layout can retain page size 16 and the existing FA3 page-table interface.
It changes allocator layout and wrapper metadata, not the attention equation.
It preserves fixed-R4 KV and adaptive copied KV without aliasing either.
It does not save KV capacity or eliminate terminal copies by itself.

The wrapper must build layer-compatible page tables for prefill and decode.
It must also replace the scalar slot in the graph key with stable tensor inputs.
Removing the slot check alone would direct some rows to the wrong cache tensor.

## Change B: no-copy terminal rows

Use zero-based recurrent slots s=0,1,2,3.
Let D(t) denote the number of executed passes for token t.
Let e(t)=D(t)-1 denote its terminal slot.

For a completed adaptive token, the selected policy defines this logical address:

```text
source_depth(t, s) = min(s, e(t))
logical_KV(request, layer, token=t, slot=s) -> computed_KV(request, layer, token=t, depth=source_depth(t,s))
```

This formula preserves the shallower computed states.
It aliases only deeper uncomputed states to the terminal computed state.
Full-R4 prompt positions retain four distinct computed states.
The official fixed-depth function and this adaptive policy remain different references.

At default page size 16, one page table cannot generally express these token-specific aliases.
Two adjacent tokens can have different exit depths inside the same logical page.
One page-table entry supplies one physical page and fixed within-page offsets.
It cannot select unrelated source pages for two offsets without additional data movement.

Token-sized pages can express these aliases through the existing documented FA3 interface.
They require much wider page tables.
A custom row-indirect attention reader is another option, but it requires a separate kernel implementation.
Gathering historical KV into temporary contiguous buffers is a correctness oracle, not free indirection.
It adds history-sized reads, writes, scratch memory, and launch costs.

The present CDB allocator reserves a slab for every depth/layer pair.
Replacing copies with aliases does not release those slabs.
Capacity reduction additionally requires a compact allocator that reserves only executed token-depth rows.
These are two separate claims and two separate changes.

## Versioned address protocol

Each logical identity contains request ID, request generation, branch version, token position, recurrent slot, and physical layer.
Each physical row identity contains arena ID, row ID, and allocation epoch.
The allocation epoch prevents an old pointer from silently naming a recycled row.

Use one row-group ID across all physical layers on a KV owner.
Each layer stores its K/V pair at that row-group index in its own arena.
This choice allows one logical address table to serve every physical layer.
It does not alias values across layers.

1. Allocate a private row group before a token executes recurrent slot s.
2. Publish no alias for an uncomputed deeper slot of that active token.
3. Write each physical layer's K/V into the private row group as its attention projection executes.
4. Mark that row group complete only after the full recurrent pass finishes.
5. If the original gate continues, retain the completed group and allocate the next slot's private group.
6. If the original gate exits, publish deeper logical aliases to the completed terminal group.
7. Publish the token's immutable history snapshot before its successor can read that history.
8. Retain every referenced row until all readers of that snapshot finish.
9. Release a row only after its references and in-flight readers reach zero.
10. Increase its allocation epoch before reuse.

The gate, q, minimum depth, maximum depth, and BF16 payload remain unchanged.
The initial protocol retains CDB's one recurrent token per request in flight.
It does not require a global barrier across independent requests.

Each attention launch uses an immutable address snapshot for its complete lifetime.
A metadata event must precede the kernel that reads that snapshot.
A completion event must precede metadata-buffer or row reuse.
The current kernel need not check epochs if the owner enforces these rules before publication.
Epoch checks remain necessary in the CPU contract and allocator bookkeeping.

The owner permits no writes to shared historical rows.
Current-token write destinations are pairwise distinct within a launch.
The FA3 API warns about ambiguous writes when cache indices repeat at lines 1029–1032.
The protocol forbids that condition for append destinations.

A speculative branch receives a separate version and private append rows.
It may share committed read-only history.
A losing branch cannot publish its mappings into the authoritative request version.
Branch acceptance must publish the complete winning mapping atomically at the request level.
This audit checks no speculative verification algorithm.

## Why fixed-depth KV cannot generally alias

Fixed R4 computes four different recurrent inputs for each physical layer and position.
The shared projection weights do not make those inputs equal.
The keys and values therefore lack a source-backed equality guarantee across depths.
The official cache uses `r * L + l` and retains each computed depth.

If two fixed-depth slots shared one writable row, the later pass could overwrite an earlier pass's history.
The next token would then read the later value when it requires the earlier value.
Read-only aliasing would require actual equality of both K and V plus compatible lifetimes.
The model source establishes no such general invariant.
Accidental equality for one row is not a reusable architectural guarantee.

With D(t)=4, `source_depth(t,s)=s` for every slot.
The proposed exact protocol therefore allocates four distinct row groups.
It removes no terminal-copy work because fixed R4 performs no terminal copies.

## Fixed-R4 opportunity and novelty boundary

Change A can batch ready fixed-R4 tokens at different recurrent phases.
This requires at least two ready tasks with compatible physical-layer work.
Staggered arrivals or independent request progress can create that frontier without early exits.
A synchronized fixed-R4 cohort need not contain such a frontier.
One ordinary C=1 decode request does not expose independent next-token work through this change.

This is useful representation support, not evidence of fixed-depth headroom.
A tied-weight-aware CDB or grouped-batching baseline can use the same page tables.
Ordinary paged attention already provides per-row page addresses.
The recurrence-specific fact permits shared-weight GEMMs across depths, but does not establish a new distributed execution axis.

Stationary KV owners could use this protocol while shared compute workers process ready activations.
That extension must still transfer required projections, hidden states, or attention results.
A GPU-local page table cannot read another GPU's cache for free.
The extension needs explicit device ownership, messages, layout, and costs.
It must beat a baseline with the same addressing and batching support.

Verdict: `CONDITIONAL_MECHANISM` as a cache/batching prerequisite.
Copy removal is `MODE_SPECIFIC`.
The standalone paging implementation remains `GENERIC_ONLY` as a distributed architecture claim.

## Nearest existing per-row implementation

No local WaveFront checkout exists in this audit.
The source worker supplied already-fetched excerpts from its pinned official source inspection.
The pin is `434520932897fcc1e26898891c7e2feaa6ab099f`.
This is cross-worker source evidence, not a local execution check.

In `src/decoding/ouro_blocks.py:264–267`, each row writes `slot(steps[i]) * n_layers + layer_idx`.
Lines 294–297 gather that row's corresponding prefix separately.
Lines 128–132 replicate terminal KV into deeper slots.
Lines 381–397 apply the original gate and invoke that fill after an early exit.
Thus per-row depth addressing and terminal filling already exist in nearby work.
This evidence does not establish mixed-request paged FA3 performance or a no-copy alias protocol.

Primary source: [pinned WaveFront Ouro implementation](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/ouro_blocks.py).

## Cheapest CPU address test

Use distinct symbolic K/V markers for every request, layer, token, and executed depth.
Compare an explicit-copy reference against the proposed immutable address map.
This test needs no model, tensor library, or GPU.

The executed toy enumerates 729 exit histories.
It uses two requests, two physical layers, R4, one full-depth prompt position, and three decode positions per request.
Each decode position chooses D=2,3,4 independently.
The toy compares 93312 logical K/V values and checks 4374 deeper alias entries.
Every checked logical value matches the explicit-copy reference.

It also checks private append addresses, distinct fixed-R4 rows, and a page-size-16 counterexample.
Separate identity checks preserve the committed map during a losing-branch edit and distinguish a recycled row's new epoch.
These identity checks do not implement concurrent publication, reference counting, or a GPU allocator.
These CPU examples check bounded cases. They do not prove a complete runtime.

The command, source, raw output, timestamp, and tool chunk appear in `artifacts/kv_address_contract.json`.
FA3 execution, model numerics, free-running quality, GPU latency, and multi-GPU execution remain `NOT_RUN`.

## Tiny probe locations

| Location | Smallest required change | Critical check |
|---|---|---|
| CDB `continuous_batching/cache.py::__init__` | One arena per physical layer containing depth regions | Matching region offsets and padding |
| CDB `model_runner.py::_prepare_recurrent_stage_buffers` | Build each row's depth-aware page table | Matching request and depth identity |
| CDB `model_runner.py::_ensure_recurrent_cache_policy_kwargs` | Replace scalar slot with tensor-address contract | No silent slot mismatch |
| CDB `model_runner.py::_run_recurrent_stage` | Capture a stable metadata tensor interface | Graph replay observes fresh IDs |
| CDB `scheduler.py::_schedule_recurrent_batch` | Permit compatible mixed-depth rows | Every row is ready before launch |
| CDB `adapters/ouro.py::recurrent_step` | Pass per-row addressing without a scalar slot override | Preserve the loop norm and original gate |
| Transformers `flash_paged.py::_paged_decode_forward` | Select a physical-layer arena and its row tables | Existing FA3 append reads/writes required pages |
| CDB `continuous_api.py::_flush_exit_kv_copies` | Publish terminal aliases instead of copying | Publish only after the full terminal pass |

First test Change A with distinct depth regions and page size 16.
Then test the documented page-size-1 API using synthetic values and unique append destinations.
Only after those checks should a compact allocator or distributed extension receive implementation effort.
Do not change all three mechanisms in the first probe.

## Cost and rejection conditions

A shared row map uses R*N int32 IDs per request before versions and allocator metadata.
At R4 and N=8192, that map occupies 131072 bytes, or 128 KiB.
The per-launch selected table contains B*N IDs with token-sized pages.
Its width is 16 times the default page-size-16 table at the same maximum context.
The kernel still reads those tables for every physical layer.

The ideal compact payload contains `4*N_prompt + sum_t D(t)` row groups per request.
The full-depth reference contains `4*(N_prompt + N_decode)` row groups.
These are structural counts, not measured savings from the current preallocated cache.
Fixed R4 gives identical counts.

Reject a representation if it changes a committed logical K/V value or creates duplicate writable append addresses.
Reject a performance claim if page-table and allocator costs remove the measured copy/batching benefit.
Reject a fixed-depth claim if its only saved work is adaptive terminal copying.
Reject a new-axis claim if a matched generic paged-cache baseline provides the same benefit.

This audit changes no existing code or artifacts.
