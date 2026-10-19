# C07 graph metadata: bounded source audit

## Decision before the V2 port

`INFRASTRUCTURE_BLOCKED` applies to graph execution in the current fixture.
Native APIs provide a bounded metadata route in principle.
They do not establish correctness for this fixture without additional evidence.
No implementation or GPU execution occurred during this audit.
This status does not establish a general limitation of Ouro or FA3.

The first blocker also affects the eager capture entry point.
The fixture assumes the V1 runner's `_model_forward` method and persistent `input_batch`.
The installed native configuration defaults dense generation models to V2.
See [runner selection](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/vllm.py:550) and [dense-model eligibility](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/vllm.py:605).
V2 [enters the real context](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu/model_runner.py:1339) and calls `self.model` directly.
It does not define `_model_forward` or retain that V1 `input_batch` field.
Its context supplies an actual `is_padding` tensor.
The current fixture does not pin V1 and rejects every non-null padding tensor.
The earlier long-context console also reports V2 for Ouro.

The parent requested a separate bounded V2 eager-hook port after this note.
That request does not erase this pre-port failure fact.

## Existing native metadata APIs

| Purpose | Native interface | Relevant requirement |
| --- | --- | --- |
| Private builder | [FlashAttentionMetadataBuilder constructor](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:355) | Actual KV spec, layer names, device, and resolved native configuration. |
| Common metadata | [CommonAttentionMetadata](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backend.py:412) | Own CPU/GPU query boundaries, sequence lengths, block tables, slot mappings, and positions. |
| Capture metadata | [build_for_cudagraph_capture](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backend.py:718) | Delegates to `build`. It does not independently enable graph-safe builder state. |
| V2 bridge | [build_attn_metadata](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu/attn_utils.py:566) | `for_cudagraph_capture=True` selects the builder capture method. |
| Remapped addresses | [update_block_table](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:717) | Shallow-copies metadata. It preserves the other referenced buffers and their ownership obligations. |

The current eager builder has no full-graph mode.
Its maximum capture size is zero after eager configuration resolution.
Calling its capture method therefore leaves the split bound at zero.
The [native code](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:489) explicitly marks that policy as not graph compatible.

A separate native builder needs a graph-enabled configuration and a capture size that admits its rows.
It must obtain the split bound from `attention_config.flash_attn_max_num_splits_for_cuda_graph`.
The installed [default](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/config/attention.py:42) is 32.
Do not assign a hand-selected split value to fixture metadata.
Do not mutate the serving builder or its configuration.
The private configuration and builder lifetime still need an explicit ownership check.

The [builder](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:400) allocates persistent INT32 scheduler storage.
Its size is `1 + round_up(max(max_num_seqs, max_capture_size), 4) * 4` elements.
With four source requests and capture sizes one or four, that is 17 elements, or 68 B per builder.
The builder copies the generated schedule there and clears its unused tail.
The returned metadata refers to that storage.
Another build on the same builder can overwrite it.
A quiescent fixture must own its builder or an independently retained graph-safe schedule.

FA3 also needs split-dependent intermediate allocations.
The source names a `[num_splits, num_heads, num_tokens, head_size]` intermediate shape.
The installed Python interface delegates allocation to the compiled FA3 operator.
This audit does not establish the complete native workspace byte count or graph-pool peak.
Those bytes need capture-time allocation evidence.

## Capture and reset obligations

Keep the existing page lease and all selected historical versions.
Build metadata for the actual one-row TP1 or four-row TP4 service shape.
Retain private query boundaries, sequence lengths, positions, padding, block tables, and current-token slot mappings.
No selected source page can become a destination.
Keep the original Attention objects and all 4L cache allocations.

The native [graph_capture context](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/parallel_state.py:1430) surrounds the CUDA capture.
It enters the TP/PP capture contexts and joins the caller stream before capture.
Custom all-reduce [registers graph buffer addresses](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/device_communicators/custom_all_reduce.py:197) when that context exits.
All ranks must use the same capture and registration order.
Do not replay before registration finishes.

The native [capture manager](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu/cudagraph_utils.py:295) warms allocation paths, then prepares fresh attention state for capture.
Inside its capture context, custom all-reduce [warmup returns `empty_like`](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/device_communicators/custom_all_reduce.py:264).
That warmup does not provide a numerical reference.
Fresh working hidden state and valid current-token state are therefore necessary before the first checked replay.
Restore the working input before every replay and charge that copy.
Do not restore hidden state after a dependent first-layer operation starts.
Historical prefix values must remain unchanged through every replay.

Use a private graph owner and stable input/output lifetimes.
Do not replace the original serving graph or call its manager to capture a different loop.
Do not assume a shared global graph pool has an independent lifetime.
Retain all original norms, residuals, casts, RoPE, FA3 operations, and TP reductions in source order.

## Numerical and performance limits

Graph-compatible FA3 scheduling can differ from eager scheduling.
Registered graph all-reduce also uses a different buffer path from eager all-reduce.
Source inspection does not prove bitwise parity across those paths.
Capture must preserve native dependencies rather than reorder operators to create overlap.
Any bitwise difference remains `UNRESOLVED`.

Capturing raw eager kernels does not reproduce all compiler fusion in the original serving graph.
Even a correct raw graph is not automatically the strongest native comparison.
No latency, graph parity, or native-headroom claim follows from this audit.

The smallest missing evidence is:

1. A real V2 C4 eager capture with exact row identity, owned metadata, and all-false captured padding.
2. A private native builder manifest with its resolved graph configuration, native split bound, schedule pointer, and workspace ownership.
3. One registered TP1 and TP4 capture/replay with exact output and current/history KV parity, plus charged reset and measured workspace capacity.

None of those checks requires a new serving engine.
This audit does not implement them.

## Pre-port source identity and actual time

The pre-port C07 worker hash is `e744aa497c0a6a6c55c01d15d227078aa26cb7ca70ae3b51f3144a8d8c1d1fec`.
The helper hash is `56b7458c261aae24c5534328c05908967d52a18a2b6d4e4d4545b5a57e335e30`.
The source audit segment ran from `2026-10-19 15:21:35 UTC` to `2026-10-19 15:24:50 UTC`.
That interval is 195 seconds.
This note precedes the separately authorized V2 port.

## Later eager-only refinement

The [V2 port contract](c07_v2_eager_capture_port_contract_v1.md) preserves the pre-port facts above.
It documents the separately authorized hook adaptation and a source-backed padding refinement.
That refinement preserves actual flags and uses exact native admission evidence for unquantized dense Ouro.
An all-false flag tensor is not an independent admission requirement for that checked route.
Graph execution remains blocked pending native graph-metadata ownership and replay parity evidence.
