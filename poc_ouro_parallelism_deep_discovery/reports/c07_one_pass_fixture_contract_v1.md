# C07 one-pass fixture: pre-run contract

## Status and scope

The separate eager fixture component is ready for review.
Its CPU suite passes 15 tests in both interpreters.
No GPU command or GPU execution occurred during implementation.
Runtime metadata support, bitwise parity, memory feasibility, and service time remain untested.
The [frozen protocol](../artifacts/c07_one_pass_fixture_protocol_v1.json) defines the first run.

This component measures one physical common-core pass on an already ready frontier.
TP4 selects four distinct requests at depths 0, 1, 2, and 3.
It never substitutes four dependent passes of one request.
TP1 selects one request row and one depth.
The selection uses captured native row order, not submission order.
This can screen a perfect-acceptance fixed-R4 frontier.
It does not establish natural frontier availability, acceptance, quality, or closed-loop speculation.

The component does not compare four simultaneous TP1 owners yet.
One TP1 invocation measures one full-weight owner.
Do not divide TP4 time or combine independent TP1 clocks into an observed DP4 makespan.

## New files and native body

The [runner](../scripts/c07_one_pass_fixture.py) uses the existing `gpu_session` guard.
The [worker extension](../c07_fixture_probe.py) modifies only its own process.
The [CPU helpers](../c07_fixture.py) define selectors, page leases, metadata cloning, and byte accounting.
The [tests](../tests/test_c07_fixture_cpu.py) execute CPU logic and the actual one-pass method with fake native objects.
No existing runner, installed source, or frozen output changes.

Both model sizes retain the original R4 model and all 4L cache slots.
The code calls every original physical layer once.
It then calls `core.norm` once.
It resets the residual to `None` at pass entry.
It retains every native normalization, residual operation, and BF16 boundary.
The embedding, gate, LM head, and sampler remain outside the service interval.
Their weights remain resident.

The body follows [OuroModel.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:355).
Each layer uses the original [OuroDecoderLayer.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:260).
The service uses physical slot 0 after fixture preparation.
That physical address does not change the selected row's logical depth.
The imported history and hidden state retain that logical depth.

## Native reference capture

The runner loads a pinned checkpoint through the existing native configuration.
It disables compilation and CUDA graphs.
It retains async scheduling, disables chunking and prefix caching, and allocates 32 GiB of native KV per rank.
The local model-length cap remains 10240.
The batched-token cap becomes 32768 so the four 4096-token prompts can enter together.

Four distinct native requests each generate two controlled output tokens.
The native singleton `allowed_token_ids` path supplies those tokens.
No custom teacher hook runs.
The capture accepts only an actual unpadded C4 cached-decode invocation.
All four rows must have one query token at the required position.
Submitted C4 is not proof that such an invocation occurs.
An unsuitable invocation remains in the rejection ledger.

The extension wraps [GPUModelRunner._model_forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu_model_runner.py:3824).
The native worker already entered its [real ForwardContext](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/worker/gpu_model_runner.py:4376) there.
Hooks copy all four pass-input hidden tensors and all four normalized pass outputs.
Each tensor contains all four request rows.
The extension snapshots each selected row's historical KV before the native decode.
It snapshots the resulting current-token KV after that decode.
Each version records request ID, native row, logical depth, token range, and capture ordinal.

Metadata tensors receive owned copies.
The original Attention objects remain in `no_compile_layers`.
Capture records original and copied tensor pointers, offsets, shapes, strides, and dtypes.
The capture path does not intentionally read a device scalar on the host.
It uses a one-element device index for the current-token snapshot.
Its host enqueue interval includes the native invocation.
That interval is not completed device-copy time.

## Quiescent page lease and address checks

Fixture preparation starts only after all four native requests finish.
The driver must report no unfinished requests.
The worker must hold no pending `execute_model_state`.
The capture then synchronizes outside any service interval.
The extension rejects every subsequent serving call.
The lease lasts until engine teardown.

Every source page through the current token remains protected.
The four source requests must own distinct non-null pages.
The destination pages come from unused IDs in the existing slot-0 pool.
The page selector excludes page 0 and every protected source page.
It gives each selected row disjoint destination pages.
All selected reference tensors and prefix snapshots exist before any destination write.

Each original cache keeps its pointer, shape, stride, and offset.
The component rejects overlapping original cache ranges.
It also rejects interleaved layer arenas.
Those layouts need a different lease proof and exceed this component's scope.
This rejection does not establish that such native layouts are incorrect.

The native packed layout is `[blocks, local_heads, 16, 256]`.
The last dimension contains 128 key values and 128 value values.
Preparation copies only positions before the current token.
It clears the leased current-token address.
The original QKV path must produce the new current-token K and V.

Every destination token address receives a pointer check against its native tensor view.
The manifest retains boundary addresses and complete page maps.
The source slots remain resident and unchanged.
Parity also checks the original selected histories after every service trial.

## FA3 metadata and parity

The component accepts only native BF16 FA3 MHA with block size 16.
It rejects cascade, DCP, speculative/prefill mixtures, microbatching, padding, and unsupported context fields.
It requires the original depth slots to use equal metadata.
TP4 retains the captured FA3 schedule because row counts and sequence lengths remain unchanged.
TP1 rebuilds an existing AOT schedule through the [native scheduler interface](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/vllm_flash_attn/flash_attn_interface.py:122).
If native AOT scheduling was absent, TP1 preserves that policy.
The remaining FA3 fields retain their captured values.

The physical pass runs under [override_forward_context](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/forward_context.py:245).
This context uses the original Attention modules and private fixture metadata.
The component checks these values against the captured native reference:

- The normalized output for every selected row.
- The produced current-token K and V for every physical layer.
- The remapped historical K and V.
- The original selected-depth historical K and V.

The criterion is exact BF16 word equality.
Any difference produces `UNRESOLVED`.
The code never silently increases a tolerance.
Each rank checks its own native reference.
This does not require or establish bitwise equivalence across different TP sizes.
All TP ranks must pass before timed trials start or continue.

## Preparation bytes and clocks

Preparation and service have separate records.
The capture-request wall includes native prefill, decode, snapshots, sampling, and IPC.
Materialization wall includes remap, address checks, and synchronization.
Export wall remains separate.
The byte ledger reports logical payload sizes, not measured HBM traffic.
Parity reads, metadata copies, temporary indices, and export add work beyond those payload sizes.

For each rank, selected-history payload is the same for TP1/one row and TP4/four rows.
This equality follows from their different local head counts.

| Model | Position | Snapshot payload per rank | Destination write per rank |
| --- | ---: | ---: | ---: |
| Base1.4B | 1024 | 201326592 B | 201326592 B |
| Base1.4B | 4096 | 805306368 B | 805306368 B |
| Base2.6B | 1024 | 402653184 B | 402653184 B |
| Base2.6B | 4096 | 1610612736 B | 1610612736 B |

Current-token reference payload is 196608 B per rank for Base1.4B.
It is 393216 B per rank for Base2.6B.
Each service interval includes a working-input reset copy.
That copy is 4096 B for TP1 or 16384 B for TP4.
The mutable working input never replaces the immutable fixture input.

The raw 32 GiB layout provides 2730/10922 blocks for Base1.4B at TP1/TP4.
It provides 1365/5461 blocks for Base2.6B.
At position 4096, source pages use 1028 block IDs.
The additional lease needs 257 IDs for TP1 or 1028 IDs for TP4.
These structural counts fit the proposed raw pool.
The actual runtime still checks its allocated shape and free destination IDs.
This accounting is not an OOM guarantee.

Service records rank-local CUDA events and host wall intervals.
Native CPU barriers occur outside those intervals.
Parity and export also occur outside them.
The RPC wall encloses all trials and diagnostics.
It is not one-pass latency.
The component claims no cross-rank clock alignment or DP4 makespan.

## Why this version remains eager-only

The native [graph_capture context](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/parallel_state.py:1430) registers TP and PP collective capture.
That interface alone does not supply graph-safe FA3 metadata.
The [FA3 builder](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:489) explicitly marks eager `max_num_splits=0` as not graph compatible.
Its [graph branch](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:418) supplies a separate split bound and scheduler storage.
This version does not replace that state with an untested constant.
It implements no CUDA graph path and no custom kernel.
Compiled/native headroom remains `UNVALIDATED`.
An eager-only result cannot establish the strongest native comparison.

## Review commands

Run these commands only after review.
They are examples, not executed results.
Use a new output name for every guarded invocation.

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session --gpus 4,5,6,7 --output outputs/c07_base14_tp4_fixture_v1 --timeout 1800 -- /home/esjung/.venvs/ouro-deep-discovery/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.c07_one_pass_fixture --size 1.4B --tp 4 --context 1024
```

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session --gpus 4 --output outputs/c07_base14_tp1_row2_depth2_fixture_v1 --timeout 1800 -- /home/esjung/.venvs/ouro-deep-discovery/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.c07_one_pass_fixture --size 1.4B --tp 1 --context 1024 --row 2 --depth 2
```

Each rank exports actual fixture tensors to `rankN/fixture.pt`.
Its manifest includes a SHA256 hash, native pointers, page maps, parity records, and diagnostic timing.
Failed captures and failed parity retain diagnostic status.
The runner then shuts down the native engine.
The external guard retains timeout and process-group cleanup responsibility.

## Evidence limits and actual time

CPU tests cover selection, leases, address boundaries, cloning, byte accounting, configuration, and operation order.
They do not execute FA3, TP reductions, GPU page copies, or native forward metadata.
The first actual GPU run can still expose an unsupported native field or numerical difference.
Such a result is a blocker for interpretation, not a reason to relax parity.

The implementation segment started at `2026-10-19 14:47:41 UTC`.
The separate [CPU validation ledger](../artifacts/c07_one_pass_fixture_cpu_validation_v1.json) records its actual end time and source hashes.
