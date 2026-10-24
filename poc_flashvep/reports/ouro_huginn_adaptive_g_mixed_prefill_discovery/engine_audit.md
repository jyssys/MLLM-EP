# Engine and compatibility audit

## Pinned source

The engine is standalone vLLM-RLT, not production vLLM.
The source pin is `4a350344027960a48384626aa646f1765fda6d4f`.
The isolated worktree is `/home/esjung/external/ouro-huginn-g-mixed-rlt`.
The removable patch remains under `provenance/`.
No previous engine worktree changes.

## Native features

`MixedPolicy.schedule()` combines ready `RECURRENT` and `PREFILLING_RECURRENT` requests.
`Scheduler.take_phases()` retains their individual recurrent depths and causal prompt ranges.
`build_execution_batch()` creates immutable `StateSpan` records.
One shared `ExecutionBatch.loop_count` applies to every span.
An incompatible multi-loop Mixed descriptor is therefore invalid.

`ModelRunner` owns native event dependencies, retained prompt states, page metadata, and graph buffer lifetimes.
The patch does not add attention kernels or replace those ownership mechanisms.
Ouro retains native async scheduling and recurrent graph replay.
Prompt-only recurrence remains eager in this source.
Mixed recurrence can exceed the configured graph width and fall back to eager execution.
Equal graph settings do not imply equal graph execution.

## Huginn support gap

The original `StageEngine` explicitly rejects non-atomic Huginn prefill.
It also rejects Huginn async scheduling and prefill UVA.
Huginn supports native synchronous recurrent/coda graphs, not Ouro's async path.
Removing the first guard alone would not make Split or Mixed correct.

Huginn retains two tensors at every recurrent frontier.
The first tensor is the changing recurrent state.
The second tensor is the fixed prelude injection.
Prelude and coda layers use their own depth-zero KV namespaces.
Four recurrent layers use sixteen distinct depth planes in this study.
The fixed configuration executes 2 prelude + 16×4 recurrent + 2 coda layer calls per token.
The checkpoint's default recurrence is 32, not the registered inference depth 16.

## Neutral Huginn extension

The common adapter supplies original prelude KV metadata for split prompt spans.
It repeats each request's native initialization generator across that request's token rows.
The existing model retains the recurrent state and fixed injection together.
Each completed prompt span also executes coda KV writes for its non-final prompt rows.
The final prompt row proceeds to ordinary sampled coda.
Thus the adapter does not omit boundary work when it exposes recurrent yield boundaries.

Every Huginn Split, Mixed, grouped-only, and hybrid arm uses this same adapter.
Atomic retains its original prepared-loop execution path.
All arms remain synchronous. Async support is not silently added.
Huginn's checkpoint context limit is 4096.
P8K and larger contexts remain unsupported rather than forced with altered RoPE.

## Grouped prompt patch

`SchedulerConfig.prefill_loop_group_size` specifies a positive maximum group.
`ScheduledRequest.loop_count` snapshots the actual legal group for that prompt range.
The scheduler clips the group at the remaining recurrence depth.
Frontier updates require an initialized causal prefix at every advanced depth.
The executor carries the group into its immutable descriptor.
The native runner executes all required recurrent passes with their original depth metadata.
It retains prompt state until the group reaches full fixed depth.

The Fixed Hybrid wrapper changes the group only before native batch selection.
If compatible prompt and decode rows are ready, it selects native Mixed G1.
Otherwise pure prompt recurrence can use its registered fixed G.
No request executes grouped decode.
The executor rejects a decode span with a non-unit loop count.
This event-based combination is a control, not the proposed adaptive method.

## Scope and interpretation

The first real-checkpoint singleton probes compare valid hidden, logits, and KV planes.
They do not prove batched task accuracy.
The later batched audit uses a separate heavy lane.
Clean comparisons retain natural outputs and label output caps explicitly.
Lazy graph work after request arrival remains inside delivered serving latency.
The paired warm confirmation records actual capture/replay/fallback counts.
