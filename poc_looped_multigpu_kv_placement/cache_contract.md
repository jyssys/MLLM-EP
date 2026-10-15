# Cache contract

## Distinct memory accounts

Report logical live KV, unique tensor storage, allocated pool, and total GPU HBM separately.
Do not call logical bytes measured HBM traffic.
Do not infer effective bandwidth from an attention or copy kernel name.
Count storage pointers once when multiple tensor views share an allocation.
Preserve dtype, stride, offset, token positions, and native cache lifetime.

## Ouro: independent recurrence slots

The Transformers reference uses `UniversalTransformerCache`.
Its cache index is `loop * 48 + physical_layer`.
Actual execution materializes 192 distinct K/V slots.
The cached nine-position reference contains 14,155,776 unique KV storage bytes.
This equals nine times 1,572,864 bytes per token.
Weight sharing does not share those K/V values.

The logical BF16 equation is:

`bytes/token = 2 * 192 * 16 * 128 * 2 = 1,572,864`.

Native TP2 and TP4 own eight and four KV heads per slot, respectively.
Their logical payloads are 786,432 and 393,216 bytes per token per rank.
All ranks together retain the same logical total.
This formula does not predict allocated engine HBM.

The vLLM implementation constructs four attention/cache modules per physical layer.
Their unique prefixes span logical layers 0–191.
The native full-attention page size is 16 tokens in the validated probes.
Each request owns paged token slots. Recurrent layers use separate cache storage regions.
The native implementation updates fresh rows without replacing the entire backing pool.
Prefix caching is disabled for measurement. Cross-request cache hits are not a latency gain.

The smoke intentionally gives each rank an 8 GiB cache budget.
Therefore actual pool bytes per rank remain near 8 GiB at TP1 and TP4.
TP4 instead obtains approximately four times the pooled token capacity.
This is not evidence that TP failed to shard KV.
The main serving pool follows the same native 85% HBM policy at each TP degree.
Record its actual block count, storage bytes, and workspace.

For a prompt of S positions and 64 generated tokens, the last returned token has no later forward.
The filled cache therefore reaches S+63 positions in a completed request.
Admission calculations can conservatively reserve S+64 positions with page rounding.
Offered concurrency is not simultaneous resident concurrency when that reservation exceeds the pool.
Queueing or preemption can complete an oversubscribed batch without proving that all requests fit together.

## Huginn: prelude, recurrent depth, and coda

The full dynamic cache uses `cache[step_index][token_position]`.
At fixed 16 steps, observed indices are prelude 0–1, recurrent 2–65, and coda -1/-2.
There are 68 live K/V slots.
The logical BF16 payload is 1,436,160 bytes per token.
A separate real pretrained H32 probe materializes 132 slots.
Its 35 cached positions contain 97,574,400 logical KV bytes and 243,936,000 unique backing bytes.
That tiny H32 probe is not a full-context H32 performance or quality result.

The dynamic cache stores token views into larger Q/K/V allocations.
The actual 32-position plus three cached-step probe retains 125,664,000 unique storage bytes.
Its logical KV elements contain 50,265,600 bytes.
The physical backing is 2.5× the logical payload in that probe.
The K views retain a rotated Q/K allocation. The V views retain a combined Q/K/V allocation.
This is a source-verified implementation layout, not a universal recurrent-model memory law.

The official `compress-s16` probe has 20 slots: two prelude, 16 recurrent, and two coda slots.
The 16 denotes recurrent cache indices, not 16 full recurrent stacks.
That probe reduces storage but changes the cached continuation tokens.
It is not an exact placement-only control or our proposed contribution.
No general accuracy claim follows from its four-token output.

## Correctness boundary

The official pretrained Ouro reference and native TP1/TP4 smoke produce the same four greedy tokens.
Their BF16 log probabilities can differ. Full benchmark equivalence is not inferred from four tokens.
Same-configuration instrumentation must preserve outputs and fixed recurrence.
Native different-TP numerical variation must remain visible in the final results.
No held-out tolerance is relaxed after observing failure.

Huginn's clean and event smoke outputs match with the same seed and full cache.
The first smoke is cold, so its wall time cannot validate event overhead against the warmed second run.
The report uses warmed repetitions for performance and labels smoke timings as diagnostics.
