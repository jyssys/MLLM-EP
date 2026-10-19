# C07 P1 bitwise failure analysis v1

Status: `UNRESOLVED`. The fixture failed its initial bitwise check. The cause remains unknown. No service timing is valid.

This audit used source inspection and retained CPU tensors. It did not execute a model. It did not change the fixture, archive helper, installed packages, or raw output. The parent did not authorize another GPU run.

## Observed run and evidence

The job used Base Ouro-1.4B, TP1, physical GPU 5, prompt length 1024, captured row 2, and logical depth 2. Depth indices start at zero. The checkpoint revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

The job directory is [fixture v2](../outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/). Its [capture record](../outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/capture_finish.json) proves a real C4 cached-decode invocation. It records four distinct request IDs, four rows, four tokens, and one scheduled token per row. Every row had 1024 computed tokens. Query offsets were `[0,1,2,3,4]`. Sequence lengths were 1025, and positions were 1024.

The sealed cohort completed its native pause, enqueue, fence, resume, and completion sequence. The preceding prefill invocation correctly failed the cached-decode admission gate. The earlier v1 admission failure remains separate. Its cause remains unknown because that run did not retain CPU batch counts.

The [service record](../outputs/c07_base14_tp1_row2_depth2_gpu5_fixture_v2/1.4B_C07_TP1/service.json) contains zero service trials. Its initial check rejected the normalized pass output and current K/V. The bitwise criterion remains unchanged.

## Measured differences

The [CPU result](../analysis/c07_p1_failure_cpu_v2.json) measures retained BF16 values with CPU float64 arithmetic. The final normalized output has shape `[1,2048]`. Both tensors contain only finite values.

| Quantity | Result |
|---|---:|
| Different BF16 words | 1889 / 2048 |
| Difference L2 norm | 0.5683764158897796 |
| Reference L2 norm | 31.52992629988989 |
| Relative L2, difference norm / reference norm | 0.01802656975737252 |
| Maximum absolute difference | 0.125 |
| Mean absolute difference | 0.009041785262525082 |

Layer 0 current K/V matches exactly: 0 different words out of 4096. Layer 1 is the first measured current-K/V mismatch: 1904 / 4096 words. Its K part differs in 776 / 2048 words. Its V part differs in 1128 / 2048 words. Every later layer also has a current-K/V mismatch. The CPU result retains each layer's counts and numerical metrics.

The original GPU checks report exact selected-history equality at all 24 layers. They compare 100663296 BF16 words for the destination prefix. They compare another 100663296 words for the original source prefix. This scope covers row 2, depth 2, positions 0 through 1023. It does not cover every request and depth. The archive does not contain a second post-service destination-history tensor. Thus, these history results remain GPU check records, not new CPU remeasurements.

## CPU and source checks

The byte archive decodes to 204219324 bytes. Its SHA256 is `901df799ba7bc98a54fbc6611843e7adbc13c2e71e7a29a06418ab33c7751777`. The archive check matches the original rank manifest.

The CPU checks establish these bounded facts:

- The fixture input exactly matches captured pass input `[depth=2,row=2]`.
- The fixture reference exactly matches captured normalized output `[depth=2,row=2]`.
- Each captured normalized output exactly matches the next pass input, for all four rows and three boundaries.
- All 96 captured native metadata entries have equal field values.
- The fixture table and current slot mapping match the recorded lease.
- Every leased page is non-null and disjoint from all source pages.

The real padding flags were `[true,true,true,true]`. The fixture retained the selected flag. This audit did not replace flags or reinterpret actual padded rows. The prior dense-route proof remains the basis for treating these flags as unused state. Actual unpadded admission comes from row/token counts, CPU arrays, positions, metadata, and written K/V.

The source retains the native operation order. [OuroDecoderLayer.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:260) calls the original attention, MLP, and four norm modules. Its input norm has separate residual-present and residual-absent branches. [OuroModel.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:367) resets the residual at each pass and calls `core.norm` afterward. The fixture [one-pass body](../c07_fixture_probe.py:420) preserves these boundaries and calls the original layer modules. It does not replace a norm, residual operation, or cast.

`current_ut` selects only the Attention object in [OuroAttention.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/models/ouro.py:198). The layer's learned projections and norms remain the same physical modules. The fixture uses slot 0 with copied selected-depth history. This source check does not prove equal runtime arithmetic.

## Unresolved causes

Three hypotheses remain distinct. None is an established cause.

1. C4-to-C1 GEMM arithmetic may differ. [UnquantizedLinearMethod.apply](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/linear.py:220) uses the default non-batch-invariant path. [The CUDA dispatcher](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/utils.py:338) selects `F.linear`. The row count changes from four to one. The record does not identify selected GEMM algorithms or their intermediate results. A matching row under matched arithmetic would contradict this explanation at that operator. No such operator-level experiment occurred.
2. C4-to-C1 FA3 scheduling may change attention arithmetic. The native builder uses the request count in [its schedule call](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:632). The fixture rebuilds scheduling for one row through [the same interface](../c07_fixture_probe.py:375). Both metadata records retain `max_num_splits=0`, the native heuristic setting. Their 13-word scheduler tensors differ. This is an observed schedule difference, not evidence of invalid metadata. The audit does not decode unused scheduler words or infer a split count from them. Equal attention results under matched Q/K/V and an equivalent schedule would contradict this explanation. Those results are absent.
3. An unobserved fixture-state mismatch may exist. The retained input, reference, page selection, and selected histories pass their checks. These checks reject several simple selection mistakes. They do not establish every internal attention state or kernel input. A state mismatch needs direct evidence before a source defect can be named.

The first current-K/V mismatch occurs at layer 1. This does not identify the first unequal arithmetic operation. [Attention.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/model_executor/layers/attention/attention.py:488) writes K/V before attention consumes the cache. Layer 0 K/V equality does not prove layer 0 Q equality. It also does not prove equal attention output, output projection, norms, MLP output, or layer 1 input. The archive does not retain these intermediates.

The original FA3 call consumes the query, page table, sequence lengths, and scheduler tensor. See [FlashAttentionImpl.forward](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/attention/backends/flash_attn.py:1014). Correct copied bytes alone do not prove equal attention arithmetic or address interpretation. TP1 also rules out a cross-rank TP reduction as the cause of this particular run. Compilation mode and graph mode were both `NONE`. A reused compiled serving graph is not established here.

The fixture changes decode counters from native zero to one. The native builder leaves these counters zero on its DCP1 path. Its standard DCP1 forward does not read them. Their consumers are inside `_forward_with_dcp`. These fields do not expose a direct DCP1 semantic error.

## Validation limits and disposition

This run establishes real C4 admission, retained pass boundaries, selected-state identity, and the recorded lease/history checks. It does not establish the fixture's physical-pass equivalence. No timing, speedup, graph parity, or strongest-native comparison follows. It does not establish TP4, Base2.6B, other selected rows/depths, or other context lengths.

The numerical difference is not a quality measurement. No quality loss or acceptance-rate change follows from these controlled tensors. This failure also does not disprove the general C07 architecture idea.

The fixture remains blocked by exact parity. The missing evidence is an operator-level comparison across the first unequal boundary, under the same selected state. That evidence would require a separately authorized experiment. This audit does not request or perform another run.

## Reproduction, hashes, and time

The [CPU diagnostic](../analysis/c07_p1_failure_v1.py) reads the verified archive into memory. It loads tensors with `map_location='cpu'` and `weights_only=True`. It writes only a new analysis file. With `--assert-bitwise`, it exits with `RETAINED_FIXTURE_BITWISE_FAILURE`. Two runs reproduce identical comparisons and metrics. This is a retained-pair failure check, not a CPU replay of CUDA operators.

The [CPU result](../analysis/c07_p1_failure_cpu_v2.json) records 15 source hashes that match the pre-run pins. These comprise 12 native files and three campaign files. It also records hashes for five additional inspected native files. Six raw JSON hashes match before and after the diagnostic. The original fixture and archive helper remain unchanged by this audit.

The existing 42 CPU tests pass. The new diagnostic passes syntax checking. A source AST test checks the depth selector and native norm-call structure. The first command used a Python environment without Torch and failed before tensor loading. The subsequent commands used the existing reference environment. No package change occurred.

The source and CPU segment started at `2026-10-19 15:53:12 UTC`. It ended at `2026-10-19 15:57:17 UTC`. The elapsed time was 245 seconds. Report drafting and lint followed this segment.
