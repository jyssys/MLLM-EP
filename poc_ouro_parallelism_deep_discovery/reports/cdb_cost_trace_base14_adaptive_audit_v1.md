# Base1.4B adaptive CDB trace audit v1

Functional gates pass. Timing attribution remains `OBSERVER_OR_DRIFT_UNRESOLVED` for all four requests. No quantitative bottleneck or end-to-end cost claim follows. The frozen thresholds remain unchanged.

This audit reads [the completed GPU5 job](../outputs/cdb_cost_base14_adaptive_gpu5_v1/). It does not execute GPU work or change raw files. The other three size/mode cells remain unmeasured by this lane. Suspending them does not establish their results.

## Counts and functional gates

The job used the four frozen calibration IDs, cap 128 per generation, native EOS/stops, and the original adaptive gate. It completed one warm run and three measured phases per request.

| Request | Prompt tokens | Generated tokens per measured phase | Decode NFE per phase | Publication windows in TRACE | Finish |
|---|---:|---:|---:|---:|---|
| `gsm8k/test/106` | 306 | 56 | 171 | 48 | stop |
| `gsm8k/test/1074` | 304 | 85 | 250 | 79 | stop |
| `math500/test/142` | 83 | 128 | 398 | 105 | length |
| `math500/test/156` | 256 | 128 | 417 | 91 | length |

Each request has one full-R4 prefill output. Decode positions and coda windows therefore equal generated tokens minus one. Across the four TRACE requests, the totals are 397 generated tokens, 393 decode positions, 1236 recurrent passes, 323 publication windows, and 393 coda windows.

I recomputed the validator results from all measured phase files. They match the saved [request validity files](../outputs/cdb_cost_base14_adaptive_gpu5_v1/cost_trace/). All functional error lists are empty. Token IDs, finish reasons, per-position exit depths, generated counts, and the checked NFE statistics match across each triplet.

Every measured phase has zero recurrent and stage graph captures. Recurrent graph hits equal recurrent batches. Stage hits equal prelude batches plus coda batches. The TRACE stage-hit counts are 110, 168, 254, and 254, respectively. The event counts match actual recurrent batches, coda batches, and nonfinal-depth exits.

This is output/depth/NFE parity. It is not a hidden-state, logit, or K/V bitwise comparison. It is not an accuracy measurement.

## Timing guards fail

The following values come from request clocks. They are not clean component costs.

| Request | Clean before, ms | Trace, ms | Clean after, ms | Clean drift | Raw trace / clean mean minus one |
|---|---:|---:|---:|---:|---:|
| GSM8K 106 | 967.694483 | 1335.092014 | 1037.439657 | 6.956659% | +33.167351% |
| GSM8K 1074 | 1574.318212 | 1628.405004 | 1896.050713 | 18.541689% | −6.153781% |
| MATH-500 142 | 2149.382033 | 2658.067205 | 2210.267920 | 2.793155% | +21.939478% |
| MATH-500 156 | 2464.231029 | 2864.466460 | 2619.033968 | 6.090689% | +12.701835% |

GSM8K 106 and MATH-500 156 fail both frozen guards. GSM8K 1074 fails the clean-drift guard. MATH-500 142 fails the trace-wall guard. All four have `attribution_permitted=false`.

The negative raw ratio for GSM8K 1074 is not negative observer overhead or a speedup. Its clean walls differ substantially. These triplets do not isolate events, Python bookkeeping, system drift, or another source as the timing cause.

The raw intervals remain diagnostic records. Do not add their CPU and CUDA durations. Do not convert their sums into a 100% breakdown. Do not subtract estimated observer cost or infer removable host/gate/publication cost. No threshold adjustment or additional run is part of this audit.

## Logical ownership and observer lifecycle

I checked all 7359 interval rows. Row IDs are consecutive. Parent IDs point to earlier containing host calls. Every recorded host interval lies inside its TRACE request clock. Child host intervals lie inside their parents. All recorded durations are finite and nonnegative.

Within each request, event-kind/pair-index tuples are unique. Event counts agree with both the interval records and export status. The source pool creates distinct event objects. Each recurrent child has graph key `[2,1,1,depth]`, matching its single-row parent metadata. Each position contains exactly the expected passes from depth zero through its recorded terminal depth.

Each nonempty publication backlog contains one matching request/position and a nonfinal exit depth. Its depth matches the final exit record. Empty backlog calls contain no CUDA pair. Coda metadata also matches the terminal depth for that position.

All four exports record completed event resolution and restored method overrides. No interval records an original or event-cleanup exception. No event pool approaches its recurrent limit: the largest use is 417 of 512 pairs. The guard exits with code zero. Its cleanup record reports no remaining owned GPU contexts and no unrelated process-group changes. The final guard status records burner restoration. This audit does not independently sample subsequent burner utilization.

These checks cover logical identities and recorded lifecycle status. They do not observe device pointers or prove copied K/V contents. The observer does not export a physical allocation or cache-value audit. Source ordering still supplies the publication-before-next-read dependency.

## Missing-head warning audit

The console lists four missing parameters: lookahead gate weight/bias and preloop gate weight/bias. These warnings are real. [OuroModel construction](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/models/ouro/modeling_ouro.py:591) allocates all three heads, even when only one is active.

The source establishes inactivity for this configured path:

1. The pinned checkpoint config has no `exit_gate_type` field. [OuroConfig](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/models/ouro/configuration_ouro.py:177) defaults that field to `early_exit`.
2. The archived [load_engine](../runtime.py:34) changes neither that field nor the gate type. It enables no lookahead or preloop checkpoint.
3. [The adapter](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/adapters/ouro.py:278) binds `decoder.early_exit_gate` for this type. The runtime asserts this exact module identity.
4. The runtime also compares the active gate's loaded weight and bias with the pinned checkpoint tensors. The successful run passed those assertions.
5. [preloop_exit_pdf](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/continuous_depth_batching/adapters/ouro.py:306) returns `None` before head execution for an early-exit configuration. Prefill uses the original non-gated full-R4 path. The monolithic [gate branches](/home/esjung/external/decode-first-cdb-upstream/src/looped_cdb/models/ouro/modeling_ouro.py:846) also distinguish these gate types explicitly.

Thus, the model allocates the newly initialized lookahead and preloop modules but does not use them here. They are not missing active trained gate weights. This conclusion is source-backed. The trace does not contain per-module execution counters. It does not test those inactive heads or establish their quality. Their allocation and initialization do not have zero cost.

## Provenance and time

The archived source snapshot matches all three implementation hashes from [execution_plan.json](../outputs/cdb_cost_base14_adaptive_gpu5_v1/cost_trace/execution_plan.json). It also matches the pinned runtime hash. The tested versions are in the [independent source review](cdb_cost_observer_independent_review_v1.md). The source ZIP does not contain protocol or source-contract JSON. Their identities remain explicit hashes in the job plan and frozen files.

Additional warning-source SHA256 values are:

- Checkpoint `config.json`: `ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3`.
- CDB `configuration_ouro.py`: `67d5ac1ef67bf2f77275f35ddefbf2c34922a3e84cb6965bd3b91539e554c47f`.
- CDB `modeling_ouro.py`: `47c21a0e5f33992ccc40464eff8f41eee414e16413f247caabdb7f975b04b538`.

The audit input digest is `e1279ba7321866ab3b59699867d66235045ba2b859270a4848a932d6919d107e`. It hashes a sorted compact JSON map of relative paths to file SHA256 values. The map covers 36 explicit files. These are eight job-level records and seven records per request. Job-level files are guard status, cleanup validation, run contract, console, source ZIP, execution plan, device record, and trace status. Request records are warm, clean-before, trace, clean-after, validity, export status, and intervals.

The read-only audit segment started at `2026-10-19 16:34:15 UTC`. Checks through `2026-10-19 16:37:27 UTC` occupied 192 seconds. Source-snapshot inventory inspection and report drafting followed. No implementation, raw record, frozen contract, threshold, or GPU state changed through this reviewer.
