# New Migration Correctness and Execution Contract

This is a finite migration qualification. It is not an original-model accuracy claim.

## Source inheritance

- Base release: `b53a8fed36a1706ac74754b53753f20edb8b913e`.
- Numerical release: `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
- Private RLT HEAD: `6b164d3844d35ab8e753df1864f0c87dc180401a`.
- Preserved RLT diff SHA-256: `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
- The source lock checks 1,637 private runtime files.
- Ouro uses `down_norm`. Huginn uses `down_norm_adapter`.
- Both retain BF16 checkpoint values and the full fixed recurrent depth.

The numerical patches change selected reduction geometry.
They add real execution cost. Every candidate and control uses the same patches.
The inherited original-HF fidelity limits remain unresolved here.
GSM8K and MATH accuracy are NOT_MEASURED in this study.

## New retained-state implementation

The initial state capacity is 4,096 prompt rows per active request.
Token chunks remain 128 or 256 rows in the qualification probes.
Prompt rows use absolute offsets rather than a recurrent ring.
The P-side scheduler blocks every depth above the selected cut.
It exports only after all prompt tokens finish the selected recurrent depth.

An intermediate cut transfers all intermediate prompt rows.
Those rows include Huginn's reinjection carrier.
The transfer includes only KV planes that the consumer can validly read.
Huginn Coda KV excludes its not-yet-written final prompt row.
Full PD transfers only the final hidden carrier, not a capacity-sized hidden slab.
The source allocation remains leased through transfer acknowledgement.
The destination resumes the remaining causal recurrent work.

Cut 0 uses the same resident P and D model workers.
It reserves D capacity without reserving P compute capacity.
The D worker performs the full Prefill and Decode.
Cut 0 transfers no KV or hidden payload.

The primary scheduler is `refill` with wavefront G1 Prefill.
Native Mixed is disabled. A guard rejects Prefill/Decode recurrent co-batching.

## Completed midpoint and endpoint probes

All entries below are MEASURED_HEAVY_CORRECTNESS.
These timings must not enter a clean performance comparison.
Each midpoint comparison uses eight measured requests and four warm-up requests per arm.

| Model | Input tokens | Token chunk | Compared cuts | Exact output sequences | Exact final Prompt KV/hidden | Receiver byte checks | Measured/required recurrent rows |
|---|---:|---:|---|---:|---:|---:|---:|
| Ouro R4 | 128 | 128 | 4 vs 2 | 8/8 | 8/8 | 12/12 | 4,576/4,576 |
| Ouro R4 | 512 | 128 | 4 vs 2 | 8/8 | 8/8 | 12/12 | 16,864/16,864 |
| Ouro R4 | 1,536 | 256 | 4 vs 2 | 8/8 | 8/8 | 12/12 | 49,632/49,632 |
| Huginn R32 | 128 | 128 | 32 vs 16 | 8/8 | 8/8 | 12/12 | 36,608/36,608 |
| Huginn R32 | 512 | 128 | 32 vs 16 | 8/8 | 8/8 | 12/12 | 134,912/134,912 |
| Huginn R32 | 1,536 | 256 | 32 vs 16 | 8/8 | 8/8 | 12/12 | 397,056/397,056 |
| Ouro R4 | 128 | 128 | 4 vs 0 | 8/8 | 8/8 | Not applicable | 4,576/4,576 |
| Huginn R32 | 128 | 128 | 32 vs 0 | 8/8 | 8/8 | Not applicable | 36,608/36,608 |

The recurrent-row denominator is `sum((prompt_tokens + output_tokens - 1) * R)`.
The first output token follows Prompt completion and does not add another Decode pass.
These counters exclude fixed-geometry padding rows. Padding is identical across methods.
The physical GPU pairs were 0/1 for Ouro and 2/3 for Huginn.
The two model probes ran concurrently. They are correctness evidence, not isolated timing evidence.

The 1,536-token Ouro probe exceeds its original 4-by-256 prompt-state ring.
Thus it checks more than the original single-chunk A implementation supported.
It does not qualify every possible long-context scheduling combination.

## Failures preserved

The first launch failed before inference because a wrapper argument conflicted with the `peer` field.
The next launch failed because receiver validity metadata was initialized twice.
Both failures remain in their original run namespaces and logs.
The third full-PD launch passed after the local adapter corrections.
No original engine source, model weight, or previous result was changed.

UCX emitted CUDA-context teardown warnings after some completed runs.
The workers still closed with successful status and completed byte checks.
Those warnings are not evidence of a successful transfer by themselves.
The actual receiver checks provide that evidence.

## All-cut independent start

The all-cut independent-start qualification passed for both models.
Ouro passed cuts 0, 1, 2, 3, and 4 on GPU4/5.
Huginn passed cuts 0, 4, 8, 16, 24, 28, and 32 on GPU6/7.
Each cut produced eight exact measured output sequences.
Each cut also passed twelve complete Prompt KV/hidden comparisons, including warm-up requests.
Every nonzero cut passed twelve producer-to-receiver byte checks.
Every cut preserved the required recurrent-row count.
The full-depth endpoint independently repeats the earlier PD_G1 execution on a different GPU pair.
The evidence resides in `state/all_cut_gate_*.json`.

## Next gates

The static map uses independent seeds and crossed GPU pairs.
Unexpected candidate output divergence stops performance expansion for that model.

Raw records reside under `raw/gate_*`.
Each run retains its configuration, source archive, logs, outputs, transport accounting, and worker counters.
The JSON files under `state/` contain exact paired comparisons.
