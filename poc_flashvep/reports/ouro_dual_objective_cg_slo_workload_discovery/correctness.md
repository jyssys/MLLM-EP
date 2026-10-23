# Correctness status

The numerical control completed 32 non-reference pretrained comparisons.
All 32 matched isolated hidden states, final logits, KV values, and eight generated token IDs exactly.
The probes cover fixed G2/G3, legal C512→C1024 changes, and G1→G3/G1→G2→G1/G3→G1 cycles.
Every tested prompt completed exactly Full R4.
The pre-existing relative-L2 tolerance remains 0.03. It was not relaxed.

These isolated probes do not establish universal batched-output parity or task-accuracy equivalence.
The previous C-only GSM8K regression remains unresolved.
New objective policies still require natural-generation and frozen benchmark comparisons.

The online selector changes metadata at existing legal boundaries only.
It preserves attention, RoPE, normalization, sampling, model weights, BF16, and depth-indexed KV layout.
It cannot resize a staged active chunk.
A separate prelude executes G=0 and receives no completed-token credit.
The raw work ledger checks actual submitted prompt tokens times executed G after native synchronization.
Every generated token must have exit depth four.
Issued frontiers are not treated as completed output tokens.

The first adaptive smoke found a measurement-wrapper lifecycle bug during policy teardown.
Its first cohorts passed Full-R4 checks, but the next engine could not allocate its cache.
The failed attempts remain quarantined and are not serving-performance evidence.
The weak-reference regression test fails before the cleanup correction and passes afterward.
Reviewed v2 smoke results must pass before broad validation.

The software suite currently contains eleven passing CPU tests.
Those fixtures are not pretrained-model quality measurements.
Data: `numerical_probe_results.csv`, raw pretrained probe records, and `failure_log.md`.
