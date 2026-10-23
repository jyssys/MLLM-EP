# Correctness status

The numerical control completed 32 non-reference pretrained comparisons.
All 32 matched isolated hidden states, final logits, KV values, and eight generated token IDs exactly.
The probes cover fixed G2/G3, legal C512→C1024 changes, and G1→G3/G1→G2→G1/G3→G1 cycles.
The tested prompt lengths are128,512,1536, and2048 tokens.
They do not establish isolated numerical parity for every8K serving state.
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
All four reviewed v2 smoke jobs subsequently passed both policy cells.

The software suite currently contains 13 passing CPU tests.
Those fixtures are not pretrained-model quality measurements.
Data: `numerical_probe_results.csv`, raw pretrained probe records, and `failure_log.md`.

## Observer and batched natural outputs

The separate event audit compares observer ON/OFF under the same nominal configurations.
Token IDs match on 87/96 request pairs. Per-request prompt submission plans match on 95/96 pairs.
Prompt-plan equality does not prove equal decode batch membership, graph geometry, or native execution order.
Mixed also has mismatches without the adaptive selector.
The earliest mismatch in the saved audit occurs at output index12, with zero-based indexing.
These results do not establish that differences are harmless BF16 rounding.
They also do not prove a KV or event bug.
The exact root cause remains unresolved without a matched whole-batch numerical replay.

Observer feedback can change causal scheduling and generated work.
The ON/OFF wall ratio is therefore not a pure measurement-overhead estimate.
No failed parity pair receives an accuracy PASS from this diagnostic.
The independent full-answer question pilot remains necessary.

Data: `observer_execution_diagnostics.csv`, `observer_output_parity.csv`, and `event_budget_residuals.csv`.
