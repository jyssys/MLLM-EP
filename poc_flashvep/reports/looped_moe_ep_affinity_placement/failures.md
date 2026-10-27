# Failures, deviations, and unresolved evidence

IN_PROGRESS: this is a preview, not a completed PoC.

1. The official IFM evaluator rejects FA4. A separate FA3 build passed its guard and checkpoint forwards.
2. A whole-vocabulary finite check exhausted memory at B16 and prefix 2,048. Bounded views now check every logit.
3. Concurrent state writers shared one temporary filename. Private temporary files fixed the reproduced publication race.
4. Unicode line splitting inflated a dataset count. Byte-level line counting preserved the unchanged JSONL dataset.
5. The original HellaSwag distribution is unavailable. No unofficial replacement enters the results.
6. The Foil evaluation environment differs from the paper requirements. Actual versions remain recorded.
7. The official adapters retain no KV cache. The measured decode is full-prefix recomputation.
8. The research EP bridge is not a production optimized generic EP engine. Its metadata and restoration costs remain material.
9. A fixed-placement traffic model is not a validated wall-time oracle. The O4 comparison remains blocked.
10. Hardware counters for HBM traffic and SM occupancy are unavailable. These metrics receive no guessed values.
11. Heavy CUPTI observations perturb execution. They remain separate from clean timing.
12. A failed whole-E2E observer gate blocks clean-E2E percentage claims. Observer-only evidence remains labeled.
13. Native IFM generation at B16/prefix-2,048 encountered allocator fragmentation. The recovery preserves model arithmetic and records allocator differences.

Huginn initialization depends on the batch-row index.
Matched native and EP evaluations preserve the official batch geometry.
Four independent B1 evaluations are not a substitute for one four-choice batch.

The study does not change the previous NO-GO results.
It does not repeat expert skipping, expert-output reuse, or loop partition pipelines.
No synthetic or analytical result appears as an actual E2E speedup.
