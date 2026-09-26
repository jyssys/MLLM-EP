# Published artifact scope

This Git branch contains the Part 1 implementation, tests, reports, frozen
cost model, compact manifests, public benchmark score records, low-resource
acceptance evidence, and virtual EP2/EP4/EP8 replay outputs.

The following large local evidence is intentionally not committed:

- model weights and benchmark images;
- physical-anchor tar archives and per-rank raw profiler streams;
- profiler raw archives;
- full public-evaluation answer/timing streams.

The authoritative local bundle manifest is
`validation_runs/20260924_completion_v1/frozen_baseline_bundle.json`. It lists
the omitted anchor archives and full evaluation streams with byte sizes and
SHA-256 digests. Their omission does not change the report verdict: physical
EP2/EP4 and baseline quality evidence were collected, while the frozen latency
simulator remains `SIM_UNVALIDATED` and EP8 remains an unvalidated sensitivity
projection.
