# Shape-transition / state-control diagnostic

These runs are fresh live Qwen3-VL vLLM V1 measurements, not synthetic replay.
They keep the same TP2/DP2/EP4 DeepEP high-throughput runtime and only change
the request template used for warmup and measured waves.

| control | warmup | measured waves | T_MoE p50 | event-wait p50 | E2E p50 | interpretation |
|---|---|---:|---:|---:|---:|---|
| text fixed | text slot 0 | text slot 0 | 1.161 ms | 0.018 ms | 843 ms | shape matched baseline |
| high-res vision fixed | high-res vision slot 3 | high-res vision slot 3 | 1.204 ms | 0.028 ms | 918 ms | shape matched baseline |
| text → high-res vision | text slot 0 | high-res vision slot 3 | 2.395 ms | 0.334 ms | 1,350 ms | persistent transition penalty |
| text ↔ high-res vision | text slot 0 | alternate slots 0/3 | 2.12 ms | 0.49 ms | 1,296 ms | penalty recurs on alternation |
| text-only shape alternation | text slot 0 | alternate slots 0/1 | 1.21–1.25 ms | 0.02–0.04 ms | 934 ms | modality/encoder boundary matters |
| telemetry-tagged text → vision replication | text slot 0 | high-res vision slot 3 | 1.20 ms | 0.04 ms | 952 ms | transition effect absent; clocks ramped 345→1980 MHz |

The first text→vision run is a state-sensitive observation (roughly 2×
layer-local T_MoE and ~0.33 ms additional event-wait), but the
matched high-resolution warmup removes it.  The text-only shape control does
not reproduce the penalty, and the independent telemetry-tagged replication
does not reproduce it either.  C1c is therefore closed as
`STATE_CONFOUNDED/TRIVIAL_ENGINEERING`, not promoted as a method claim: static
shape-aware warmup/bucketing is an obvious fix, and the E2E numbers are not
request-matched across modalities.

The first telemetry run also observed GPUs 1–4 ramping from 345 MHz to
1980 MHz.  Clock/DVFS state is recorded as a cofactor; no utilization number
is inferred from it and no paper claim is made without a clock-controlled
replication.
