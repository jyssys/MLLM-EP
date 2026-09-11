# Heavier-regime scaling

## Method

The first MoE layer was replayed with fixed BF16 inputs and fixed top-8 routes.
For every `(active-size label, batch)` factorization, local physical work was
`M = active * batch`. Equal-M controls used the same input and routing. Each
point used 5 warmups, 30 timed repetitions, and 3 independent engine restarts;
the logical latency is the maximum same-device CUDA-event duration across the
two EP ranks.

The `active-size` label in this controlled replay is deliberately a work-size
knob. It must not be confused with the unresolved mask count in stock dInfer;
the real-trajectory experiment below shows that those quantities are not the
same.

## Rank-critical medians

| Local physical M | Naive dispatch | Naive expert | Naive combine | Naive MoE | HT dispatch | HT expert | HT combine | HT MoE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.256 | 0.427 | 0.097 | 0.965 | 0.286 | 0.316 | 0.080 | 0.993 |
| 8 | 0.263 | 0.430 | 0.099 | 0.968 | 0.296 | 0.315 | 0.084 | 1.002 |
| 32 | 0.264 | 0.437 | 0.104 | 0.980 | 0.292 | 0.318 | 0.081 | 1.002 |
| 64 | 0.260 | 0.468 | 0.096 | 0.982 | 0.288 | 0.353 | 0.061 | 1.011 |
| 128 | 0.261 | 0.527 | 0.040 | 0.989 | 0.290 | 0.419 | 0.057 | 1.022 |
| 256 | 0.262 | 0.573 | 0.066 | 1.054 | 0.296 | 0.467 | 0.058 | 1.085 |

All values are milliseconds. Full tables include M=2/4/8/16/32/48/64/96/
128/192/256.

## Crossover characterization

There is a stage-composition crossover but not a backend-winner crossover:

- Naive dispatch is essentially flat at 0.256--0.264 ms.
- Naive expert fraction rises from 44.3% at M=2 to 54.3% at M=256.
- DeepEP HT expert fraction rises from 31.8% to 43.0%.
- Despite the changing fraction, full MoE grows only 9.2% (Naive) and 9.3%
  (HT) from M=2 to M=256. Startup/runtime overhead remains substantial.
- At equal physical M, alternate active/batch factorizations differ by only a
  few microseconds in the typical case. The fused MoE only sees the flattened
  physical work, not the semantic factorization.

Thus heavier work makes expert compute relatively more important, but the
measured range does not reach a qualitatively compute-dominated regime in
which a different backend wins.

## Real dInfer work accounting

The more important negative result is that unresolved positions do not become
a smaller physical EP payload in stock dInfer:

- Batch 1, cache off, gen=64: masked positions fall 64 -> 1 while every layer
  continues to process M=128.
- Batch 4, cache off, gen=64: masked positions fall 256 -> 1 while every layer
  continues to process M=512; 42 model forwards were observed.
- Batch 1, prefix cache, fixed gen=256: within each block M is constant even as
  the block's mask count falls. Physical M changes only at prefix/block
  boundaries (320 refresh calls and steady 256/192/128/64 windows).

This falsifies the premise that early/middle/late active ratio directly exposes
multiple EP message-size regimes on this dense forward path. Reducing the
physical fresh work would instead be an Epoch-like logical-work optimization,
which is outside this PoC.

## Request controls and observer tax

The production-like structural control used prefix cache, block length 64,
fixed generation length 256, disabled EOS, and 25 physical forwards. It is
closer to the official dInfer benchmark than the old gen=64/cache-off run, but
still deliberately bounded rather than a full gen=1024 dataset benchmark.

The gen=64 trace run was about 77.3% slower than the comparable clean Naive
median. Route tracing adds gate work, CUDA events, and synchronization, so
traces are used only for structure. Clean runs determine request latency.

Data: `analysis/scaling_by_shape.csv`,
`analysis/scaling_by_physical_m.csv`, and
`analysis/trajectory_structure.csv`.
