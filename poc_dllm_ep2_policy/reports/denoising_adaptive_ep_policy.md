# Denoising-adaptive EP policy

## Real trajectory

The decisive paired trace is a fixed-length, prefix-cache request with prompt
64, generation 256, block length 64, EOS disabled, and 25 physical forwards.
Naive and DeepEP HT generated identical tokens. For each physical forward, the
latency is the sum of rank-0 same-device CUDA MoE spans over all 16 layers.
No cross-GPU absolute timestamps are subtracted. Rank 0 is also the larger
aggregate stage-time rank for both traces (Naive 436.4 vs 395.9 ms across the
three hooked stages; HT 324.8 vs 286.3 ms), so it is the relevant traced
critical side at request aggregate. Per-layer rank-1 spans were not logged;
this limitation makes the oracle structural rather than a production switcher
validation, but cannot turn a rank-0-critical 0.144% bound into the required
5% without an unobserved and contradictory critical-rank reversal.

The same active ratio was paired across backends. Four cache-refresh iterations
contain two physical forwards and are retained separately because their M
differs.

| Active-ratio bin | Physical forwards | Naive winners | HT winners | Naive total (ms) | HT total (ms) |
|---|---:|---:|---:|---:|---:|
| Early (>0.67) | 12 | 1 | 11 | 244.953 | 202.883 |
| Middle (0.33--0.67) | 1 | 0 | 1 | 19.148 | 17.106 |
| Late (<=0.33) | 12 | 2 | 10 | 219.409 | 195.680 |

There is no early-HT/late-Naive crossover. HT wins every phase in aggregate.
The three isolated Naive wins are sub-millisecond trace fluctuations, not a
stable regime.

## Best-static and perfect oracle

| Policy | Observer-heavy 16-layer MoE sum (ms) | Gain vs best static |
|---|---:|---:|
| Static Naive | 483.510 | -16.32% vs HT |
| **Best static: DeepEP HT** | **415.669** | 0% |
| Perfect per-forward future oracle | 414.946 | **0.174%** |
| Best active-ratio threshold | 415.669 | **0%** |

The impossible oracle saves 0.723 ms over 25 physical forwards. Dividing that
optimistic saving by the clean DeepEP HT request median (501.984 ms) gives a
request-level upper bound of **0.144%**. It ignores switching cost and therefore
favors the hypothesis.

The best active-ratio threshold is the degenerate policy “always DeepEP HT” and
recovers 0% of the oracle saving. This is causally consistent with the work-
accounting result: active ratio changes while physical M remains constant
within a block.

## Logical work versus runtime cost

This PoC preserves top-k, selected experts, and dense physical forward work.
The mask count is a semantic decoding state, but the stock runtime still sends
the full model window through MoE. Turning the mask count into a smaller EP
payload would remove or reuse logical work; it is not an EP backend policy and
is explicitly excluded here.

The measured 6.01% clean benefit is static replacement of Naive with DeepEP HT.
The adaptive residual after choosing that best static backend is only 0.144%
of request latency under a zero-cost future oracle.

## Decision

**EP2 NO-GO.** The <5% hard gate fails by roughly 35x. No threshold policy,
dynamic backend switch, or production controller is justified.

Raw paired data: `analysis/paired_real_trajectory.csv`; oracle:
`analysis/summary.json`.
