# Track 3 — Vision-Prefill vs Text/Decode EP Policy Asymmetry

**Decision: NO-GO.** Phase-specific optima exist weakly for single-unit latency,
but their lower envelope improves MoE latency by only 1.62% and projected
prefill TTFT by 0.71%; throughput headroom is effectively zero.

## Supported policy axes

- DeepEP HT `num_sms`: 4/8/12/16/20, source-verified and randomized.
- Whole-unit aggregation: prefill 1/2/4 and decode 1/4/16/64. Aggregation
  concatenates complete captured request/layer units into one Dispatch → Expert
  → Combine invocation. It never cuts one request along the token axis.
- Communication/expert batching granularity is changed together by the single
  compact invocation, matching the actual stock data path.
- Stream priority: unsupported by a safe exposed runtime API in this installed
  vLLM/DeepEP path, so it was not fabricated via an unrelated stream.

Each point has three warmups and ten measured iterations; rank-critical medians
are reported.

## Single-unit latency curves

| Workload | sms=4 | sms=8 | sms=12 | sms=16 | sms=20 | best |
|---|---:|---:|---:|---:|---:|---:|
| Text prefill | 1.796 | 1.402 | 1.320 | 1.289 | **1.265** | 20 |
| Vision prefill | 1.671 | 1.313 | 1.247 | **1.228** | 1.289 | 16 |
| Decode M=1 (median of two independent modality runs) | **0.875** | 0.880 | 0.886 | 0.907 | 0.882 | 4 |

There is a real direction difference, but its magnitude is small. With one
static SM choice, sms=16 minimizes the equal-phase sum at 3.424 ms. The
phase-specific choices (20/16/4) total 3.368 ms: **1.624% MoE latency oracle**.
Using measured prefill MoE shares gives only **0.715% projected TTFT gain**.

## Aggregation and throughput

Whole-request aggregation strongly amortizes the MoE operator: at sms=20,
per-unit text cost falls from 1.265 ms (g=1) to 0.732 ms (g=4), and vision from
1.289 ms to 0.692 ms. This is generic batching, not modality-aware policy.
A single static “fill all currently available whole requests” rule selects g=4
for the four-request prefill pool and g=64 for the decode pool. After allowing
that obvious demand-driven behavior, the best static sms=20 and phase-specific
sms choices differ by only **0.00036%** in the summed per-unit throughput
objective.

Thus it would be incorrect to call g=4 vs g=64 an adaptive phase oracle: one
static large aggregation cap naturally receives only four prefill requests and
64 decode units. The trivial batching rule recovers essentially the entire
throughput envelope.

## Prior-art/triviality risk

DeepEP explicitly supports SM-count control, and current DeepEP V2 targets
minimal SM occupation ([official README](https://github.com/deepseek-ai/DeepEP/blob/main/README.md)).
More directly, the September 2026 paper
[Analytical Resource Management for Fine-grained MoE Computation-Communication Overlap](https://arxiv.org/abs/2609.07536)
selects communication CTA/resource partitions from workload geometry. A
sub-1% request oracle here is therefore both economically and noveltly weak.

## Gate

- Best-static to phase-specific MoE latency >=10%: **FAIL** (1.624%).
- Amdahl-adjusted TTFT >=10%: **FAIL** (0.715%).
- Throughput oracle >=10%: **FAIL** (0.00036%).
- Modality curve difference: **weak**, not economically material.
- Correctness: **PASS**.
- Recommendation: do not implement phase-specific EP control.
