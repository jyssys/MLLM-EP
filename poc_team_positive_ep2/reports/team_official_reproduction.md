# TEAM official positive-control reproduction

## Result

**Gate A: PASS.** TEAM is materially faster than the corresponding SDAR
baseline, its structural work direction matches the paper, and the apparent
quality discrepancy in the short HumanEval run is explained by output
truncation.

| Restart | Requests | Baseline mean | TEAM mean | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 8 | 58.251 s | 24.949 s | 2.335x |
| 2 | 2 | 59.233 s | 34.565 s | 1.714x |
| 3 | 2 | 59.406 s | 32.429 s | 1.832x |
| **Median** | — | — | — | **1.832x** |

Restart 1 used four GSM8K and four HumanEval samples.  Restarts 2 and 3 used
the same GSM8K/HumanEval pair in opposite baseline/TEAM order.  All clean
timings exclude model loading and warm-up.

## Quality boundary

| Bounded setting | Baseline | TEAM | Interpretation |
|---|---:|---:|---|
| GSM8K, 128 tokens | 3/4 | 3/4 | Matched |
| HumanEval, 128 tokens | 2/4 | 1/4 | One TEAM function truncated mid-body |
| GSM8K + HumanEval boundary cases, 256 tokens | 2/2 | 2/2 | Apparent delta disappears |

The bounded subset cannot establish benchmark-level quality equivalence.  It
does establish the positive-control requirement: under the same decoder and
checkpoint, the only observed short-run quality mismatch was an output-budget
artifact.  No speed claim uses an output with a different generation budget.

## Structural direction

The observer-heavy 32-token trace produced:

| Metric | Baseline | TEAM | Change |
|---|---:|---:|---:|
| Model forwards / NFE | 24 | 14 | -41.7% |
| MoE layer calls | 1,152 | 672 | -41.7% |
| Active-expert events | 63,280 | 31,134 | -50.8% |
| Mean active experts / call | 54.93 | 46.33 | -15.7% |
| Token-expert pairs | 307,200 | 342,528 | +11.5% |

The last row is important. TEAM reduces forward count and expert activation
events but sometimes evaluates four speculative candidates in one larger
forward.  It is therefore incorrect to summarize TEAM as uniformly reducing
every physical assignment count.

## Instrumentation boundary

At the same 32-token point, trace overhead relative to clean smoke was +36.7%
for baseline and +12.2% for TEAM.  Structural traces are used only for work
counts; request speedups come exclusively from clean runs.

TEAM's stated mechanism is temporal/spatial routing consistency plus decoded
token caching, restricted expert activation, and speculative exploration.^1
The reproduced NFE and active-expert trends agree with that mechanism.

## Figures

- `plots/01_stage_a_clean_latency.png`
- `plots/02_stage_a_speedup_by_task.png`
- `plots/03_stage_a_work_reduction.png`
- `plots/04_stage_a_quality.png`

## Source

1. Wei et al., “[TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](https://arxiv.org/abs/2602.08404),” 2026.
