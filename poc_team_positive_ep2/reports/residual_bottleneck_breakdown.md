# Residual bottleneck after TEAM

## Attribution sequence

The first reference trace appeared expert dominated, but that result failed the
trivial-fix attack.  The Python per-expert loop consumed 4.884 s inside TEAM and
made expert execution look like 60.6% of the request.  An existing vLLM fused
expert primitive reduced clean TEAM request latency from 7.751 s to 3.338 s.
Accordingly, the fused control—not the Python loop—is the substrate used for
candidate headroom.

## Matched reference path: baseline to TEAM

These are observer-heavy, same-device CUDA-event sums and are used only for
attribution.

| Stage | Baseline | TEAM | Direction |
|---|---:|---:|---:|
| Router | 354.3 ms | 361.0 ms | +1.9% |
| Route preparation | 476.1 ms | 326.1 ms | -31.5% |
| Dispatch | 176.3 ms | 120.9 ms | -31.4% |
| Expert | 7,541.6 ms | 4,884.2 ms | -35.2% |
| Combine | 814.6 ms | 343.0 ms | -57.9% |
| Whole MoE | 9,450.7 ms | 6,096.9 ms | -35.5% |
| Attention | 1,206.5 ms | 1,059.3 ms | -12.2% |

TEAM reduces the number of model/MoE invocations; router time remains nearly
fixed because every surviving forward still computes fresh gates for all fresh
rows.

## Production-like fused TEAM residual

The clean request denominator is 3,337.7 ms.  Stage spans are from a separate
event-instrumented run whose wall time was 2.48% lower, so percentages are
attribution ratios rather than additive causal effects.

| Residual stage | Attributed time | Share of clean request |
|---|---:|---:|
| Attention | 1,016.1 ms | **30.44%** |
| Other decoder work | 800.0 ms | 23.97% |
| Expert | 342.8 ms | 10.27% |
| Router | 323.5 ms | 9.69% |
| Route preparation | 314.8 ms | 9.43% |
| Combine | 216.6 ms | 6.49% |
| Outside decoder | 153.3 ms | 4.59% |
| Dispatch | 113.8 ms | 3.41% |
| Unattributed MoE wrapper | 51.5 ms | 1.54% |
| LM head | 5.2 ms | 0.16% |

Whole MoE remains 1,363.1 ms, or 40.84% of the clean request.  No individual
exactly removable residual component reaches a credible 10% new-method oracle:
attention is largest but is not an EP residual; expert work is already fused;
router reuse changes semantics; and dispatch/combine are required by remote
ownership.

## Communication and physical work

| Metric | Baseline reference | TEAM reference | Change |
|---|---:|---:|---:|
| MoE calls | 1,152 | 768 | -33.3% |
| A2A collectives | 2,304 | 1,536 | -33.3% |
| Remote assignments | 149,338 | 199,528 | +33.6% |
| Dispatch + combine remote hidden bytes | 1.223 GB | 1.635 GB | +33.6% |

TEAM's speculative M=128 forwards explain this inversion: lower NFE does not
imply lower bytes.  Yet eliminating all dispatch and combine time on the fused
control is only a 9.90% physically impossible upper bound.  A realistic backend
can recover only a fraction of that.

## Fixed versus scaling costs

- **Scales down with fewer forwards:** route preparation, collective call count,
  attention calls, expert launches, combine calls.
- **Can scale up with speculative width:** token-expert assignments and hidden
  payload bytes.
- **Approximately fixed per surviving row/call:** router/gating and framework
  launch/normalization costs.
- **Substrate artifact:** Python per-expert loop fragmentation; existing fused
  execution recovers it.

## Figures

- `plots/08_ep2_stage_before_after.png`
- `plots/09_team_fused_residual.png`
- `plots/10_ep2_communication.png`
- `plots/11_expert_kernel_shapes.png`
