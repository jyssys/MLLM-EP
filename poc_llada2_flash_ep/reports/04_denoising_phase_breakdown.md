# Denoising-phase breakdown

## Logical liveness versus physical execution

In the observer-light EP4 trace with submitted batch 16 and microbatch 8:

| Phase | decision-live / physical rows | accepted positions / forward |
|---|---:|---:|
| Early | 82.88% | 23.11 |
| Middle | 49.30% | 20.44 |
| Late | 13.31% | 11.68 |
| All | 41.56% | 17.50 |

The runtime continues to execute dense 32-position blocks while the genuinely
live fraction collapses. This is large, systematic dLLM structure. It is not a
new method opportunity here: Epoch's Expert Atlas, Liveness, and FreshLane
directly target this exact gap.

## EP geometry by phase

The longer batch-2 generation-128 trace keeps physical M=32 and yields:

| Phase | remote fraction | active experts | tiny experts ≤4 rows | rank-load CV | mean rank fanout |
|---|---:|---:|---:|---:|---:|
| Early | 75.03% | 55.34 | 69.21% | 0.483 | 3.054 |
| Middle | 74.91% | 70.92 | 76.09% | 0.413 | 3.068 |
| Late | 74.98% | 79.35 | 79.41% | 0.369 | 3.083 |

Late iterations have fewer decision-live positions but a more diffuse
physical expert working set. That counterintuitive combination explains why
phase alone does not create an obvious cheap EP switching rule.

## Phase timing

Observer-light block shares are stable: MLP is 50.31%/53.44%/54.04% in
early/middle/late, while attention is 30.54%/26.64%/26.52%. The runtime does
not naturally become cheap in late refinement because physical work does not
shrink with logical liveness.

These are CUDA-event trace proportions, not direct clean BCT reductions. The
block-only trace added 19.46% wall at microbatch 8; detailed inner-MoE tracing
added 328.27%, so observer-heavy absolute times are excluded from E2E claims.
