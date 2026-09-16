# H3 — Block Co-activation and Communication-Aware Placement

## Verdict

**NO-GO.** Neither full-future block-local placement nor early-predicted placement has meaningful full-stage headroom, and expert migration cannot amortize.

## Setup

- Baseline: contiguous simulated EP4 ownership, 64 routed experts per rank.
- Compared: global balanced static permutation learned on requests 0–63, block-local full-future oracle, and placement chosen only from refinements 1–2.
- Same-token top-k co-routing and same-forward/block co-activity graphs were built separately; their graph cosine is 0.690, confirming they are not interchangeable.
- All placements preserve 64 experts/rank and do not duplicate weights.

## Held-out static placement

| Metric | Change versus contiguous EP4 |
|---|---:|
| Current-block remote bytes | **+14.30% worse** |
| Current-block destination fanout | **+14.68% worse** |
| Current-block max assignment load | 5.69% lower |
| Current-block routed-MoE stage | 0.072% worse |
| Full physical routed-MoE stage mapping | 0.038% worse |

The optimizer can make assignment counts look more balanced while damaging source locality. The calibrated dispatch/combine and expert model correctly turns that tradeoff into no latency gain.

## Block-local and early-predicted headroom

| Policy | Full-stage reduction P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| Full-future block oracle | -0.028% | 0.083% | 0.146% | 0.301% |
| Refinements 1–2 only | -0.035% | 0.083% | 0.158% | 0.309% |

Even the impossible full-future oracle is far below the 5% stage gate. Its median remote-byte result is 13.91% worse; the early policy is 15.08% worse.

## Migration cost

The global permutation moves 185 expert IDs per layer, or 3,515 expert instances across 19 routed layers: 22.11 GB of BF16 weights. The measured-endpoint lower-bound migration time is 68.50 ms. Only 1.42% of evaluated blocks had even a nominal early-placement break-even window; most blocks have non-positive stage saving, so their correct break-even is infinite.

## AR-MoE control and interpretation

Static communication-aware expert placement is generic MoE. The dLLM-specific claim would require early refinements to predict a useful later block placement. Here early prediction approximates an oracle whose physical headroom is already zero. This is not `ORACLE_ONLY`; the oracle itself fails.
