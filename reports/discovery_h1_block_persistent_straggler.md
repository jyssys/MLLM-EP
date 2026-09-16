# H1 — Block-Persistent EP Straggler

## Verdict

**HOLD (characterization only).** Consecutive refinements have highly persistent critical-rank identities, but most of the apparent EP4 persistence is already explained by a globally dominant rank. The dLLM-specific increment does not clear the prescribed GO gate.

## Evidence boundary

- Workload: GSM8K 128 requests, threshold 0.95, fixed split 0–63 discovery and 64–127 held out.
- Physical evidence: selected-state `TRUE-EP2` replay on GPUs 0–1 with DeepEP dispatch, owner-local BF16 `torch._grouped_mm` expert execution, and reverse combine.
- EP4/EP8 evidence: `SIMULATED-EP4/8-EP2-CALIBRATED`; it is not physical EP4/EP8.
- The replicated-state bridge all-gather is excluded from EP dispatch/expert/combine accounting.
- Before this PoC, the repository had shown average structural imbalance only. It had **not** tested critical-rank persistence.

## Held-out results

| Target | Layer-local adjacent match | Matched-random match | Difference | 95% block-bootstrap CI | Adjacent load cosine | Same-request cross-block match |
|---|---:|---:|---:|---:|---:|---:|
| true-family EP2 model | 98.69% | 87.90% | +10.71 pp | [+10.15, +11.29] pp | 0.999991 | 70.63% |
| simulated EP4 | 99.32% | 84.80% | +13.35 pp | [+12.30, +14.36] pp | 0.999984 | 95.57% |
| simulated EP8 | 98.83% | 81.12% | +16.17 pp | [+14.92, +17.45] pp | 0.999959 | 100.00% |

For simulated EP4, the same-block advantage is statistically stable but below the suggested +15 pp gate. The load-vector cosine advantage over matched random is only 0.00030, far below the +0.10 alternative gate.

## Layer-local versus forward aggregate

| EP4 metric | Result |
|---|---:|
| Layer-local critical-rank run length P50 / P90 / P99 / max | 14 / 24 / 31 / 32 refinements |
| Forward-aggregate adjacent critical-rank match | 97.82% |
| Current-generated-32-row adjacent critical-rank match | 88.42% |
| Early-1 → future critical-rank accuracy | 96.67% |
| Early-1–2 → future critical-rank accuracy | 96.65% |
| Global-majority baseline | 94.14% |
| Previous-block baseline | 93.55% |

The early signature adds only 2.51 pp over the global-majority baseline. The 95.57% cross-block match also shows that the dominant effect is not confined to repeated refinements of one block. This is the main reason the very high raw persistence is not promoted as a new dLLM-specific straggler phenomenon.

## True-EP2 selected-state validation

The original 96-shape expert-compute envelope ended at 8,725 local assignments and misidentified out-of-envelope states reaching 10,793 assignments. The calibration was extended to 512 measured heavy-trace shapes (4,091–11,792 assignments), after which structural critical rank agreed with the measured slower expert rank in all 12 H1 cases.

However, measured adjacent persistence was 100% in both the structurally selected `high_persistence` and `low_persistence` cohorts. Thus true EP2 supports the existence of a stable hotspot but **does not confirm the predicted high-versus-low persistence contrast**. This prevents a GO.

## Tail context

Across held-out simulated EP4 invocations, max/mean rank load is 1.285 P50, 1.624 P95, 1.887 P99, and 1.935 max. Critical-rank expert compute is 0.826 ms P50 and 0.893 ms P90–max under the calibrated replay envelope.

## AR-MoE control and interpretation

A persistent globally hot expert/rank can also occur in AR MoE. The dLLM-specific test is excess same-block persistence over both matched random and cross-block controls. That excess is positive, but the global-majority and cross-block controls explain most of the observed identity stability.

No straggler method should be implemented from this result. A physical EP4 check with expert-popularity normalization would be the appropriate validation if H1 is revisited.
