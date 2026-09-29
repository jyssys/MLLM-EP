# PoC-1 — TRUE EP4 rank-aware physical bridge

Evidence: **MEASURED_TRUE_EP4_FROZEN_INPUT_ROUTE_INTERVENTION**. Verdict: **STRUCTURAL_ONLY_NO_RUNTIME_GAIN**.

The controlled `SAME_PAIR_COLLAPSE25` lane transforms each condition's current native route, keeps every path on its original owner rank, and changes only the within-rank expert identity. Path count and per-rank destinations are exact within each invocation; cross-wave matching is measured explicitly below.

## Primary paired results

| Condition | max-rank active reduction | expert execution gain | decode MoE gain | dispatch gain | combine gain |
|---|---:|---:|---:|---:|---:|
| NATIVE_TAIL_K6 | 16.058% | -0.236% [-2.476, 1.872] | -1.820% [-6.495, 2.556] | -5.637% | -8.404% |
| RANK_AWARE_K6 | 18.724% | 1.261% [-0.538, 3.106] | 1.638% [-2.269, 5.214] | 0.425% | -1.891% |
| SAME_PAIR_COLLAPSE25 | 11.138% | 0.424% [-1.505, 2.351] | -0.677% [-4.951, 3.391] | -4.048% | -6.530% |

Rank-aware increment over native tail in max-rank active-expert reduction: **2.667 pp**. Exact cross-run pair/rank-load match: **False**; near-match gate: **True**. Each collapse transform itself preserves the current invocation's rank destinations and path count exactly.

Direct matched K6 rank-aware vs native-tail expert-stage gain: **0.484%** (95% CI [-1.185, 1.990]); decode-MoE gain: **0.716%** (95% CI [-3.012, 3.575]).

No TPOT/E2E claim is made; this report isolates the physical MoE bridge.
