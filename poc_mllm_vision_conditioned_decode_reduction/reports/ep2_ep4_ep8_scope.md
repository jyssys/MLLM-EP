# EP2 / EP4 / EP8 evidence scope

| Target | Contract | Evidence in this PoC | Latency status |
|---|---|---|---|
| EP2 | TRUE TP1-DP2-EP2, 64 experts/rank, DeepEP | validated baseline path reused; candidate timing not run because both oracle gates failed | N/A |
| EP4 | TRUE TP2-DP2-EP4, 32 experts/rank, DeepEP | validated baseline path reused; diagnostic vector capture physically executed on GPUs 4–7 | diagnostic only; candidate timing N/A |
| EP8 | TP2-DP4-EP8, 16 experts/rank | owner-map/route replay only | **STRUCTURAL_EP8_PROJECTION_ONLY** |

EP8 latency, TPOT, E2E, and speedup are intentionally absent. The old simulator
was not used. Remote-path/byte quantities require an explicit target-topology
source rank; no generic source-rank assumption is silently applied.

Track A produced no promoted modified route, so its final EP8 projection is not
run. Track B's registered low-error mask accepts zero paths, so its safe EP8
effect is zero. A separate 42.637% W16 route-recurrence maximum is retained only
as an unsafe structural upper bound rejected by measured output error.
