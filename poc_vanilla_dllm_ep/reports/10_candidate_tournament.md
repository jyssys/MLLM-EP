# Candidate tournament

All candidates start below the promotion line. Values are filled from direct
request traces or measured-cost oracles after Stage 0/1. “Perfect” is never
reported as a live speedup.

| Candidate | Semantics | Perfect E2E oracle | Feasible E2E | Main kill reason | State |
|---|---|---:|---:|---|---|
| A RLR-GV | Verifier-exact only | 0% | 0% | Draft costs 66–70%; inadequate union/consensus | KILL |
| B K-step local draft | Needs valid dLLM trajectory verifier | 0.000052% | 0% | K4 survival proxy `7.83e-6`; no live verifier | KILL |
| C BHERC | Exact identical weights | 0.153% | 0.153% | Replica-rank expert penalty dominates; replication collision | KILL |
| D Elastic EP | Exact with resident state | 0% | 0% | EP1 wins every M and captured step | KILL |
| E Conditional scope | Approximate or verifier-exact | 0% quality-safe | 0% | Top-2 rank mass ≥0.9 on only 5.45%; failed verifier signal | KILL |
| F Delta dispatch | Exact dense / approximate codec | 0% exact; 3.11% impossible FP8 cap | 0% | Exact delta has same size; codec cap below gate | KILL |
| G Layer-gated global EP | Approximate | 76.3% impossible free-MoE cap | 0% validated | Lag-1 update rel-L2 49.1%; Epoch/DICE collision | KILL |
| H Overlap | Exact if dependency-safe | 6.23% impossible free-comm cap | 0% | No independent batch-one work | KILL |
| I Near-tie rank coherence | Approximate | 0.59% proportional cap | 0% | High prior-art risk and below gate | KILL |
| J Exact route-layout reuse | Exact | 0.10% loose cap | 0% | Full rank-count vector repeats only 0.144% | KILL |

Promotion gates follow the working specification: below 5% kill, 5–8% weak,
8–12% promising, 12–20% strong, above 20% very strong. A live prototype is
reserved for a credible candidate above 12%, except a rank-local candidate may
receive one additional diagnostic if it clearly removes global EP invocations.

No candidate reaches even the 5% implementation gate. Candidate G's large
number is deliberately shown as an *impossible* free-work bound, not headroom:
its reuse approximation fails the numerical screen and directly overlaps
Epoch/DICE. The exact candidates C/D/F/J are all below 0.2% feasible E2E.
