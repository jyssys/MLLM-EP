# 03 — Dynamic schedule search and oracle boundary

Seven n=32 phase schedules and one promoted phase schedule were evaluated as
**actual full-generation trajectories**; they are in
[DYNAMIC_ORACLE_RESULTS.csv](../DYNAMIC_ORACLE_RESULTS.csv), whose
`evidence_type` explicitly says *candidate, not a perfect future oracle*.
The project did **not** construct an exhaustive future-aware per-step O2.
Changing `k_t` perturbs later logits/acceptance, so summing independent
one-step savings would be a fake oracle. The observed schedule is a lower-bound
candidate, not an upper-bound method headroom claim.

For official full GSM8K n=1,319, the observed score-preserving phase schedule
was 0.9/0.825/0.825: 146 vs 144 correct, 9 paired wins/7 losses,
NFE -8.1%, and direct BCT -6.89% versus threshold-0.900. The paired quality
95% bootstrap interval was [-0.455,+0.758] pp. Thus it is supported at
**0.5 pp** non-inferiority but not at **0.3 pp**. It lost seven baseline-correct
individual requests, so it is not a per-request safe/future-trajectory exact
oracle. The full HumanEval pair showed +1 pass and -3.95% BCT; both benchmarks
have low absolute baseline scores, limiting robustness claims.

| Quality budget (official GSM8K full) | Best *observed* evaluated point | Descriptive direct BCT gain vs 0.900 | Evidence boundary |
|---|---|---:|---|
| Exact bounded score / no observed drop | phase 0.9/0.825/0.825 | 6.89% | Score +2/1,319; seven paired losses; not an exact trajectory oracle |
| ≤0.3 pp, paired-certified | static 0.900 | 0% | Phase quality lower CI -0.455 pp misses budget |
| ≤0.5 pp, paired-certified | phase 0.9/0.825/0.825 | 6.89% | One full-set policy restart; below 10% promotion gate |
| ≤1.0 pp, paired-certified | static 0.825 | 8.33% median | Two static restarts vs one baseline; phase does not beat this descriptive static point |
| ≤2.0 pp | static 0.825 | 8.33% median | No extra safer/faster evaluated action |

This table is a *measured candidate frontier*, not a perfect latency oracle.
The dynamic-versus-best-static improvement at 0.5 pp is 6.89%, while at
1.0 pp it is not positive. It therefore fails the spec's 10% E2E gate for
≤0.5 pp and cannot support a new EP method. More expensive future-aware beam
search, a live controller, and RL were not promoted. No theoretical post-Epoch
row-removal credit was added to the measured BCT.
