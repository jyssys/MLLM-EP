# Final decision — EP-Cost-Aware Adaptive Refinement Budget PoC

**Label: `GENERIC-DLLM-SIGNAL`.** A phase-only acceptance schedule shortened
complete LLaDA2.0-Flash EP4 generations on matched GSM8K and HumanEval pools,
but it did not satisfy the working contract's direct-E2E promotion gates or
establish any incremental value from physical MoE/EP cost. It is not a new
dLLM-MoE-EP method, a perfect future-aware oracle, or a production speedup.

The attached [working contract](/home/esjung/MLLM-EP-github/poc_flashvep/reports/ep_cost_aware_adaptive_refinement_budget_poc_spec.md)
was read from the original repository (SHA-256
`7fd3901ee51132eb495ab24a34500a9d799584f0e6f42909fe5e91cd06386f62`).
Its Section 9 expressly requires *full HumanEval if feasible* and a
*substantially larger GSM8K subset or full set* before sub-percentage-point
quality claims. Accordingly, the 32-item screens were used only to choose a
candidate; official full GSM8K **1,319** and full HumanEval **164** were
evaluated with paired benchmark scoring and bootstrap quality intervals.
One GSM8K item is 0.0758 pp and one HumanEval item is 0.610 pp. The bounded
32-item score has 3.125-pp resolution and was not used to claim 0.3/0.5 pp.

## Measured result and gate

All headline times are clean, complete-pool end-of-generation BCTs under the
same checkpoint, temperature 0, block 32, BF16, DeepEP normal, dense TP4 /
routed EP4, and matched prompts/harness. The semantic-only phase policy was
thresholds **0.9 / 0.825 / 0.825** for early/middle/late; it uses no expert or
rank-load information. HumanEval mini32 exhausted HBM, so the matched full
HumanEval control uses mini16. [Baseline truth](00_baseline_truth.md),
[clean E2E validation](09_e2e_validation.md), and
[machine-readable frontier](../QUALITY_LATENCY_PARETO.csv) give provenance.

| Official matched task | Baseline → phase score | Paired quality 95% interval | Baseline → phase clean BCT | Direct gain | NFE reduction | Restart boundary |
|---|---:|---:|---:|---:|---:|---|
| GSM8K 1,319 | 144 → 146 correct (+0.152 pp) | -0.455 to +0.758 pp | 227.782 → 212.089 s | **6.89%** | 8.09% | One full-set restart each |
| HumanEval 164 | 13 → 14 pass (+0.610 pp) | 0 to +1.829 pp | 47.155 → 45.294 s | **3.95%** | 8.78% | One full-set restart each |

The GSM8K paired lower bound supports a **≤0.5-pp** non-inferiority reading
but not **≤0.3 pp**; the observed 6.89% gain misses the spec's **≥10%**
E2E gate at ≤0.5 pp. HumanEval's one extra pass is not evidence of broad
quality improvement; absolute baseline scores were low (GSM8K 10.9%,
HumanEval 7.9%). On a 512-item GSM8K pool, independent three-restart medians
suggested 12.93% BCT gain, but its paired quality lower bound was -1.367 pp
and restart-time ranges overlapped. Neither the 512-item nor full task is a
statistically established ≤1-pp, ≥15% promotion point.

At ≤1 pp, static threshold **0.825** on full GSM8K had 69/512 on the
intermediate pool and **140/1,319** on the full pool. Its two-restart full-pool
BCT median was **208.812 s** versus the one-restart 0.9 baseline **227.782 s**
(descriptive 8.33%); paired full-pool score lower bound was **-0.986 pp**.
It is descriptively faster than the one-restart phase result **212.089 s**.
Static 0.8 lowered full-pool NFE 14.5% but had worse repeated-median BCT.
Thus NFE removal does not map monotonically to direct time and the tested
dynamic schedule does **not** dominate the strongest evaluated static frontier.
See [static frontier](01_static_frontier.md) and
[observed candidate/true-oracle boundary](03_dynamic_oracle.md).

## Mechanism and evidence boundary

The stock confidence-threshold decoder already chooses a variable number of
accepted positions `k_t`; the opt-in phase threshold changes that acceptance
budget without altering router, experts, ownership, EP backend, or model
weights. The resulting NFE change is a **generic dLLM** mechanism. A fresh
structural trace confirmed actual EP4 ownership (256 routed experts, 64 per
rank), remote DeepEP dispatch, owner execution, reverse combine, and top-k
conservation: for one layer-16 wave `M=992`, there were **7,936** expert pairs
and **5,857** remote assignments. But the timing observer increased n=32 BCT
about **57%** for baseline and **112%** for phase. Its dispatch/expert/combine
event timings cannot be mapped to clean E2E or interpreted as a clean
MoE-work saving. Per-action routed assignments, MoE time, accepted tokens per
forward, or physical EP payload savings were **not** measured at sufficiently
low observer overhead. No Epoch-like compacted-row saving is credited to the
current dense runtime. See [EP specificity](04_ep_specificity.md) and
[risk calibration](05_risk_calibration.md).

Seven small-screen phase schedules and the promoted policy were each rolled
out to final generation; independent one-step savings were never summed.
No exhaustive future-aware per-step O2, action-level EP-cost controller,
contextual bandit, RL, live EP-specific ablation, MATH/MBPP, online serving,
or post-Epoch runtime was run because the observed gate failed. The
`DYNAMIC_ORACLE_RESULTS.csv` evidence-type column explicitly marks actual
trajectory **candidates**, not a perfect upper bound. The
`EP_SPECIFICITY_ABLATIONS.csv` and `RISK_CALIBRATION.csv` have only headers
instead of invented data. Required quality-versus-routed-assignments/MoE-time
figures likewise cannot be plotted from a clean measurement.

## Answers to working-contract final questions

1. **Strongest static frontier:** on full GSM8K, 0.9 scored 144/1,319 at
   227.782 s once; 0.825 scored 140/1,319 at 208.812 s two-restart median
   and is paired-certified only within 1 pp. 0.8 scored 142/1,319 but its
   repeated-median time worsened. No complete seven-threshold full-set
   frontier was feasible; the seven-way static screen was n=32.
2. **Accepted tokens/forward tolerance:** not quantified cleanly. Threshold
   0.825 versus 0.9 reduced full GSM8K NFE **11.25%** while scoring four
   fewer items; that is not a measured TPF tolerance or ≤0.5-pp certificate.
3. **Early aggression:** the n=32 early-only 0.825 schedule increased NFE
   102→103 with 12/32 versus 13/32; this is a negative economic signal,
   not a statistically supported early semantic-risk conclusion.
4. **Middle value:** phase 0.9/0.8/0.9 reduced n=32 NFE 102→96 but lost one
   correct answer. Middle-plus-late 0.825 preserved that screen's 13/32
   score at NFE 95. The promoted benefit cannot isolate middle as uniquely
   highest-value.
5. **Late value:** late-only 0.825 had 12/32 and unchanged NFE 102. Neither
   universal semantic safety nor low marginal future cost was established.
6. **NFE:** promoted phase NFE fell 8.09% on full GSM8K and 8.78% on full
   HumanEval; static 0.825 fell 11.25% on full GSM8K.
7. **Iterations:** summed forward evaluations fell by those NFE amounts;
   distinct per-block iteration counts were not independently reported.
8. **Routed assignments:** fresh EP4 structural waves conserved `8*M`, but
   total clean full-trajectory routed-assignment reduction was not measured.
9. **Dispatch/expert/combine:** observer-heavy stage events exist, but the
   clean per-policy reductions are unresolved; no cost attribution claim.
10. **Exact-quality oracle:** no perfect oracle exists. The fastest tested
    *observed-score-nondecreasing* full GSM8K dynamic candidate gained
    6.89% versus 0.9, yet had seven baseline-correct paired losses and
    only one full-set timing restart.
11. **≤0.3 pp:** paired-certified *evaluated* choice is 0.9 (0% gain); the
    phase quality lower interval -0.455 pp misses this budget.
12. **≤0.5 pp:** paired-certified *evaluated* phase candidate gained 6.89%
    versus 0.9, below the 10% implementation gate.
13. **≤1.0 pp:** evaluated static 0.825 has a descriptive 8.33% two-restart
    median gain versus a one-restart baseline; it is faster than the one-run
    phase candidate. These are not perfect-oracle numbers.
14. **≤2.0 pp:** same fastest evaluated static candidate; no stronger
    schedule/threshold was validated at full sample size.
15. **Dynamic versus static:** not dominated at 1 pp. At 0.5 pp the phase
    candidate beats the paired-certified 0.9 control by 6.89%, but does not
    satisfy direct-E2E promotion.
16. **EP cost beyond confidence + live M:** not established. The promoted
    policy is phase/confidence only, and no matched-quality D/E versus A/B/C
    live ablation was performed.
17. **EP-specific incremental gain:** **unknown**, not zero; the required
    ≥3–5% direct-E2E evidence is absent.
18. **Simple cost–risk controller recovery:** not implemented because the
    observed gate failed and there is no defensible perfect future oracle
    to define a recovery percentage.
19. **Contextual bandit:** not justified or evaluated.
20. **Full RL:** not justified or evaluated.
21. **Learning Unmasking Policies:** a phase-threshold schedule is not a
    clearly independent method; unlike a future EP-aware action, it showed
    no measured physical-cost advantage. See [prior-art audit](10_prior_art.md).
22. **TEAM:** TEAM changes dLLM MoE execution/selection with model-specific
    caching/speculation; this PoC changed only the vanilla confidence budget.
    No TEAM head-to-head or additive adaptation was measured.
23. **Quality credibility:** paired bootstrap intervals and McNemar checks
    were computed on full official cohorts. They support the narrow
    non-inferiority statements above, not universal equality or improvement;
    one full-set timing restart per dynamic policy limits speed precision.
24. **Math and code:** GSM8K and HumanEval both showed descriptive BCT
    reductions, **6.89%** and **3.95%**, respectively; MATH/MBPP were not
    promoted, and low absolute scores limit generalization.
25. **Paper-level MoE-EP frontier:** **no**. There is a generic acceptance/
    NFE signal but no demonstrated EP-specific incremental frontier or
    dynamic-best-static superiority at a promoted quality budget.

## Decision and next step

Do **not** implement an EP-aware acceptance controller, bandit, or RL from
this result. The only scientifically useful continuation would be a
low-overhead, per-action clean MoE/EP cost trace followed by matched-quality
confidence+live-M versus confidence+physical-EP-cost controls on a stronger
quality-valid setup. Reopen the method gate only if direct E2E exceeds the
contract's quality-budget thresholds and physical EP features add at least
3–5% beyond generic dLLM features. GPU 4–7 were returned to the requested
task-owned `run_utilize.sh` burn after measurement; physical GPUs 0–3 were
not used or modified.
