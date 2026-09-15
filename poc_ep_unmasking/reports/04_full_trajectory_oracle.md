# 04 — Counterfactual full trajectory

O0 stock trajectories were measured cleanly: GSM8K 8 had 106 NFE,
7.4620 s BCT, and 5/8 bounded exact-answer score; HumanEval 8 had 55 NFE,
5.7934 s BCT, and 2/8 bounded functional score.

O1 near-tie future-aware, O2 quality-constrained, and O3 speed/quality Pareto
counterfactual trajectories were **not run**. The strict near-tie freedom was
only 10.7% of valid decisions across the GSM8K complete and HumanEval partial
traces. A frozen-route one-swap rank-load screen yielded a combined 0.228%
mean change at delta 0.02 and zero median. It does not justify launching a
large full-rollout oracle, particularly when matched transfer count preserves
total live rows and expert assignments.

No actual alternative final answer, quality-safe request E2E oracle, or
post-compaction latency is measured. The status is an **early economic/semantic
gate NO-GO for this operating point**, not proof that every unmasking policy
is impossible. `FULL_TRAJECTORY_RESULTS.csv` records O0 only and explicitly
marks O1–O3 as not run.
