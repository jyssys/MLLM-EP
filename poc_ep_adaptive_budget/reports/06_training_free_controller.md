# 06 — Training-free controller gate

No live EP-cost-aware controller was implemented. The clean phase-only
schedule yielded -6.89% BCT at paired-certified ≤0.5 pp on GSM8K full and
-3.95% on HumanEval full. The spec's meaningful implementation point
`≤0.5 pp + ≥10% direct E2E` was not reached on either promoted task. It also
did not outperform the descriptive static 0.825 frontier at a 1.0-pp budget.
Most importantly, no EP feature had incremental quality–latency value beyond
confidence/ready-pool state demonstrated. Under those conditions a controller
would be an implementation of generic adaptive unmasking, not a validated EP
system method.

Future design, **not implemented or measured**: maximize predicted future
distributed MoE cost saved subject to a calibrated semantic-risk budget;
compare confidence-only, confidence+progress, confidence+physical M,
confidence+MoE, and confidence+full EP at matched quality. The quality-risk
model and D/E policy frontier are absent, so there is no honest oracle-capture
percentage for an explicit controller. [POLICY_SWEEP.csv](../POLICY_SWEEP.csv)
contains only actual phase-candidate trajectories, not a live controller.
