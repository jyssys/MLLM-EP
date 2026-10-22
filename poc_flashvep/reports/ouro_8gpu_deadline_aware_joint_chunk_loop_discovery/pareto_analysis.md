# Fixed-policy operating points

`figures/diagnostic_pareto.png` plots all measured C choices at P8177/N8.
The axes are mean long-request TTFT and pooled observed ITL p99.
Every point aggregates twelve event-lane cohorts from four process starts.
These are diagnostic operating points, not qualified SLO frontiers.

Smaller G can reduce long individual quanta while increasing interruptions before one output completes.
Smaller C can reduce prompt rows but sharply increase submission count and TTFT.
Larger C can improve prompt efficiency while increasing exposed decode gaps.
Staging and Mixed packing alter these trade-offs independently of G.

Staged G4 is mandatory for attribution.
At C1024, its paired mean cohort-p99 is 28.87% lower than Atomic.
The additional G2-versus-staged-G4 change is uncertain: +5.25%, 95% CI [−24.17%, 34.67%].
This comparison does not isolate G2 as the sole cause of Atomic-vs-G2 tail improvement.

No adaptive Pareto frontier is measured.
The execution-budget envelope suggests partial-work choices but omits queue-level rollout.
It cannot replace a causal A1/A2/A3 comparison or tuned Mixed.
