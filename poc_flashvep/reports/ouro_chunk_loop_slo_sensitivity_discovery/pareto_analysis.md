# Pareto and joint-selection interpretation

The descriptive frontier minimizes long-request TTFT and background cohort-p99 ITL.
`pareto_frontier.csv` states that exact objective.
FigureC displays fixed-policy points.
It does not convert point dominance into statistical dominance.

The complete grid includes Atomic C-only, Split G1, Grouped G2, and stock Mixed.
Staged G4 is an additional matched-path control.
Shortlisted cells have ten cohorts, while other grid cells retain three.
The paired tables compare common blocks and trials.
Unmatched mean differences are not a substitute for those intervals.

G2 C1024 occupies an intermediate TTFT–tail region in the new8K/N8 surface.
Atomic C1024 has lower mean TTFT and worse p99.
Mixed C512 has higher mean TTFT and better p99.
Split C1024 is another relevant full-grid point.
An intermediate point is not evidence that a two-dimensional adaptive policy beats every one-dimensional policy.

The held-out serving shortlist is a frozen representative menu.
It is not an exhaustive per-SLO optimization over all grid points.
In particular, Split C1024 and Mixed C1024 have discovery data but are outside that serving menu.
This limitation prevents a claim of globally best fixed serving capacity.

The observed best-policy SLO panel is post-hoc sensitivity analysis.
It selects the best measured fixed point for a displayed cell.
It is not a causal controller that knows future arrivals or output lengths.
It does not demonstrate an online scheduling gain.

A joint-adaptive claim needs stronger controls.

1. A calibrated adaptive C-only policy must receive the same deadlines and processing budget.
2. An adaptive G-only policy must use the same observable state and decision frequency.
3. Stock Mixed and the best validated fixed controls must remain available.
4. All policies must meet the same completion, queue, quality, and sample-size requirements.

Those online comparisons are not implemented in this study.
Their absence is not proof that joint adaptation cannot help.
It is a reason not to claim that it does help.
Unresolved batched-output differences are another independent gate.
