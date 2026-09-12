# Final Decision

## NO-GO

The required causal chain fails at its last two links:

1. **Refinement state changes:** yes. Mini32 decision-live median falls from 90.4%
   early to 18.1% late.
2. **Physical sparse shape changes:** partly. Aggregate M and expert-row density fall,
   primarily because the ready pool drains; each scheduled request still executes
   32 physical rows, while remote fraction/fanout remain almost constant.
3. **Optimal mini size changes:** not systematically under a valid ready-set control.
   Mini32 wins/ties on 64/65 waves.
4. **Dynamic beats best static:** only **0.650%** in a zero-overhead perfect oracle.

The result is below the specification's 3% NO-GO threshold. The 12.905% naive
phase oracle is rejected because it compares different ready-pool populations.
The hypothetical compacted-runtime oracle is also only 0.224%.

No live controller, TP4 control, or deep prior-art audit was triggered. More RAWS
engineering cannot recover a 5--8% result from a 0.650% perfect upper bound on this
workload and substrate.
