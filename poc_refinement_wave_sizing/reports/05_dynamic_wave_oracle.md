# Perfect Dynamic Oracle

## Invalid positive that is excluded

A phase-average lookup reports 12.905% over best static by selecting mini32 for
early/middle and mini16 for late. This violates the ready-set constraint described
in the replay report and is excluded from all conclusions.

## Primary feasibility-aware oracle

For every canonical mini32 wave, the exact ready request set was retained and split
into chunks of size 1/2/4/8/16/32. Ties retain mini32. Only wave 37 showed a strict
benefit: an early 32-request/M1024 wave selected mini16 and saved an estimated
40.781 ms (19.18% locally). The other 64/65 waves retained mini32.

| policy | normalized request cost | gain vs best static | valid? |
|---|---:|---:|---|
| best static mini32 | 6273.443 ms | 0% | yes, clean median anchor |
| naive phase lookup | 5463.879 ms | 12.905% | **no; population mismatch** |
| perfect feasible per-wave dynamic | 6232.662 ms | **0.650%** | yes, primary oracle |

This oracle is stronger than a realizable controller: it knows the future cost of
every feasible partition, charges no observation/switching overhead, and treats
all clean BCT as removable. It still fails the 3% NO-GO gate by 4.6×.

Evidence: [`DYNAMIC_ORACLE.csv`](../DYNAMIC_ORACLE.csv) and
[`PER_WAVE_ORACLE.csv`](../PER_WAVE_ORACLE.csv).
