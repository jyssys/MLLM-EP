# Feasible oracle accounting

`ORACLE-0` is the prior perfect zero-cost wave projection: 16.11% at matched
p50 and 23.83% at optimistic p25. It has no request join and is not a direct
E2E result.

The new live run supplies a stricter request-level upper bound. Existing
downstream event-wait proxy time is 13.2% of logical MoE span but omits the
internal DeepEP notify/barrier wait; it cannot be treated as hideable debt.
Using the only non-overcounting direct join (cap assigned excess by each
request E2E) gives 1.092% aggregate. This bounds `ORACLE-1` (existing slack),
while `ORACLE-2` communication priority and `ORACLE-3` backpressure have no
safe measured implementation and therefore no positive direct evidence.

The prior repeated ALWAYS_SYNC, selective drain, and simple previous-dispatch
controls were unstable and throughput-costly. The best realistic oracle is
therefore below the required 18% gate; no method prototype was justified.
