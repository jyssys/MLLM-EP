# Expert-row trace

The trace contains 4,650 request-wave-layer cases and all 256 routed-expert
counts for each case. Both the measured dense runtime and the exact-route,
future-known live-row sensitivity contain 1,190,400 expert rows.

For every compacted case the following invariants pass:

- `sum_e M_e == 8 * fresh_M`;
- each expert maps to exactly one of four owners by contiguous 64-expert ranges;
- the four local vectors sum to the global vector;
- expert IDs and top-k assignments are unchanged; only non-live source rows are
  filtered in the sensitivity corpus.

[EXPERT_ROW_TRACE.parquet](../EXPERT_ROW_TRACE.parquet) is the compacted
sensitivity corpus; [DENSE_EXPERT_ROW_TRACE.parquet](../DENSE_EXPERT_ROW_TRACE.parquet)
is the actual dense physical runtime. The corresponding summaries explicitly
carry `worklist_scope`, so measured and counterfactual rows cannot be silently
mixed.

