# Batch-regime sensitivity

Evidence: **MEASURED_TRACE_STRUCTURAL_REPLAY_NO_LATENCY**

The common frozen route pool was replayed without modifying routes. The known
idle-DP dummy row was removed before aggregation.

| Global batch | Requests used | Mean active-expert union | P50 | P90 | Paths per active expert | Status |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 20 | 8.000 | 8 | 8 | 1.000 | complete |
| 2 | 20 | 14.376 | 15 | 16 | 1.134 | complete |
| 4 | 20 | 24.290 | 25 | 29 | 1.369 | complete |
| 8 | 16 | 39.209 | 41 | 48 | 1.742 | complete; 4 omitted |
| 16 | 16 | 56.839 | 59 | 71 | 2.468 | complete; 4 omitted |
| 32 | 0 | N/A | N/A | N/A | N/A | insufficient 20-request pool |

Larger waves increase expert-frequency concentration and potential grouping,
but these baseline unions do not establish safe consolidation. Track A's
measured output oracle, not this structural table, controls promotion. Track B
is intra-request; batch size changes working-set/cache pressure but not the
definition of temporal reuse.

The reported distinct-expert weight-byte quantity is a working-set proxy, not
measured HBM traffic. No latency is inferred from this table.
