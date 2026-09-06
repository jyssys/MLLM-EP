# Checkpoint H2 (~1.5 hours)

- New replication anomaly: fixed text c8 rep2 had T_MoE p50 2.056 ms vs 1.168 ms in c8, with dispatch share 41.1% and event-wait share 16.1%; same-M comparisons remain confounded by a different observed M=456 bin.
- Fresh high-resolution vision c8 shifts normal expert share to 42.2% (text c8 31.0%) but only +4.3% T_MoE p50.
- No candidate yet passes the direct `>=15%` E2E oracle gate with causality and generality.
- Active work: c8 rep3 fixed text; after it completes, compare same-M bins and treat any cross-run drift as a hardware/runtime-state anomaly requiring telemetry, not a method claim.
- Queue remains above eight pending nodes (see ACTIVE_FRONTIER.md).
