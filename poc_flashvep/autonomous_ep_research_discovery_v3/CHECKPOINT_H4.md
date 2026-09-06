# Checkpoint H4 (final)

- All high-mass fresh probes are complete: concurrency (c4/c8/c16), token budget (MBT 8192/4096), text/vision composition, mixed low/high load, and long prefill.
- Strongest controlled effect: MBT4096 doubles layer-local T_MoE at matched M and raises request p50 42%, but it is a one-knob static configuration effect with no throughput gain; it fails the non-triviality gate.
- Residual mining (90,240 rows) finds only high-M/high-active-expert enrichment; fanout/rank geometry remains null.
- Final decision: `SEARCH_SPACE_EXHAUSTED_NO_GO`; no candidate satisfies direct >=15% oracle + causal + generality.
