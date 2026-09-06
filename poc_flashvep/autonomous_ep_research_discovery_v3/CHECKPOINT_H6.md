# Checkpoint H6 — 2026-09-06 20:50 KST

- Wall time: ~3 h 30 min from the first fresh run; search tree has 16
  measured/diagnostic nodes and 10 documented pending siblings.
- Strongest positive remains C1c (one 2× T_MoE multimodal transition), but a
  fresh telemetry replication is normal. It fails reproducibility and the
  trivial-engineering gate.
- Strongest robust negative: fanout, rank-load, and residual features add no
  held-out latency information; fixed-tail request mass remains 1.09%.
- Backend probe: deepep_low_latency is unsupported in this local vLLM/DeepEP
  configuration because its NVSHMEM QP-depth assertion fails before serving.
- Remaining action: consolidate reports/artifacts and close all candidates.
  No untested branch has a direct >=15% oracle without re-opening a closed
  direction, so this checkpoint authorizes final `SEARCH_SPACE_EXHAUSTED_NO_GO`.
