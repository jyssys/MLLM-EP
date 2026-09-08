# Two-hour checkpoint — 2026-09-07 16:00 KST

No winner selected. All three remain in baseline screening.

- SERE: fresh ChartQA B1 loss is substantial, but B16 largely recovers quality.
  Decode-only application does not recover B1 quality. Natural-EOS generation
  sometimes lengthens; fixed-token throughput is therefore insufficient to claim
  useful request speed. These are hypotheses, not a diagnosed MLLM failure.
- MoDES: full official 1,024-item, 48-layer/modality calibration finished. The
  unchanged 100-grid frontier search is running on four quality replicas. Resumed
  evaluations match cached KL/skip statistics exactly. These replicas are NOT EP
  throughput measurements. Pilot ChartQA loss overlaps the original paper's
  acknowledged tradeoff and is not a novel failure by itself.
- Libra: actual lookahead recall is within the published general range. The
  official repository remains unavailable. The literal Appendix B planner has
  been separated from a more optimistic ambiguous interpretation. Real-weight
  replication/local-remote execution diagnostics are next, not count-to-latency
  extrapolation.
- Live EP sanity: TP2/DP2/EP4 DeepEP HT verified on all four ranks. Vanilla versus
  MoDES no-op generated tokens matched 6/6; zero-weight versus sentinel skip 6/6.
  An isolated DeepEP identity-expert test passed 288 rank cases including SERE
  duplicate IDs and MoDES -1 sentinels. Failed earlier ports remain labeled as
  instrumentation/implementation failures, not paper-method failures.
- Prior-art attack: XShare directly covers batch-aware expert selection including
  heterogeneous batches and EP. PROBE covers learned next-gate correction and
  hiding-window-aware replication with split-phase transfers. These block obvious
  successor stories unless a distinct material cause is demonstrated.

Next: finish MoDES calibrated frontier; confirm SERE with official BF16-norm
similarity; run clean request-level EP comparisons and bounded Libra cost probe.
