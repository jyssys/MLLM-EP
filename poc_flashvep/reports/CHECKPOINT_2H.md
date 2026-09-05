# T+2h checkpoint — online runtime-regime discovery

The planned multi-hour window was compressed by repeated engine start-up and
one stalled DP run; all available GPU time was used for bounded online traces
and the remaining time for automated analysis. No adaptive runtime method was
implemented.

- Strongest positive: clean HT SMS configurations have only small,
  shape-specific differences; route-joined exact-M lower-envelope maximum is
  2.10%.
- Strongest negative: adding fanout/sender geometry to the distribution+rank
  model changes held-out RMSE by **+0.001%** (effectively zero).
- Open anomaly: fixed M/layer prefill tails remain, but route-corrected
  rank/timing correlation is only ≈0.14; rank-local timing is too noisy for
  stage causality.
- Backend status: HT is verified; LL fails at DeepEP's `nvshmem_qp_depth`
  assertion for MBT=8192 and 1024.

## Late live-run update (T+2h)

The hook-enabled recovery runs subsequently completed with over one million
rank/layer records each.  The final route-joined aggregate now includes
6,105,264 rank rows and 1,524,288 complete invocations; no-row SMS4/SMS16
attempts remain diagnostics only.
Two additional SMS4/SMS16 launches were terminated after their hook variable
was omitted and produced no measurement rows; they are retained as launch
diagnostics, not runtime evidence.  The primary aggregate will be regenerated
after the recovery run closes, with this run included and all no-row attempts
excluded.
