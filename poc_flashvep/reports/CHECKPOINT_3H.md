# T+3h checkpoint — online runtime-regime discovery

The live run budget did not reach the nominal three-hour wall target because
the tested runtime repeatedly required model reloads and one long run stalled
after producing a usable partial trace. This shortfall is recorded explicitly;
GPU time is not padded with idle waits.

- Clean online rows: 628,032 rank/layer observations; 156,144 complete
  four-rank invocations across prefill and decode.
- Timing sanity is **INSUFFICIENT** (median grouped CV 29.8%, p90 71.9%).
- HT lower-envelope headroom is 0% median and 3.4% maximum for exact-M bins.
- Queue remains populated with generated H1R–H12R follow-ups for a future
  instrumentation-corrected sprint.

## Late completion update

Four additional hook-enabled HT runs completed after this checkpoint (SMS
20/12/8 at concurrency 8 and SMS20 at concurrency 2).  The final primary
route-joined aggregate contains 6,105,264 valid rank rows and 1,524,288
complete invocations.  Exact-M SMS lower-envelope headroom is 0.11% median,
1.71% p90, and 2.10% maximum in prefill; decode is 0.25%.  The compact
time-block model remains a null for fanout geometry (+0.001% RMSE change).
