# Experiment log

* 2026-09-06: created `flashvep/deepep-wait-aware-tail-poc` from
  `972dbda147631faa10f7dfa1c39ededcbd66f30e`.
* Added read-only DeepEP stage wrappers and a wait observer.  Verified that
  `EventHandle` exposes only `current_stream_wait()` and no query operation.
* `stock_run1`: 67,968 invocations; decode dispatch p99 1.84 ms, max 2.06 s.
  All 67,968 records had `previous_event_present=false`; event-wait proxy
  p99 1.03 ms.  Decode >20 ms dispatch events: 64.
* `always_sync_run1/2/3`: global synchronization diagnostic; used only as an
  upper-bound/overhead control.  It is not a production policy.
* `comm_drain_run1/2/3`: relevant DeepEP communication-stream drain before
  dispatch.  Run 1 reduced >20 ms events (64→26), but runs 2/3 were unstable
  and had larger p50/tail values; this rejects a robust unconditional drain.
* `oracle_selective_run1`: offline invocation-id selection from a prior run;
  only 72 local rows were selected and local IDs are not a perfect cross-run
  identity.  It is labeled an oracle diagnostic, not an online predictor.
* `oracle_selective_waittail_run1`: 57 IDs selected from prior wait-proxy and
  dispatch tails; selective drain incurred substantial run-level overhead.
* `online_simple_run1/2`: threshold policy (`previous dispatch >=1 ms`),
  selected 773 and 489 local invocations respectively; p99 dispatch improved
  in these short runs but extreme tails remained and no paired E2E TTFT signal
  was available.
