# Experiment log

1. Created branch `flashvep/final-ep-tail-go-nogo` from the existing fixed-
   shape economic-gate work.
2. Terminated only stale user-owned processes on physical GPUs 1--4; GPUs
   0/5/6/7 were not touched.
3. Fixed the local measurement hook's misplaced `_layer_id` body; no model,
   routing, placement, or scheduler semantics changed.
4. `direct_stock_run1`: hook import failed with `NameError`; no invocation
   artifact used.
5. `direct_stock_run2`: 49,536 raw rows / 24,768 logical; live HT sanity run.
6. Added child-engine context-file handoff for wave labels and scheduler
   request IDs.
7. `direct_stock_run3`: 37,248 raw / 18,624 logical, 2,304 contextual;
   context handoff verified.
8. Low-latency bounded attempt: failed at unmodified NVSHMEM queue-depth
   assertion; no unsafe workaround.
9. `direct_stock_run4`: 86,400 raw / 43,200 total logical, 13,824 contextual
   raw / 6,912 contextual logical across 12 waves; dispatch giant tails and
   request metrics captured.
10. Ran `final_ep_tail_go_nogo/analyze_final_gate.py`; direct upper bound
    1.092%, with-max / without-max / without-top5 sensitivity 1.092% / 0.615%
    / 0.377%.

All timing uses same-device CUDA events. Raw runs are retained in this result
root; only explicitly selected code and summary artifacts are committed.
