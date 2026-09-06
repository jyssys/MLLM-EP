# Cross-experiment meta-analysis

## Consistent facts across prior campaigns

1. Normal per-layer T_MoE is roughly 1.1–1.4 ms in common Qwen3-VL EP4
   regimes, while request p50 spans roughly 0.3–1.3 s.  A paper candidate must
   therefore affect repeated whole-step/queue mass, not rare individual MoE
   spikes.
2. Dispatch/expert shares shift substantially with concurrency and scheduler
   token budget, but static configuration changes and throughput trade-offs
   explain the large reported differences.
3. Routing histogram, rank load, and fanout do not predict held-out online
   latency well.  The unexplained variance is likely timing/state/coordination
   rather than another static routing scalar.
4. Large tails localize to dispatch, but direct request joining reduced their
   aggregate removable share to 1.09%.  Tail magnitude is not economic mass.
5. One text→vision transition doubled MoE cost, yet matched warmup and a
   telemetry replication removed it.  Clock/observer/runtime state must be
   first-class controls.
6. DBO doubles operator invocations in the tested fixed request and inflates
   latency, while DBO is disabled in the validated baseline.  A new direction
   cannot rely on that known broken regime.

## Cross-report gap used by this sprint

All detailed online stage reports inherited per-layer CUDA synchronization and
GPU→CPU route capture.  This is inconsistent with the source's deliberate
asynchronous HT interface.  The first fresh experiment therefore measures the
observer, after which the sprint will mine request-critical mass with a
deferred-event observer.

