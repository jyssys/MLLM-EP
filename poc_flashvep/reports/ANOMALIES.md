# Runtime-discovery anomalies

| ID | Observation | Bounded interpretation | Follow-up/status |
|---|---|---|---|
| ANOMALY-01 | The vLLM SM knob is accepted for HT and clamps values above 20; HT traces at 8/12/20 have similar p50 but noisy tails. | Any adaptive-SM claim requires matched-route runs; current online waves are descriptive. | Active: aggregate per-regime envelope. |
| ANOMALY-02 | DeepEP LL initialization fails at the normal 8192-token budget (and even during the engine dummy profile at 1024) with `nvshmem_qp_depth` assertion. | LL is not a drop-in alternative for this validated MLLM serving envelope; this is a runtime feasibility boundary. | Record as blocked backend condition. |
| ANOMALY-03 | Same M/layer/rank online records show large tails while fanout/rank features vary little. | Candidate state/queue/first-use effect, not a new routing feature. | Active tail decomposition. |
| ANOMALY-04 | Route-id joining preserves fixed-shape tails, but rank-imbalance correlation falls to ≈0.14 (prefill and decode); timestamp-window aggregation had reported much larger values. | Cross-device timestamp joins were a confound.  The remaining tails are scheduler/stream-state candidates, not proven EP imbalance. | Keep as instrumentation-corrected follow-up; no method. |
| ANOMALY-05 | Two hook-missing SMS4/SMS16 launches started engines but produced no rows; LL attempts failed before requests at the same NVSHMEM queue-depth assertion. | Launch/hook setup and backend feasibility are separate from natural runtime evidence. | Excluded from primary aggregate; retained as diagnostics. |
