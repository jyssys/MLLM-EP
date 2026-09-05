# Anomalies tracked during the execution-regime sprint

| ID | Observation | Follow-up | Status |
|---|---|---|---|
| ANOMALY_01 | The earlier block-ordered A2/A32 comparison changed sign when the first case changed. | Global + per-shape warmup, shuffled interleaving, 10–30 paired repetitions. | The original sign flip is not reproduced; residual state variance remains in block runs. |
| ANOMALY_02 | F4/F1 changes from near-zero at M=128 to a positive penalty at M≥448, reaching +31% expert at M=1024. | H1 boundary sweep and 30-repetition M=1024 replication. | Reproducible high-M regime; not a 128/512 power-of-two discontinuity. |
| ANOMALY_03 | Dispatch and expert often increase together while combine decreases. | H3 local expert-only replay and phase decomposition. | Supports expert/packing interaction with phase cancellation. |
| ANOMALY_04 | Same fanout and aggregate rank load but different balanced destination-pair geometry is nearly null. | H6 M=512 and M=1024 paired controls. | Traffic geometry alone is not sufficient in tested shapes. |
| ANOMALY_05 | A layer-44 real-route M=512 sample (`method`) produced a 4.76 ms combine outlier. | Repeated real-route layer transfer; preserve as variance evidence. | Not used as a causal fanout claim; real routes lack matched F1/F4 controls. |
| ANOMALY_06 | H5 late-layer M=1024 retry failed during engine initialization. | Preserve failed attempt and rely on successful layer-4/24/44 M=512 plus layer-24 M=1024. | Layer-44 M=1024 is BLOCKED, not silently imputed. |

All successful replay cases reported route/token identity and correctness true.
