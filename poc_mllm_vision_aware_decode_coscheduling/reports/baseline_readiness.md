# Baseline readiness

Status: **READY_REUSED**. The frozen Part-1 quality and physical TRUE EP4 evidence were reused after its recorded integrity audit; the full MMStar/ChartQA evaluation was not repeated.

- Model revision: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- Model contract: BF16, 48 sparse MoE layers, 128 experts, top-k 8
- Physical contract: GPUs 4,5,6,7; TP2-DP2-EP4-PP1; DeepEP; 32 contiguous experts/rank
- Frozen bundle SHA256: `cdf380d9d1b790ad97da3065d0d913d542ab77ca8a8cc7e8cbdb2ea327cf66d3`
- Asset integrity: `True` (30 digests checked; 0 failures)
- Authoritative spec SHA256: `6304fec817493890687f2cbd7c0f51adcad0a5e610d49c4b0dae60e8e9613f07`
- Prior long-reasoning evidence: reused compact routing traces only; no simulator latency and no new optimization were used.

The baseline audit records `QUALITY_BASELINE_READY=true` and `ACTUAL_EP4_READY=true`. Stage A is an offline replay over measured TRUE-EP4 routing assignments; it does not claim a new physical timing measurement.
