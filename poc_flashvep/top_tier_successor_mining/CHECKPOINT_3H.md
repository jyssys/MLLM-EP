# Checkpoint at resumed active work — 2026-09-07 17:28 KST

Elapsed since original start: approximately 3 h 28 min, including status-only
intervals while the calibration subprocess continued. Not all elapsed time was
active agent analysis. No final verdict or paper candidate ranking.

## Evidence that survived

- Exploratory SERE ChartQA loss localizes to decode; batch16 largely removes it.
  This makes batch/threshold and natural-termination controls essential. Official
  BF16-norm calibration is now available for confirmation. No direct E2E gain yet.
- Supplied Libra code, not the fallback planner, executes on four GPUs with real
  reduced-layer Qwen weights. 24/24 compared first greedy tokens match, but logits
  are not bit-exact. Full-layer correctness and MLLM bridge remain unvalidated.
- Full MoDES importance calibration completed. Target70 frontier has actual
  skipping71.3284% and KL.0110761; target85 is still running.

## Important negative / confound

Lower Vision lookahead recall does not imply a useful Libra successor: the
actual planner's perfect-prediction locality gain is small. Pilot OCR loss is
already known from MoDES and adjacent work. Neither is a novelty finding.

## Next work

1. Finish official frontier; evaluate fresh held-out quality with these policies.
2. Capture exact VL embeddings/MRoPE/DeepStack and test native Libra full layers.
3. Measure SERE and MoDES with clean, direct-request EP serving comparisons;
   separate natural EOS from fixed output and diagnostics from performance.
4. Complete all-three headroom/trivial-fix screens before choosing one deep dive.

GPU accounting: use `GPU_TIME_LOG.csv` plus ongoing run timestamps and telemetry;
do not count idle time/burn or overlapping worker intervals twice.
