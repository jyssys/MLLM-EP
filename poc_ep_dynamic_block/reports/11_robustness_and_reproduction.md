# Robustness and reproduction

## Controls that were rerun

- The strongest headline fixed points use three independent launches.  GSM8K
  B32/mini32/t0.9 has median 9.453 s and score 13/32.  HumanEval
  B32/mini16/t0.85 has median 13.881 s and score 18/32; its B8 higher-quality
  point has median 20.167 s and score 19/32.
- Dynamic schedules were run in their forward and reversed orders, with fresh
  process launches.  Adding a global schedule barrier fixed neither their NFE
  inflation nor their latency deficit.
- A no-early-stop, threshold-1.0 endpoint control still produced different NFE
  across B.  On GSM8K the B8--B128 NFE range was 224--185; on HumanEval it was
  281--255.  This rules out early stopping as the sole decoder-effort confound.
- Fixed-B schedules produced exactly the same answer strings and generation
  lengths as the ordinary fixed-B path for 32/32 samples at both B32 and B64.
  The schedule implementation is therefore a valid decoder anchor; the mixed
  trajectories are genuinely different trajectories, not a fixed-path port
  failure.

## Longer generation

At target length 384 on GSM8K, fixed B32 and B128 took 14.128 s and 13.756 s,
whereas the fastest tested mixed schedule took 21.648 s and also lost bounded
quality.  At target length 512 on HumanEval, fixed B128 took 10.876 s, while the
best quality-matched mixed schedule took 19.131 s.  A separate B16/B32 control
gave the same direction: both mixed orders were much slower than either fixed
endpoint.  The negative result is not a short-generation artifact.

## Observer tax

Physical EP component timing is observer-heavy and is never used as clean BCT
evidence.  Across the comparable n=8 traces, median wall-time tax is about
50.1% for GSM8K and 37.3% for HumanEval, with larger outliers at B128.  Clean
latency, score, and NFE come from uninstrumented runs; trace rows are used only
for shape and component relationships.

## Failure diagnosis and repair history

The stock config-42 path silently forced B32, and the first variable-B attempt
also inherited B32-aligned cache buckets.  Both issues were repaired and fixed
B8/16/32/64/128 were validated.  A first mixed-schedule barrier deadlocked
because already-waiting sequences were included in a cache-boundary predicate;
the predicate was corrected and the barrier control completed.  HumanEval
mini32 exceeded the DeepEP source-row/HBM envelope and is reported as a
capacity limit rather than a method failure.

Every launch, including failed setup attempts, is retained in
`ATTEMPT_LOG.csv`; successful four-GPU wall time and failure status are retained
in `GPU_TIME_LOG.csv`.
