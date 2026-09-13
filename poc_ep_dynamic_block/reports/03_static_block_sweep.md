# Static block-size sweep

The static campaign covers B=8/16/32/64/128, submitted batches 8 and 32,
multiple mini sizes, exact terminal extents, and threshold controls. Clean
request timing is in STATIC_BLOCK_SWEEP.csv; observer-heavy stage timing is
kept separate.

## Main clean result

At n=32 and threshold 0.9:

- GSM8K's fastest measured point is B32/mini32: 9.453 s median over three
  restarts, 13/32.
- GSM8K B64/mini16 raises the bounded score to 14/32 at 11.725 s median over
  three restarts.
- HumanEval B64/mini16 is 14.749 s and 17/32; after threshold calibration,
  B32/mini16 at threshold 0.85 is 13.881 s median and 18/32.
- HumanEval's highest observed bounded score is 19/32 at B8/mini16,
  threshold 0.85, but costs 20.167 s median.

There is no universally dominant fixed B. The task-level quality-latency
frontier changes with quality target, but this alone does not imply a useful
within-request dynamic schedule.

## Capacity

HumanEval B32/mini32 exceeded the feasible HBM envelope; it is recorded as a
capacity failure and was not repeatedly relaunched. Legal configurations obey
DeepEP's 1024 source-row/rank cap.
