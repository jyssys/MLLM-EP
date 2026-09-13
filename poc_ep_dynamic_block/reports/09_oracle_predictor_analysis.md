# Trigger and predictor analysis

The prerequisite target does not exist at useful magnitude. In the isolated
future-aware request tournament, the median safe mixed-schedule gain is 1.014%
for GSM8K and 0.369% for HumanEval. In the batched n=32 runs, every mixed
trajectory is slower than the global quality-matched fixed frontier.

Consequently no learned controller is fit and no accuracy number is reported
for predicting an oracle label that is mostly noise.

The final exhaustive actual-trajectory search strengthens this gate: the best
mixed schedule is 29.22% slower than fixed on GSM8K and 51.10% slower on
HumanEval at equal bounded score. There is no positive batched target for an
EP-aware predictor.

## What the observable features do explain

Physical M, rows/expert, tiny-expert fraction, and dispatch bytes track the
per-wave cost regime. Fanout and remote fraction add little because both are
nearly constant across B. These features do not determine semantic NFE or
quality, which are what change the request-level winner.

## Semantic-only versus EP-only versus joint

- Semantic-only dynamic sizing is directly occupied by AdaBlock/DSB and was
  not rebranded here.
- EP-only features can choose an efficient physical wave, but best-per-B mini
  calibration already recovers that static effect.
- A joint policy has no credible residual oracle to recover.

The correct result is NOT TRIGGERED, not a zero-accuracy controller.
