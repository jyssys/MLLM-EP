# EP2 / EP4 anchor manifest

- EP2: physical GPUs 4,5; TP1/DP2/SP1/PP1/EP2; DeepEP; 64 experts/rank.
- EP4: physical GPUs 4,5,6,7; TP2/DP2/SP1/PP1/EP4; DeepEP; 32 experts/rank.
- Fit/development/final-test request counts: 32 / 64 / 16 (16 / 32 / 8
  two-request workload groups). Earlier test cohorts were promoted to
  development after each repair; the final v4 16 are new image-disjoint
  requests not used for model choice.
- Each final timing anchor uses two warm-ups and nine measured repeats. The predictor
  target is the repeat median, not a contended single sample.
- Structural EP2 trace remap versus actual EP4: assignment relative-L1 median
  1.603%,
  P90 2.619%;
  unique-row median 1.051%,
  P90 1.675%.
