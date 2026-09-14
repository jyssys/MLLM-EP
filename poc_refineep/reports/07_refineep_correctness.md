# Correctness validation

## Existing-path replay

For every policy/repeat/rank, the benchmark records source and received
token-expert assignment counts. Across 25 real cases, 15 controlled cases,
five independent restarts, and 30 measured repeats per restart:

- global received assignment count equals global source assignment count;
- normal route handles preserve token identity and expert ownership;
- low-latency BF16 combine has maximum relative L2 `0.000896853` against
  normal, below 0.1%;
- normal/cached-normal comparison is numerically exact in the measured
  semantics;
- the error is consistent with BF16 reduction ordering, not missing routes.

The replay uses uniform router weights and an identity-like weighted expert
operation to isolate communication reconstruction. It validates dispatch and
combine semantics, not an unimplemented RefineEP kernel.

## RefineEP status

No third path exists because O3 failed the implementation gate. Therefore it
would be false to claim RefineEP token/expert correctness, full trajectory
equivalence, GSM8K/HumanEval equality, or NFE equality. Those checks are marked
`NOT_RUN_GATE_BELOW_12_PERCENT` in the machine-readable artifacts.

The inherited normal-path baseline retains deterministic bounded-task outputs
and NFE across topology-identical runs; this only establishes the substrate
anchor.
