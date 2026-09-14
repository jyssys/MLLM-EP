# End-to-end integration

Status: `NOT_RUN_GATE_BELOW_12_PERCENT`.

No LLaDA2 runtime integration was attempted because there is no third path to
integrate. `E2E_RESULTS.csv` separates:

- measured current best-static request anchors;
- post-compaction analytical existing-path results;
- the analytical O3 target;
- explicit prototype status.

The projected 23.63%/24.28% total reduction from current dense execution to O3
must not be interpreted as RefineEP. Hypothetical liveness compaction plus an
existing LL path accounts for almost all of it. The new-kernel increment over
the strongest post-compaction existing path is 2.29%/2.10%.

Because no live exact prototype exists, there is no measured BCT, throughput,
HBM, NFE, or quality gain attributable to RefineEP.
