# End-to-end integration

`NOT IMPLEMENTED — ORACLE GATE FAILED.`

The measured request baseline remains unchanged. There are three distinct
numbers that must not be conflated:

1. **Measured owner-local operator:** PyTorch grouped is 20.52--22.08% faster
   than the installed production fused expert on selected dense real shapes.
2. **Projected existing-backend replacement:** 6.06--6.51% request E2E, before
   integration/packing effects; not measured full-model speedup.
3. **Novel RefineGEMM headroom over the strongest whole kernel:** ideal
   2.85--3.31%, credible 1.43--1.66% request E2E; post-compaction credible
   0.57--1.06%.

Only item 3 is the promotion metric, and it fails even the 5% gate. Figure
[15_request_level_refinegemm_oracle.png](../figures/15_request_level_refinegemm_oracle.png)
visualizes the separation. [E2E_RESULTS.csv](../E2E_RESULTS.csv) contains only
measured baseline rows; projections remain in
[REFINEGEMM_ORACLES.csv](../REFINEGEMM_ORACLES.csv).

