# IFM MoE Huginn — model-specific result

Verdict: **NO_GO_HEADROOM for the tested native B1 grid**.
P0 status: CONDITIONAL_BACKEND_FIDELITY.
P1 completed. P2 completed with stated measurement limits. P3 did not run.

Checkpoint: `IFM/LoopedLM-P1-moe-huginn-500b` at `f1321d5272a11348d5093ad7ab1d310eb9de2e11`.
Foil uses only the Foil-1 subfolder.
No other checkpoint acts as a proxy.

The best tested upper-CI-NLL-eligible point is `O1_drop_0.3_renorm0`.
It omits 19.060% of total expert edges.
Its ΔNLL is 0.01328 [0.01094, 0.01561] nats/token.
The optimistic prefix-512 timing estimate is 1.05254×.
That estimate is not measured E2E acceleration.
The downstream 1 percentage point margin remains unverified.

## All held-out language arms

| Arm | Actual omitted work | ΔNLL [95% CI] | Mean top-1 agreement | Cache MiB |
| --- | --- | --- | --- | --- |
| B0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |
| C1_router_0.05 | 0.01% | -0.0003 [-0.0006, 0.0000] | 0.9945 | NOT_RECORDED |
| C1_router_0.1 | 0.07% | -0.0001 [-0.0008, 0.0005] | 0.9800 | NOT_RECORDED |
| C1_router_0.2 | 1.25% | 0.0005 [-0.0005, 0.0013] | 0.9709 | NOT_RECORDED |
| C1_router_0.3 | 6.00% | 0.0028 [0.0015, 0.0042] | 0.9564 | NOT_RECORDED |
| C2_reuse_guard0.0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |
| C2_reuse_guard0.05 | 0.17% | -0.0000 [-0.0008, 0.0006] | 0.9766 | NOT_RECORDED |
| C2_reuse_guard0.1 | 0.17% | -0.0000 [-0.0008, 0.0006] | 0.9766 | NOT_RECORDED |
| C2_reuse_guard0.2 | 0.21% | 0.0011 [-0.0001, 0.0022] | 0.9764 | NOT_RECORDED |
| C2_reuse_guard0.4 | 8.70% | 0.2473 [0.1867, 0.3174] | 0.8840 | NOT_RECORDED |
| C2_reuse_guardNone | 60.18% | 4.3506 [4.1751, 4.5341] | 0.2184 | NOT_RECORDED |
| N1_independent_site_drop_0.05 | 3.58% | 0.0094 [0.0075, 0.0114] | 0.9382 | NOT_RECORDED |
| N1_independent_site_drop_0.1 | 6.93% | 0.0225 [0.0196, 0.0255] | 0.9190 | NOT_RECORDED |
| N1_independent_site_drop_0.2 | 13.21% | 0.0506 [0.0457, 0.0561] | 0.8873 | NOT_RECORDED |
| N1_independent_site_drop_0.3 | 19.06% | 0.0856 [0.0782, 0.0938] | 0.8602 | NOT_RECORDED |
| N1_random_drop_0.05 | 3.58% | 0.0122 [0.0102, 0.0142] | 0.9397 | NOT_RECORDED |
| N1_random_drop_0.1 | 6.93% | 0.0312 [0.0277, 0.0351] | 0.9193 | NOT_RECORDED |
| N1_random_drop_0.2 | 13.21% | 0.0711 [0.0643, 0.0788] | 0.8817 | NOT_RECORDED |
| N1_random_drop_0.3 | 19.06% | 0.1281 [0.1168, 0.1417] | 0.8452 | NOT_RECORDED |
| O1_drop_0.05_renorm0 | 3.58% | 0.0013 [0.0002, 0.0024] | 0.9616 | NOT_RECORDED |
| O1_drop_0.05_renorm1 | 3.34% | 0.0024 [0.0009, 0.0038] | 0.9524 | NOT_RECORDED |
| O1_drop_0.1_renorm0 | 6.93% | 0.0024 [0.0010, 0.0037] | 0.9562 | NOT_RECORDED |
| O1_drop_0.1_renorm1 | 6.38% | 0.0072 [0.0053, 0.0091] | 0.9410 | NOT_RECORDED |
| O1_drop_0.2_renorm0 | 13.21% | 0.0065 [0.0048, 0.0082] | 0.9432 | NOT_RECORDED |
| O1_drop_0.2_renorm1 | 12.02% | 0.0163 [0.0137, 0.0187] | 0.9202 | NOT_RECORDED |
| O1_drop_0.3_renorm0 | 19.06% | 0.0133 [0.0109, 0.0156] | 0.9301 | NOT_RECORDED |
| O1_drop_0.3_renorm1 | 17.34% | 0.0273 [0.0242, 0.0305] | 0.9034 | NOT_RECORDED |
| O2_reuse_0.05 | 3.27% | 0.0016 [0.0005, 0.0028] | 0.9644 | NOT_RECORDED |
| O2_reuse_0.1 | 6.31% | 0.0082 [0.0065, 0.0100] | 0.9502 | NOT_RECORDED |
| O2_reuse_0.2 | 11.86% | 0.0309 [0.0275, 0.0344] | 0.9147 | NOT_RECORDED |
| O2_reuse_0.3 | 16.82% | 0.0674 [0.0615, 0.0736] | 0.8813 | NOT_RECORDED |
| q0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |


All 128 held-out examples enter each primary arm.
No-op and q=0 rows provide the strict zero-change reference.
The larger report contains actual generation accuracy counts and parsing limitations.
See [report.md](report.md) for timing definitions, counterexamples, and the overall decision.
