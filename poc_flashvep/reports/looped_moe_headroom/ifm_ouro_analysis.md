# IFM MoE Ouro — model-specific result

Verdict: **NO_GO_HEADROOM for the tested native B1 grid**.
P0 status: CONDITIONAL_BACKEND_FIDELITY.
P1 completed. P2 completed with stated measurement limits. P3 did not run.

Checkpoint: `IFM/LoopedLM-P1-moe-ouro-500b` at `f0e9c8af22c10f34ac716c52495ceb00525f3556`.
Foil uses only the Foil-1 subfolder.
No other checkpoint acts as a proxy.

The best tested upper-CI-NLL-eligible point is `O1_drop_0.2_renorm0`.
It omits 16.835% of total expert edges.
Its ΔNLL is 0.01361 [0.01122, 0.01592] nats/token.
The optimistic prefix-512 timing estimate is 1.04622×.
That estimate is not measured E2E acceleration.
The downstream 1 percentage point margin remains unverified.

## All held-out language arms

| Arm | Actual omitted work | ΔNLL [95% CI] | Mean top-1 agreement | Cache MiB |
| --- | --- | --- | --- | --- |
| B0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |
| C1_router_0.05 | 0.05% | 0.0002 [-0.0005, 0.0010] | 0.9757 | NOT_RECORDED |
| C1_router_0.1 | 0.24% | 0.0008 [0.0001, 0.0015] | 0.9749 | NOT_RECORDED |
| C1_router_0.2 | 1.99% | 0.0007 [-0.0003, 0.0017] | 0.9687 | NOT_RECORDED |
| C1_router_0.3 | 12.99% | 0.0102 [0.0083, 0.0121] | 0.9333 | NOT_RECORDED |
| C2_reuse_guard0.0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |
| C2_reuse_guard0.05 | 0.16% | 0.0131 [0.0115, 0.0147] | 0.9691 | NOT_RECORDED |
| C2_reuse_guard0.1 | 0.16% | 0.0131 [0.0115, 0.0147] | 0.9692 | NOT_RECORDED |
| C2_reuse_guard0.2 | 0.16% | 0.0131 [0.0115, 0.0147] | 0.9691 | NOT_RECORDED |
| C2_reuse_guard0.4 | 1.37% | 0.0391 [0.0347, 0.0434] | 0.9549 | NOT_RECORDED |
| C2_reuse_guardNone | 48.30% | 3.3565 [3.2088, 3.4991] | 0.3173 | NOT_RECORDED |
| N1_independent_site_drop_0.05 | 4.44% | 0.0635 [0.0578, 0.0692] | 0.8973 | NOT_RECORDED |
| N1_independent_site_drop_0.1 | 8.71% | 0.1375 [0.1241, 0.1528] | 0.8472 | NOT_RECORDED |
| N1_independent_site_drop_0.2 | 16.84% | 0.2874 [0.2627, 0.3131] | 0.7744 | NOT_RECORDED |
| N1_independent_site_drop_0.3 | 24.40% | 0.4949 [0.4575, 0.5321] | 0.7088 | NOT_RECORDED |
| N1_random_drop_0.05 | 4.44% | 0.1508 [0.1403, 0.1620] | 0.8991 | NOT_RECORDED |
| N1_random_drop_0.1 | 8.71% | 0.3227 [0.3054, 0.3408] | 0.8392 | NOT_RECORDED |
| N1_random_drop_0.2 | 16.84% | 0.7204 [0.6907, 0.7515] | 0.7364 | NOT_RECORDED |
| N1_random_drop_0.3 | 24.40% | 1.2144 [1.1658, 1.2635] | 0.6413 | NOT_RECORDED |
| O1_drop_0.05_renorm0 | 4.44% | 0.0027 [0.0016, 0.0038] | 0.9616 | NOT_RECORDED |
| O1_drop_0.05_renorm1 | 4.21% | 0.0178 [0.0154, 0.0200] | 0.9276 | NOT_RECORDED |
| O1_drop_0.1_renorm0 | 8.71% | 0.0046 [0.0031, 0.0062] | 0.9488 | NOT_RECORDED |
| O1_drop_0.1_renorm1 | 8.17% | 0.0356 [0.0324, 0.0390] | 0.8983 | NOT_RECORDED |
| O1_drop_0.2_renorm0 | 16.84% | 0.0136 [0.0112, 0.0159] | 0.9233 | NOT_RECORDED |
| O1_drop_0.2_renorm1 | 15.65% | 0.0708 [0.0661, 0.0757] | 0.8603 | NOT_RECORDED |
| O1_drop_0.3_renorm0 | 24.40% | 0.0311 [0.0277, 0.0344] | 0.8968 | NOT_RECORDED |
| O1_drop_0.3_renorm1 | 22.60% | 0.1076 [0.1008, 0.1145] | 0.8319 | NOT_RECORDED |
| O2_reuse_0.05 | 3.50% | 0.0035 [0.0026, 0.0045] | 0.9628 | NOT_RECORDED |
| O2_reuse_0.1 | 6.83% | 0.0103 [0.0086, 0.0121] | 0.9435 | NOT_RECORDED |
| O2_reuse_0.2 | 12.99% | 0.0334 [0.0300, 0.0371] | 0.9023 | NOT_RECORDED |
| O2_reuse_0.3 | 18.58% | 0.0845 [0.0782, 0.0914] | 0.8566 | NOT_RECORDED |
| q0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |


All 128 held-out examples enter each primary arm.
No-op and q=0 rows provide the strict zero-change reference.
The larger report contains actual generation accuracy counts and parsing limitations.
See [report.md](report.md) for timing definitions, counterexamples, and the overall decision.
