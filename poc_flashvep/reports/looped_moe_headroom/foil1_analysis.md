# Foil-1 — model-specific result

Verdict: **NO_GO_HEADROOM for the tested native B1 grid**.
P0 status: PASS.
P1 completed. P2 completed with stated measurement limits. P3 did not run.

Checkpoint: `ShourenWSR/how-to-loop-moe` at `b3d128b44cbefd54e2c8784224a110186c8931f1`.
Foil uses only the Foil-1 subfolder.
No other checkpoint acts as a proxy.

The best tested upper-CI-NLL-eligible point is `O1_drop_0.2_renorm0`.
It omits 9.162% of total expert edges.
Its ΔNLL is 0.01221 [0.00717, 0.01748] nats/token.
The optimistic prefix-512 timing estimate is 1.01506×.
That estimate is not measured E2E acceleration.
The downstream 1 percentage point margin remains unverified.

## All held-out language arms

| Arm | Actual omitted work | ΔNLL [95% CI] | Mean top-1 agreement | Cache MiB |
| --- | --- | --- | --- | --- |
| B0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |
| C1_router_0.05 | 0.11% | -0.0006 [-0.0013, 0.0002] | 0.9811 | NOT_RECORDED |
| C1_router_0.1 | 0.28% | -0.0010 [-0.0020, 0.0000] | 0.9711 | NOT_RECORDED |
| C1_router_0.2 | 0.98% | -0.0001 [-0.0014, 0.0012] | 0.9632 | NOT_RECORDED |
| C1_router_0.3 | 4.08% | 0.0003 [-0.0015, 0.0022] | 0.9396 | NOT_RECORDED |
| C2_reuse_guard0.0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |
| C2_reuse_guard0.05 | 0.56% | 0.0430 [0.0384, 0.0476] | 0.9361 | NOT_RECORDED |
| C2_reuse_guard0.1 | 0.74% | 0.0648 [0.0590, 0.0709] | 0.9220 | NOT_RECORDED |
| C2_reuse_guard0.2 | 3.72% | 0.0784 [0.0722, 0.0848] | 0.8956 | NOT_RECORDED |
| C2_reuse_guard0.4 | 31.46% | 0.6544 [0.6156, 0.6944] | 0.6193 | NOT_RECORDED |
| C2_reuse_guardNone | 37.18% | 1.5990 [1.4918, 1.7087] | 0.4245 | NOT_RECORDED |
| N1_independent_site_drop_0.05 | 3.20% | 0.0065 [0.0038, 0.0094] | 0.9164 | NOT_RECORDED |
| N1_independent_site_drop_0.1 | 5.57% | 0.0153 [0.0115, 0.0192] | 0.8990 | NOT_RECORDED |
| N1_independent_site_drop_0.2 | 9.16% | 0.0316 [0.0255, 0.0377] | 0.8730 | NOT_RECORDED |
| N1_independent_site_drop_0.3 | 11.97% | 0.0528 [0.0450, 0.0607] | 0.8509 | NOT_RECORDED |
| N1_random_drop_0.05 | 3.20% | 0.0050 [0.0020, 0.0081] | 0.9188 | NOT_RECORDED |
| N1_random_drop_0.1 | 5.57% | 0.0142 [0.0097, 0.0188] | 0.8970 | NOT_RECORDED |
| N1_random_drop_0.2 | 9.16% | 0.0316 [0.0257, 0.0378] | 0.8727 | NOT_RECORDED |
| N1_random_drop_0.3 | 11.97% | 0.0529 [0.0456, 0.0608] | 0.8509 | NOT_RECORDED |
| O1_drop_0.05_renorm0 | 3.20% | 0.0005 [-0.0020, 0.0029] | 0.9388 | NOT_RECORDED |
| O1_drop_0.05_renorm1 | 3.06% | 0.0077 [0.0043, 0.0113] | 0.9324 | NOT_RECORDED |
| O1_drop_0.1_renorm0 | 5.57% | 0.0018 [-0.0020, 0.0056] | 0.9229 | NOT_RECORDED |
| O1_drop_0.1_renorm1 | 5.32% | 0.0123 [0.0082, 0.0167] | 0.9173 | NOT_RECORDED |
| O1_drop_0.2_renorm0 | 9.16% | 0.0122 [0.0072, 0.0175] | 0.8962 | NOT_RECORDED |
| O1_drop_0.2_renorm1 | 8.89% | 0.0211 [0.0162, 0.0265] | 0.8956 | NOT_RECORDED |
| O1_drop_0.3_renorm0 | 11.97% | 0.0383 [0.0291, 0.0474] | 0.8655 | NOT_RECORDED |
| O1_drop_0.3_renorm1 | 11.72% | 0.0501 [0.0412, 0.0591] | 0.8671 | NOT_RECORDED |
| O2_reuse_0.05 | 3.30% | -0.0009 [-0.0027, 0.0008] | 0.9514 | NOT_RECORDED |
| O2_reuse_0.1 | 5.97% | 0.0019 [-0.0002, 0.0040] | 0.9364 | NOT_RECORDED |
| O2_reuse_0.2 | 10.33% | 0.0213 [0.0174, 0.0255] | 0.9033 | NOT_RECORDED |
| O2_reuse_0.3 | 12.91% | 0.1250 [0.1057, 0.1467] | 0.8253 | NOT_RECORDED |
| q0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | NOT_RECORDED |


All 128 held-out examples enter each primary arm.
No-op and q=0 rows provide the strict zero-change reference.
The larger report contains actual generation accuracy counts and parsing limitations.
See [report.md](report.md) for timing definitions, counterexamples, and the overall decision.
