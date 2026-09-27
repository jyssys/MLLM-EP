# Offline oracle headroom

The held-out, aggregate step 1-128 future-trace checkpoint oracle reduced mean layer/checkpoint wave union from **31.315** to **30.545** experts: **2.458% vs FIFO** and only **0.263% vs length-matched**.

The authoritative gate requires at least 10% against both controls. Both clauses failed. Discovery showed the same scale: **2.039%** vs FIFO and **2.668%** vs length control.

| Policy | Mean union | P50 | P90 | Reduction vs FIFO | Reduction vs length | Oracle recovery | Weight-set proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 FIFO / frozen random | 31.315 | 32.0 | 36.0 | 0.000% | -2.251% | 0.000% | 281.84 MiB |
| S1 length-matched | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S2 text-prefill | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S3 full-prefill / ELDR-style | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S4 vision/text-separated | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S5 future-trace oracle | 30.545 | 31.0 | 35.0 | 2.458% | 0.263% | 100.000% | 274.91 MiB |

The verified expert tensors occupy 9.0 MiB per expert/layer. The table's weight-set values are `ESTIMATED_DISTINCT_WEIGHT_BYTES_NOT_HBM_TRAFFIC`; they are not measured cache residency or HBM traffic.

The reused trace contains 22 fixed checkpoints rather than every integer step. The oracle assignments are optimized for their aggregate 1-128 objective and then reused in the per-window report. Window rows therefore are neither independently optimized per-window oracles nor a claim about unrecorded steps.
