# Policy working-set comparison

All predictor policies use the same exact wave partitioner and the same top-32 per-layer predicted-set objective. Only the feature changes: text-only, modality-erased full-prefill, or separated vision/text channels.

| Policy | Mean union | P50 | P90 | Reduction vs FIFO | Reduction vs length | Oracle recovery | Weight-set proxy |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 FIFO / frozen random | 31.315 | 32.0 | 36.0 | 0.000% | -2.251% | 0.000% | 281.84 MiB |
| S1 length-matched | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S2 text-prefill | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S3 full-prefill / ELDR-style | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S4 vision/text-separated | 30.626 | 31.0 | 35.0 | 2.201% | 0.000% | 89.545% | 275.63 MiB |
| S5 future-trace oracle | 30.545 | 31.0 | 35.0 | 2.458% | 0.263% | 100.000% | 274.91 MiB |

On held-out, S1, S2, S3, and S4 selected exactly the same two waves. Their union, Jaccard, per-rank union, rank distribution, max/mean, and remote fraction are therefore identical. S4's apparent **2.201%** FIFO reduction is fully matched by the routing-blind length control.

The total request set and routes do not change across policies. Thus aggregate rank assignment distribution, rank max/mean, and remote-route fraction are expected to remain unchanged; the measurable structural target is wave-local expert union.
