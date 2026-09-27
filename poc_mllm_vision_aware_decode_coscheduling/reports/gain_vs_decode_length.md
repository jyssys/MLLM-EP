# Structural gain versus decode window

| Decode window | FIFO union | Length union | Text union | Full union | Separated union | Oracle union | Oracle reduction vs FIFO |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-32 | 29.490 | 29.214 | 29.214 | 29.214 | 29.214 | 28.811 | 2.300% |
| 33-64 | 32.729 | 32.044 | 32.044 | 32.044 | 32.044 | 31.531 | 3.660% |
| 65-96 | 33.518 | 32.135 | 32.135 | 32.135 | 32.135 | 32.784 | 2.191% |
| 97-128 | 32.263 | 31.229 | 31.229 | 31.229 | 31.229 | 31.656 | 1.881% |
| 1-128 | 31.315 | 30.626 | 30.626 | 30.626 | 30.626 | 30.545 | 2.458% |

The aggregate oracle is the primary gate. Its frozen assignment does not dominate the length control in later windows: at 65-96 and 97-128 it is worse than the length partition. This indicates phase-dependent overlap rather than a stable, large batching headroom.

No latency-versus-length curve is reported because Stage B was not authorized. These are measured routing-structure metrics, not speedups.
