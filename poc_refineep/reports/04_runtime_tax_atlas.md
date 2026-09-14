# Runtime tax atlas

## Measured boundary

`RUNTIME_TAX.csv` measures communication semantics: layout, dispatch, a small
weight-application step, and combine. It does not measure an invented
RefineEP kernel. Expert compute is accounted separately in the request oracle.

The normal-path layout kernel is only about 32--33 us at the event boundary;
reusing its route handle saves about 40--46 us per layer-wave. This is far less
than the normal-to-LL gap for medium/small work, so fixed buffer/handle reuse
alone does not define a competitive third path.

| shape | normal (ms) | LL (ms) | measured payload LB (ms) | credible target (ms) |
|---|---:|---:|---:|---:|
| very-small | 0.2449 | 0.0909 | 0.00050 | 0.05050 |
| small | 0.2457 | 0.0903 | 0.00220 | 0.05220 |
| medium-small | 0.2492 | 0.0928 | 0.00801 | 0.05801 |
| medium | 0.2610 | 0.1292 | 0.03491 | 0.08491 |
| large | 0.2768 | 0.2125 | 0.07192 | 0.12192 |

The payload bound uses a measured 854.7 GB/s aggregate two-way DeepEP
bandwidth from the same physical substrate at global M=4096, not theoretical
H100 peak. At small M, startup/control dominates; at large M, actual payload
matters more.

Selected Nsight Systems traces show legacy low-latency dispatch/combine kernel
medians near 71.6/53 us for a very-small case and 113/116 us at M=1024. The
trace contains large NVSHMEM/NCCL initialization outliers, so total trace wall
is not used as clean timing. Nsight kernel medians only corroborate the CUDA
event envelope. Nsight Compute was unavailable and no SM-utilization number is
fabricated.

The key result is a distinction between large *operator-local* reducible tax
and small *request-level* headroom: even an impossible zero-control path can
only remove a bounded fraction after the already fast LL endpoint and the rest
of the model are included.
