# Static4 physical results

Ten paired Full/Static4 request repeats were collected in five independent
two-repeat sessions to avoid a long-lived external-DP shutdown failure. CI
units are request repeats, not the 48 layers.

## Communication and decode priority

| Metric | Result |
|---|---:|
| S1 dispatch | 0.401 ms/layer |
| S2 dispatch | 0.308 ms/layer |
| S3 dispatch | 0.341 ms/layer |
| S4 dispatch | 0.296 ms/layer |
| Four-slice service total | 1.347 ms/layer |
| Max slice quantum, median across repeats | 0.455 ms |
| Max slice quantum, mean [95% CI] | 0.632 [0.445, 0.889] ms |
| Decode dispatch after S2 | 0.299 ms/layer |
| Decode full priority path | 1.034 ms/layer |
| Raw Static4 CIDD | -0.114 ms/layer |
| Nonnegative Static4 CIDD | 0.000 ms/layer |
| CIDD reduction versus Full | 100% bounded estimate |

The negative raw CIDD means no dispatch inflation was observed relative to the
0.413 ms/layer healthy isolated B8 decode reference; it is not interpreted as
a physical speedup and is clipped to zero for the reduction calculation.

Mean per-layer logical slice bytes were 7,530,667 / 7,528,619 / 7,526,400 /
7,525,376 bytes. Their mean CV was `0.000540`; all routes of a token remained
in one slice. The total 30.111 MB/layer is consistent with the prior roughly
1.445 GB figure being a 48-layer sum, not a per-layer payload.

## Expert efficiency and prototype costs

| Metric | Full | CP512 | Static4 |
|---|---:|---:|---:|
| ExpertReloadAmp | 1.000x | 3.821x median | 1.000x |
| Expert total | 34.840 ms | 90.986 ms | 28.048 ms |
| Expert delta versus Full | — | +161.2% | -19.5% |

Static4 executed exactly one prefill expert invocation per layer. There is no
CP512-like fragmentation. Prototype totals per request were: 226.744 ms CPU
slice construction, 546.728 ms arrival-barrier wait, 6.242 ms GPU repack,
31.786 ms four-slice combines (7.806 ms above Full), and 4.972 ms scatter.
Barrier time reflects rank-arrival skew and must not be added independently to
wall time as if it were non-overlapped compute; it identifies the fast-path
problem.

The four-arm comparison is:

| Arm | CIDD ms/layer | communication quantum | ReloadAmp | expert total | evidence |
|---|---:|---:|---:|---:|---|
| Full | 6.484 | 33.351 ms median max | 1.000x | 34.840 ms | measured TRUE EP4 |
| CP512 | 1.652 | 29.967 ms median max | 3.821x | 90.986 ms | measured TRUE EP4 |
| Layered | 6.484 | full-prompt quantum retained | 1.000x | 34.840 ms | `LAYERED_EP_PHYSICAL_REPLAY` |
| Static4 | 0.000 bounded | 0.455 ms median max slice | 1.000x | 28.048 ms | measured TRUE EP4 prototype |
