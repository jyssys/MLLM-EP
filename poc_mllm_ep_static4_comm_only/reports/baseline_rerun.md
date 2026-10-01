# Matched baseline rerun

Full and CP512 were rerun on physical GPUs 2,3,6,7 with the same frozen
2,029-token input (1,980 visual tokens), resident global B8 decode workload,
model revision, BF16 weights, and TRUE EP4 backend. Values are request-repeat
statistics; layers are not treated as independent samples.

| Arm | CIDD ms/layer | mixed dispatch ms/layer | max dispatch quantum ms (median) | ExpertReloadAmp | expert total ms | instrumented prefill wall ms |
|---|---:|---:|---:|---:|---:|---:|
| Full | 6.484 (n=3) | 8.216 | 33.351 | 1.000x | 34.840 | 982.088 |
| CP512 | 1.652 (n=2 aligned) | 2.065 | 29.967 | 3.821x median | 90.986 | 1,110.626 |

The Full CIDD 95% bootstrap interval is `[3.829, 11.738]` ms/layer; the
CP512 interval is `[1.548, 1.756]`. Full's first measured repeat was a high
tail (13.055 ms/layer dispatch); it is retained rather than silently removed.
The CP512 CIDD has two aligned request repeats because one external-DP repeat
lacked the corresponding isolated-decode event. This is reported as limited
sample evidence, not hidden.

The healthy isolated global-B8 decode dispatch in the CP512 M0 lane averaged
0.413 ms/layer. The Full lane's decode-only diagnostic was itself inflated by
heavy route capture, so this healthy isolated lane is also the explicit
reference for Static4's raw CIDD. The raw evidence remains under
`artifacts/baselines/`.
