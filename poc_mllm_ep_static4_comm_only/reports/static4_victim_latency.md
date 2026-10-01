# Victim continuation and wall-time result

M2 used the live serving execution: after the Static4 decode combine, the
resident requests continued through the remaining model and completed their
next forced token. This is stronger than a communication-only microbenchmark,
but it also exposes prototype overhead.

| Paired request metric (n=10) | Full | Static4 | Static4 − Full |
|---|---:|---:|---:|
| response-window victim peak | 147.137 ms | 879.313 ms | +732.176 ms |
| injected prefill wall | 388.453 ms | 1,158.193 ms | +769.740 ms |

Static4 prefill wall overhead was `+199.64%` mean (95% paired-bootstrap CI
`[+179.62%, +217.89%]`). Therefore the smaller CUDA collective quantum did not
translate into victim ITL/VSEL improvement in this composition. The main
contributors are the per-layer D2H/Python slice planner, cross-rank readiness
barrier, temporary-buffer repack, and four handle-specific finalizes.

The component M0 Full run's independently aligned peak VSEL was 401.647 ms,
but it used heavy diagnostic capture; it is not mixed numerically with the
clean paired victim table. A reliable GPU-ready timestamp was not available,
so no CWT is claimed.
