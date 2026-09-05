# Representative fixed-shape tail cases

All values are same-device CUDA-event milliseconds from the new live hook.

| phase/M | layer | DP/EP | whole | layout | dispatch | expert | combine | previous whole | first divergence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode/1 | 5 | 0/0, 0/1 | 1944.24 | 0.04 | 1941.81 | 0.64 | 0.06 | 0.94/1.23 | dispatch |
| decode/1 | 28 | 0/0, 0/1 | 229.84 | 0.04 | 228.76 | 0.44 | 0.05 | 0.97/0.96 | dispatch |
| prefill/137 | 0 | 1/2, 1/3 | 54.2--54.34 | 0.07--0.12 | 21.14--21.29 | 32.02--32.16 | 0.07 | ~1.2 | dispatch + expert |
| prefill/2048 | 0 | 0/0--1 | 73.57--76.07 | 11.09--11.50 | 0.24--2.62 | 0.03--30.47 | 0.48 | n/a | layout/expert |
| prefill/2048 | 1 | 1/2, 1/3 | 88.04--89.12 | 0.08--0.12 | 85.47--86.71 | 0.02--0.03 | 0.06 | 76.9--77.1 | dispatch |

The decode anchors are the cleanest fixed-shape result: same M/layer and identical DP-local progress, but an extreme dispatch span while expert/combine remain normal. Large prefill events can have multiple-stage contention, so they are not used alone to attribute the decode root cause.

## Population summary

- Baseline decode M=1: n=71,424; whole p50/p90/p99/max = 1.236/1.946/2.409/1,944.237 ms; 17 events >10 ms.
- Baseline prefill M=284: n=384; whole p50/p90/p99/max = 1.797/3.354/3.471/3.546 ms; no >10 ms events.
- Baseline prefill M=137: n=1,152; whole p50/p90/p99/max = 1.038/2.589/4.900/54.336 ms.
- Baseline prefill M=536: n=384; whole p50/p90/p99/max = 3.370/4.728/13.872/34.640 ms.
