# Gate/up--SwiGLU--down--route-weight boundary atlas

The owner-rank BF16 replay contains exactly two useful grouped GEMMs
(`4096 → 2048` gated up, `1024 → 4096` down), followed by SwiGLU and
route-weight multiplication. Component CUDA events were taken independently;
their sums are **not** used as full-stage latency. Across actual dense
routes, timing-weighted standalone gate/up represents about 58--60% of
the grouped expert stage, down 36--38%, SwiGLU 3.8--6.3%, and route-weight
3.5--5.4%. Components can exceed 100% in aggregate because their
separate contexts have launches/intermediate buffers. Full stage,
not their sum, is the oracle denominator.

| Real route phase | active experts/local (GSM / HE) | local rows (GSM / HE) | SwiGLU / stage GSM | SwiGLU / stage HE |
|---|---|---|---:|---:|
| Dense early | 45.7 / 50.7 | 2044 / 2068 | 4.4% | 3.8% |
| Dense middle | 48.0 / 52.7 | 1822 / 966 | 4.2% | 4.3% |
| Dense late | 36.7 / 40.0 | 546 / 378 | 6.3% | 5.1% |
| Future-known live compacted late | 20.3 / 13.3 | 219 / 75 | 9.0% | 12.4% |

The very smallest cases can have activation+weight isolated fraction
>50%; they carry little **absolute** expert time. Using stage-weighted
request replay rather than averaging these high percentages prevents
spurious late-only promotion. [PHASE_COST_SUMMARY.csv](../PHASE_COST_SUMMARY.csv)
has the full table.

Two representative Nsight Systems traces (observer-heavy, GPU 4 only)
are [dense](../results/nsys_dense.nsys-rep) and
[future-compacted](../results/nsys_compacted.nsys-rep). These local binary
profiles are excluded by the repository's standard gitignore; the
[persisted extracted kernel counts/durations](../NSIGHT_KERNELS.csv)
support all numbers below even when a clone lacks the raw profiles.
Across 13 traced forward passes per shape (five warmups, eight profiled):

| Shape | GPU GEMM kernel calls | prepare-grouped calls | elementwise calls | GEMM GPU mean | prep GPU mean | elementwise GPU mean |
|---|---:|---:|---:|---:|---:|---:|
| Dense GSM early; 1015 local rows, active53 | 26 | 26 | 39 | 275.1 µs | 2.26 µs | 7.73 µs |
| Future-compacted HE middle; 137 rows, active19 | 26 | 26 | 39 | 102.8 µs | 2.30 µs | 4.53 µs |

This confirms two native GEMM-preparation kernels and three elementwise
ones per MLP invocation. The PyTorch replay has separate `silu`,
`gate*up`, and `down*weight` operations. Nsight CPU/NVTX range durations
are observer-heavy and must **not** be subtracted as removable gaps;
the GPU kernel-duration envelope is used only as a bound.

Analytic intermediate buffers per routed row: gate/up BF16 output
`2×1024×2 B = 4096 B`, an HBM write+read round-trip of 8192 B;
activated output `1024×2 B`, round-trip 4096 B. Combined
**possible** materialization is 12,288 B/row (12 MiB at 1024 rows);
its actual HBM traffic, cache hit rate and fraction of expert latency
are **not measured**, as Nsight Compute counters were unavailable.
Independent memcpy latency across 64--1024 rows was around 12--19 µs
but includes a separate launch and is **not** independently removable
work to add to measured SwiGLU. A separate, intentionally
double-counting `O2_plus_all_materialization_unrealistic` sensitivity
adds a generous standalone-memcpy proxy to O2 and stays <8% E2E,
even for future-known live-compacted routes. See
[MATERIALIZATION_SENSITIVITY.csv](../MATERIALIZATION_SENSITIVITY.csv).

SwiGLU useful arithmetic must still happen after fusion; removing all
standalone activation runtime is an impossible optimistic O2 ceiling.
Moreover, a two-GEMM forward generally still materializes the **activated**
intermediate for the down projection. An up-GEMM/SwiGLU epilogue can
avoid the earlier 2048-wide gate/up materialization and separate activation
launch but not magically eliminate all gate/up→down bytes. We assign only
60% of the small Nsight elementwise+preparation GPU envelope to the
*credible* O3; no memcpy byte proxy is added again. Local route-weight
multiplication is a replay primitive, whereas real DeepEP finalize already
applies top-k weights/reduction before reverse combine; route-weight
fusion is therefore not independently novel.
