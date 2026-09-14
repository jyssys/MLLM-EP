# Fixed-capacity mismatch

DeepEP's LL capacity parameter is a **per-source-rank maximum**, not global M.
In the 146 chronological real shapes, the maximum source-rank M has median 55,
p90 179, p99 252.4, and maximum 256. A peak contract of 256 therefore has a
median mismatch ratio of 4.65x, p90 57.6x, and maximum 256x.

Despite that large logical mismatch, the broad GPU matrix did not show a
monotonic latency penalty as the invocation contract increased from 32 to
2048. At M=32, three independent process restarts gave a normalized median
penalty of only 3.5% at capacity 32 relative to the best observed capacity;
larger capacities were within roughly 0--3%. This is noise-scale and has the
wrong monotonic direction for the proposed weakness.

Memory does scale linearly. For hidden=4096, EP4, 64 local experts, BF16, the
legacy capacity-256 contract requires about 2.03 GiB/rank NVSHMEM heap plus
1.00 GiB/rank for two receive buffers. Capacity 2048 grows those components to
16.25 and 8.00 GiB. This is a real memory-reservation issue, but no serving
latency mechanism was established. See `CAPACITY_MATRIX.csv` and Figures 03--04.
