# Best live overlap PoC

The strongest live exact diagnostic is the complete-wave three-stage schedule:

```text
combine(i-1) -> dispatch(i+1)     [DeepEP communication stream]
expert(i)                           [compute stream]
```

Across six real captured states, rank-critical savings are 0.106--0.186 ms per
layer with overlap efficiency 0.48--0.61. The minimum compute-output cosine is
0.99999988 and relative L2 is zero; all generated answer sequences match clean
baseline hashes.

This is intentionally **not** integrated into the production request path. It
requires three independent waves, whereas one request's refinement iterations
are sequentially dependent and best-static mini32 provides a single wave.
Calling it a 4--7% request speedup would be incorrect. Its scope-correct upper is
an independent-wave service/throughput oracle below 8%, before fill/drain and
queueing.

No direct-request live prototype passed the analytical gate. The implementation
therefore stops at read-only diagnostic replay rather than adding a production
scheduler or custom kernel.
