# Final decision: NO-SERVING-GAP

DeepEP LL remains efficient under the tested concurrent, heterogeneous LLaDA2
refinement stream. Normal crosses over only between global M 4096 and 8192,
outside the real post-compaction range. The large per-rank capacity mismatch
inflates memory but does not produce a stable latency tax. The two-buffer-safe
pipeline improves Q=16 throughput by 13.1% over serial LL, and no Q<=16 point
reverses the backend winner.

The best-existing oracle is exactly 0%. The optimistic combined structural
oracle is +0.16% throughput/-4.53% p99 at 82.5% Poisson load,
+0.78%/-6.09% at 95%, and +2.19%/-2.14% at closed-loop Q=32. These results are
route-replay EP-stage serving, not full-model E2E. They fail both the throughput
and tail implementation gates.

Accordingly no custom CUDA kernel was built, no full-model online speedup is
claimed, and the paper-level RefineStreamEP direction is closed on this
4xH100 EP4 BF16 substrate.
