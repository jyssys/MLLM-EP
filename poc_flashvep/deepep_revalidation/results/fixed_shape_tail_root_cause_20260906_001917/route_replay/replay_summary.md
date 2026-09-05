# Fixed-route replay summary

- `isolated_text18_l45_r0/isolated.json`: exact hidden input and top-k route, M=2984, g=30, 20 warmups + 100 measured iterations, routing unchanged. Expert CUDA samples median 0.235 ms, p99 0.285 ms, max 0.382 ms.
- `controlled_text18_100/raw/rank*.jsonl`: same request and real DeepEP path, 100 measured iterations × 4 ranks. Layer 45 aggregate dispatch p50/p99/max = 0.366/1.004/2.393 ms; expert = 0.481/0.580/0.745 ms; combine = 0.101/0.614/0.677 ms.

Neither replay exhibits the 10--1,944 ms online dispatch tails. This separates route/input-conditioned execution cost from online outstanding-state cost for the giant tail class.
