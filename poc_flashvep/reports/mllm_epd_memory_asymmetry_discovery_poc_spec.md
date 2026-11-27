# Memory-asymmetry discovery in native MLLM EPD

The primary experiment uses Qwen2.5-VL-7B-Instruct BF16 at `cc594898137f460bfe9f0759e9844b3ce807cfb5` on the established three-H200 E/P/D group. The runtime is the pinned vLLM 0.26 overlay. InternVL3-8B is the second architecture. Active decode keeps all KV local to D. The E-side pool stores immutable tensors and participates at admission or cache fill only.

## Comparison contract

- NativeStrong keeps its original compiled E path, BI0, chunked prefill, NIXL P→D, continuous batching and paged KV. Its original request-scoped inline encoder transfer remains enabled.
- The encoder-cache and prefix-cache comparisons use the same stable content identity and lifecycle. Three methods use the same stock cache manager and prefix hashes. Native APC retains P pages, the host tier places evicted-eligible pages in pinned host DRAM, and the peer tier places the same bytes on E. The initial implementation imports remote hits into P before suffix prefill and uses the existing P→D route.
- Encoder identity includes the model/processor contract, content digest and experiment isolation scope. Prefix identity additionally depends on exact causal token sequence, the ordered multimodal identifiers and the native block hash chain. The immutable model, original input and positions are kept in all paths.
- Capacity-only residency, concurrent transfers, natural-content reuse, and explicit controlled repeats have separate experimental labels. Original image/question pairs remain unchanged. Offered arrivals use a registered Poisson seed. Existing task-specific and historical serving SLOs are used consistently inside each paired comparison.
- Host, CPU and GPU bytes use GiB unless labeled otherwise. Raw image bytes mean the original compressed file. Logical KV bytes exclude paged rounding, allocator reservation and replicas. Timed client SSE includes queueing and dispatch lateness.

## Screening and decision gates

1. Recheck stock E capacity with touched 0/8/16/32/64/80 GiB resident arenas and original Main100. A candidate budget also needs an active-transfer test.
2. Measure actual image/embedding/KV payload sizes, vLLM encoder reuse, HBM/peer transfer and NVLink contention. Use natural content reuse and bounded adversarial working sets to separate capacity from traffic or sorting effects.
3. Evaluate pending and preempted pools only if a real blocked-P or D-preemption mechanism appears. Weight switching is a transfer feasibility check unless a real multi-model workload provides a bottleneck.
4. Compare no cache, native encoder and APC together, and host/peer exact prefix tiers at matched added bytes. Include a larger feasible P-local APC baseline when reserve allows. Tie reports to actual store/load counters and exact numerical checks.
5. For a positive online verdict, require the user-specified online gates, protected decode tails and task quality. Report independent starts, provenance and limitations. Highlight same GPU budget and same cache-byte results separately.

The historical SLO uses TTFT ≤ 5,000 ms, every observed client ITL ≤ 34 ms and TPOT ≤ 33 ms. Fixed-output controlled traces emit at least 16 tokens. Natural EOS keeps the original stopping rule. Campaign data and server logs stay outside Git. Commit the schemas, scripts, compact receipts and figures required to reproduce them.
