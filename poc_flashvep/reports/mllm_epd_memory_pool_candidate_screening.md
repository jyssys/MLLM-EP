# Memory-pool candidate screening

The primary runtime is Qwen2.5-VL-7B on H200 4/5/6 with one worker per stage. Requiring complete D-resident active KV avoids the repeated Encoder and Decode synchronization in EADA. No cache entry uses a transformed embedding, noncausal prefix, lossy KV, or new model weights.

## A · Pending KV

Historical NativeStrong QPS 1/2.5/5 used the Free proxy path for 300 requests. Its coordinator-pending field is structurally zero without a gate. It cannot observe the engine queue. The p95 wall gap from P-ready to D submission was 0.386/0.429/0.449 ms, with at most one request in the gap at a time. These are host observations. The exact NIXL source release time and blocked P bytes need separate evidence. D saturation can produce a long decode queue even if freeing P KV changes only where a request waits. Fresh NativeStrong Natural149 showed P KV peaks of 6.06% and 6.07% across the two sequences, P-ready→D-submit p95 of 0.448 and 0.429 ms, and the same Free proxy path. The sampled P queue reached 19, while native D EngineCore waiting reached one. Observed P queueing therefore does not establish that completed KV blocked new prefill. The pool was not activated against this trace. [Historical extraction](../mllm_epd_memory_asymmetry_discovery/results/historical_pressure.json)

## B · LM-ready Embedding

Natural149 offers 1,511 prior-image occurrences and 55 block-compatible exact-prefix requests. The measurement distinguishes image content opportunities from a deployed conservative ordered-document key. A stable transfer lease and cached-tensor ownership remain valid until all request owners and CUDA IPC acknowledgments retire. The encoder only skips forwards for actual entries in the retained tier. Actual native output equivalence is tested separately from the Python/HF grouped geometry checks. [Workload audit](../mllm_epd_memory_asymmetry_discovery/workload_audit.md)

### Native representation correctness

With APC and E retention disabled, two cycles of eight original questions completed 16/16 requests on H200 4/5/6. Neither cycle recorded an E hit. The encoder produced 23 BF16 tensors per cycle, and the audit matched all 46 measured tensors to their request owner and original image geometry. The unpaired runtime warmup produced one additional tensor. Both measured cycles ended with zero live leases and zero outstanding CUDA IPC exports. [Native digest and stage counters](../mllm_epd_memory_asymmetry_discovery/results/stable_ec_native_geometry_audit.json)

| Native comparison | Distinct image files | Pairwise byte-identical BF16 tensors |
|---|---:|---:|
| Recompute the same request | 14 | 23/23 |
| Same ordered images, different question | 7 | 28/28 |
| Same image, different original companions | 2 | 8/8 |

The changed-companion cases are two naturally occurring Mantis request pairs with no source document identifiers. Pairwise counts reuse tensor observations. The test establishes C0 for these original sequences and logical image grids. It leaves arbitrary image reordering, actual multi-request launch geometry, comparison to the separate SDPA diagnostic, all-layer KV state, output quality, and cross-model generality open. Diagnostic hashing synchronously copies tensors to the host, so this run supplies no timing claim. [Audit reducer and fixture selection](../mllm_epd_memory_asymmetry_discovery/stable_ec_audit.py)

### Storage and lifetime

The GPU retention path counts distinct E backing allocations and evicts only idle entries with no leases or outstanding exports. The opt-in pinned-host control uses the same exact E-side CUDA IPC transfer to P. It stores the completed source on host after the final clone ACK, stages an E copy on a host hit, then frees the staged allocation after use. A still-resident pinned copy avoids a second E→host write on repeated release. The five CPU tests cover request ownership, pending ACK, bounded storage, host staging and original processor identities. [Connector](../mllm_epd_memory_asymmetry_discovery/stable_ec.py) · [CPU tests](../mllm_epd_memory_asymmetry_discovery/test_stable_ec.py)

The active E GPU bound and retained-tier bound are separate in pinned-host mode. Pinned canonical bytes and temporary E staging may coexist, and the registry bounds exclude the stock model, Vision activations, and CUDA allocator reserve. Host load/store counters include blocking copies and allocation on the E worker. The control can stage an image before downstream APC or the P encoder cache makes that copy unnecessary. A P-local pinned cache that bypasses E, steady-state large pinned budgets, and host-pressure behavior require separate observations. [Connector](../mllm_epd_memory_asymmetry_discovery/stable_ec.py)

## C · Exact causal prefix KV

The native OffloadingConnector owns the hash chain, active reservations and LRU policy for both host and E-GPU storage. Canonical page copies on the P peer stream move exact BF16 bytes between P and E. Imported KV is read only after completion, then P completes the suffix and the existing NIXL path hands off to D. This method loads through P; direct E→D and P→D suffix placement remain separate design candidates. Four scratch-page cases passed exact 28-layer checks across ordinary scatter and partial first page offsets on peer and host backends. [Page witness](../mllm_epd_memory_asymmetry_discovery/results/peer_kv_witness.json)

## D · Preempted KV

Historical raw worker logs sampled D KV utilization at up to 2.2% and P at up to 5.4%, without a precise historical preemption count. Fresh NativeStrong Natural149 counted zero P and zero D preemptions in both 149-request sequences. D KV occupancy peaks were 5.71% and 4.48% in one-second scheduler samples. The additional QPS1 and QPS5 NativeStrong runs also recorded zero P/D preemptions, and their native P sampled occupancy peaked at 6.51% and 6.80%. Cache+APC counted zero preemptions at each of the three rates. Under these native settings, the dedicated D resume path was not triggered. The physical campaign measured D→E and E→D scratch transfers, pinned host paths and concurrent decode separately. [Physical transfer and recompute](./mllm_epd_representation_cache_economics.md)

## E · Weight cache

Actual 14.185 GiB of BF16 Qwen language parameters across 339 tensors copied from E to a D scratch model buffer in 42.46 ms wall median. The pageable-host input through two 64 MiB pinned staging buffers took 560.56 ms. This physical comparison covers the same tensor bytes. Runtime switching, model loading, graph warmup and a multi-model traffic replay remain `NOT_RUN`. The single-model workload provides no model-switch pressure. [Weight transfer receipt](../mllm_epd_memory_asymmetry_discovery/results/weight_copy.json)

## Full-model peer restore witness

An isolated serial test used 2 GiB of P KV and an 8 GiB peer tier to force eviction. The P worker fetched exactly 4,608 rows from E, with no local prefix hits, and recorded 264,241,152 B of physical E→P page copies. For the 4,714-token original request, those restored rows matched the cold recomputation byte-for-byte in all 28 layers on both P and D. NIXL also matched all 28 layers between P and D through the 4,713 imported rows in each arm. Prompt IDs, MRoPE positions, and MRoPE delta were identical. All 16 generated token IDs and D greedy argmax positions matched.

The 106 uncached rows ran with different P geometry. Complete prompt KV across arms was bitwise identical at one layer and the maximum relative L2 was 0.005145. The observed restore mechanism is byte-exact for the stored prefix. The pilot does not establish whole-forward bitwise reproducibility or task accuracy. Extra snapshots and CPU tensor copies make the test `CORRECTNESS_ONLY`. [Captured receipt](../mllm_epd_memory_asymmetry_discovery/results/peer_native_full28_correctness.json)
