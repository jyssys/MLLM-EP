# Environment and Evidence Boundary

- Date: 2026-09-12--13 KST.
- Repository branch: `flashvep/refinement-aware-dynamic-wave-sizing-poc`, based on `1bcd811`.
- Model: local revision `LLaDA2.0-flash-744c3f8`, BF16, 32 layers, hidden 4096, 256 routed experts, top-8, one shared expert, block length 32.
- Runtime substrate: dInfer `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` plus the previously validated true-EP bridge and this PoC's trace-only additions.
- Software: PyTorch 2.8.0+cu128, SGLang 0.5.3.post1, DeepEP 1.2.1+73b6ea4.
- GPUs: physical 0--3 only, UUIDs `f217c8a0`, `a77f3471`, `24200107`, `17488c15`; all pairs report NV18 connectivity.
- Execution: dense TP4 plus routed EP4, DP1, no sequence parallelism. Each rank owns 64 complete routed experts; DeepEP normal mode performs remote dispatch and reverse combine. This topology and layer correctness were established in the parent PoC.
- Workload: bounded GSM8K pool of 32 submitted requests, generation budget 32, confidence threshold 0.9, identical prompts/configuration for all wave sizes.

The clean runs contain no RAWS/shape instrumentation. Trace runs add same-device
CUDA events at the whole-model boundary and at representative early/middle/late
layers (1/16/31), plus route-shape copies. Trace latency is never used directly
as clean BCT. Rank timing rows are collapsed with the maximum rank duration for
each sequential stage; rank rows are not added as if they were request time.

The runtime warns that DeepEP uses its default 20 communication SMs and that no
model-specific Triton expert config exists. These settings are held fixed. The
results characterize this validated substrate, not every possible DeepEP tuning.

GPUs 4--7 and their processes were never touched. The repository-owned burn on
0--3 was stopped before measurement. One mini32 trace launch failed before model
execution with a TCPStore nonce error; its artifact is preserved under
`rshape_tcpstore_failed` and excluded from all method evidence.
