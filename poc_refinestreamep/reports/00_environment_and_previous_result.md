# Environment and previous result

The experiment ran in the isolated worktree `MLLM-EP-refinestreamep` on branch
`flashvep/refinestreamep-serving-first-poc`, based on commit `a0c31c8`. Physical
GPUs 4, 5, 6, and 7 are H100 80 GB devices connected pairwise by NV18. Every
GPU launch asserted `CUDA_VISIBLE_DEVICES=4,5,6,7`; processes on GPUs 0--3 were
observed but never touched.

The downloaded model revision is `inclusionAI/LLaDA2.0-flash` at local revision
`744c3f8`: BF16, 32 layers, hidden 4096, 256 routed experts, top-k 8, one shared
expert, MoE intermediate 1024, eight routing groups/top-four groups, context
32768. The previously validated execution contract is dense TP4 plus routed
EP4, 64 routed experts per rank, remote dispatch, owner-rank fused expert
execution, and reverse combine.

The communication runtime is legacy DeepEP V1 commit `73b6ea4`, PyTorch
2.8.0+cu128 and CUDA 12.8. This matters: current [DeepEP main](https://github.com/deepseek-ai/DeepEP)
is V2, uses NCCL Gin, and unifies high-throughput and low-latency operations in
`ElasticBuffer`; it requires a materially different environment. V2 was audited
as prior art but not silently substituted for the validated LLaDA2 substrate.

The preceding RefineEP PoC established a post-compaction M range of 1--1024 and
found LL faster than Normal for every isolated real route. This PoC therefore
tests concurrency, capacity, and serving rather than repeating that conclusion.
