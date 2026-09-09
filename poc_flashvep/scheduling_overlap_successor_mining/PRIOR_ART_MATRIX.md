# Adversarial prior-art matrix (live)

No novelty claim or successor promotion yet. Searches performed 2026-09-08;
primary paper technical bodies for the three bases have been read.

| Work | Already addressed | Potential collision / required distinction | State |
|---|---|---|---|
| [Layered Prefill](https://arxiv.org/abs/2510.08055) | Layer-axis prefill, chunk-amplified weight reloads, length-adaptive group count, group count tradeoff | Simple adaptive grouping is explicitly future work; measure residual beyond best group/chunk knobs | Full base audit |
| [FastPP](https://www.usenix.org/conference/osdi26/presentation/hwang) | Prefill-induced PP bubbles, attention-aware online ALP, dual-objective batch rebalancing | 'Token count misses context' already handled; E2E regret after ALP/static partition controls required | Full base audit |
| [NanoFlow](https://www.usenix.org/conference/osdi25/presentation/zhu-kan) | Nano-batch operation scheduling, SM partition, async CPU scheduling, profiling/search | Low-load weakness and workload-specific re-search already known; require material residual after portfolio/static-per-workload control | Full base audit |
| [DynaFlow](https://arxiv.org/abs/2605.21603) | Decouples model definition from physical schedule; supports multiple context-sensitive intra-device strategies | 'Make NanoFlow programmable/adaptive to workload' alone collides; its DeepSeek DBO study already controls batch size/context and its static-policy ablation loses the gain | Technical body read; high collision risk |
| [TokenWeave](https://arxiv.org/abs/2505.11329) | Wave-aware token split, comm/compute overlap, reordered/fused AllReduce–RMSNorm on Hopper | Small-token MoE splitting overhead is explicitly observed; H100 NanoFlow overlap ablation is modest; do not rediscover either | Technical body read; high collision risk |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefill/decode co-location | Baseline/reference only; do not relabel plain chunk optimization as successor | Reference |
| [TriInfer](https://mlsys.org/media/mlsys-2026/Slides/3756.pdf) | Stage-level encode/prefill/decode scheduling, image/token budgets, two-stream encoder–language execution, profiled hybrid disaggregation | Generic 'include encoder cost / overlap vision and decode' is already addressed; a distinct MoE-dependent interaction would be needed | Official MLSys slides read; full-paper screen pending if relevant |
| [Semantic Parallelism](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html) | Model/data locality co-scheduling | Existing project locality/rebatching negatives remain closed | Collision screen |
| [VPP](https://arxiv.org/html/2608.26523v1) | Fixed chunks with folded virtual stages, asynchronous handoffs and cross-request packing | Layout versus dynamic-chunk fragmentation is already a proposed MoE direction; not a fresh claim by itself | Technical body read, Aug. 27 2026 preprint; not labeled accepted |
| [FinDEP](https://arxiv.org/html/2512.21487) | Fine-grained disaggregated attention/expert scheduling, shared experts, task granularity/order search | Generic shared-expert/communication overlap needs a narrower distinction; deployment differs from co-located EP | Technical sections 2.3–4.2 inspected; no local reproduction claim |
| [Bullet](https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf) | Layer-wise SLO scheduling and dynamic SM partitions for concurrent prefill/decode; contention-aware model and online correction | Generic replacement of static NanoFlow plans by dynamic phase/resource control directly collides | Primary technical sections 3.2/3.4/4.1/4.2/4.4 inspected 2026-09-09 |

## VPP adversarial detail

VPP assumes approximately linear prefix-dependent stage-cost growth and explicitly
notes that less attention dominance or disabled EP weakens this regularity. Its
evaluation uses 16 Ascend 910C devices, TP8/PP2, one generated token per request,
and swept chunk budgets. Thus its “mixed” evaluation is mixed input lengths, not
steady prefill-plus-long-decode interference. It reports sensitivity to sparse
attention and admits that short-concurrency queueing can dominate TTFT. A potential
FastPP successor must distinguish its causal variable and request metric from
this already explored layout/chunk tradeoff; different hardware alone is not
novelty. The preprint is a collision risk, not a baseline result from this study.

OpenReview PDF retrieval hit browser verification; alternate public arXiv pages
were used to locate DynaFlow/TokenWeave. No CAPTCHA bypass or claim based on
inaccessible full text. Exact novelty assessment remains pending actual material
failure evidence and candidate-specific deeper comparison.

## Additional 9 September adversarial checks

Bullet's evaluated multi-GPU path is TP, including Qwen3-235B-A22B-FP8; it is
not evidence of solving arbitrary DeepEP routing. Its execution-state model
includes sequence/context lengths, both phase batch sizes and SM allocations.
An EP successor would need a measured limitation beyond these variables, not
merely an adaptive plan selector. No such limitation has been demonstrated here.
Source: [Bullet §§3.2, 4.1](https://xianweiz.github.io/doc/papers/26asplos_bullet.pdf).

FinDEP's technical §4 explicitly selects between attention/shared-expert task
orders and searches two levels of tensor partitioning. Its performance model
balances overlap against additional launches. Thus simply overlapping shared
experts or tuning nano size is not an unoccupied method space. The separated
attention/expert device groups differ from our native co-located NanoFlow EP
path; that difference alone is not novelty. Source:
[FinDEP §§2.3–4.2](https://arxiv.org/html/2512.21487).
