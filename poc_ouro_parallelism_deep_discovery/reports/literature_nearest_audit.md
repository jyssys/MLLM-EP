# Nearest literature audit

Status: source audit complete for the eight assigned works. All runtime tests are `NOT_RUN`.

The worker first recorded time at 2026-10-19T09:55:24Z. The worker wrote this checkpoint after 2026-10-19T10:01:03Z. The time account excludes earlier unrecorded preparation. These minutes overlap the coordinator's campaign time. They do not add campaign elapsed time.

The recorded audit segment ends at 2026-10-19T10:05:56Z. Its observed duration is 632 seconds, or 10 minutes 32 seconds. The segment includes source reading, code inspection, analysis, artifact writing, and validation. It contains no deliberate wait. This is not ten campaign hours.

## Main conclusion

LoopSpec and WaveFront already implement mixed-depth token batching and pipelined self-speculation. A new distributed candidate must identify its additional ownership, placement, or communication mechanism. Moving an existing schedule onto several GPUs does not establish novelty or useful latency reduction.

LoopSpec and WaveFront provide the closest algorithmic controls. Neither paper reports the campaign's H100 P2/P4 comparison. FlashLoop provides a nearby approximation control. LLA and PLT change trained components or architecture. Their results cannot establish training-free Ouro headroom.

All performance values below are `SOURCE_REPORTED`. This worker measured no GPU performance or task quality. The source registry preserves authors, submission histories, versions, access times, and code revisions. The search log separates missing evidence from false claims.

## Pinned identities

| ID | Exact title | Pinned version and revision date | Official code at observed HEAD |
|---|---|---|---|
| S04 | LoPA: Scaling dLLM Inference via Lookahead Parallel Decoding | [2512.16229v2](https://arxiv.org/abs/2512.16229v2), 2025-12-22 | [LoPA](https://github.com/zhijie-group/LoPA/tree/433177a608dee8901bae0e7c02f9b6f341a6f6d0), `433177a608dee8901bae0e7c02f9b6f341a6f6d0` |
| S05 | DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models | [2402.19481v4](https://arxiv.org/abs/2402.19481v4), 2024-07-14 | [distrifuser](https://github.com/mit-han-lab/distrifuser/tree/8aebdd6d0db87884ba03dc596c98d28f3f38a1a3), `8aebdd6d0db87884ba03dc596c98d28f3f38a1a3` |
| S06 | AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising | [2406.06911v3](https://arxiv.org/abs/2406.06911v3), 2024-09-26 | [AsyncDiff](https://github.com/czg1225/AsyncDiff/tree/47cae2b021ec14a75eac8c37b14f69baf7f5cd07), `47cae2b021ec14a75eac8c37b14f69baf7f5cd07` |
| S08 | LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers | [2609.17184v1](https://arxiv.org/abs/2609.17184v1), 2026-09-15 | [loopspec](https://github.com/kaist-flexml-lab/loopspec/tree/41aa710ed7c9ce908cffa03597be5d1df19d7b56), `41aa710ed7c9ce908cffa03597be5d1df19d7b56` |
| S09 | WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models | [2609.23033v2](https://arxiv.org/abs/2609.23033v2), 2026-09-26 | [wavefront-decoding](https://github.com/summerbro-hhj/wavefront-decoding/tree/434520932897fcc1e26898891c7e2feaa6ab099f), `434520932897fcc1e26898891c7e2feaa6ab099f` |
| S10 | FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates | [2609.29812v2](https://arxiv.org/abs/2609.29812v2), 2026-09-28 | [FlashLoop](https://github.com/Superone77/FlashLoop/tree/aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6), `aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6` |
| S11 | Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers | [2607.15456v2](https://arxiv.org/abs/2607.15456v2), 2026-07-24 | `OFFICIAL_CODE_NOT_LOCATED` |
| S12 | Parallel Loop Transformer for Efficient Test-Time Computation Scaling | [2510.24824v1](https://arxiv.org/abs/2510.24824v1), 2025-10-28 | `OFFICIAL_CODE_NOT_LOCATED` |

The last two statuses describe this search. They do not assert that no implementation exists. Non-author reproductions are not official code. PLT's HTML contains an August 2026 document heading. The arXiv submission history still lists only v1. This audit pins the arXiv identifier rather than inferring a new version.

## Mechanism and scope

### S04: LoPA

LoPA explores token-filling orders in masked diffusion LMs. Its confidence-based winner selection changes the generation distribution. It does not provide ordinary distribution-preserving speculative verification. The algorithm needs no new training, but D2F checkpoints already contain distilled components. The CUDA runtime uses branch parallelism on H200 GPUs in BF16. The headline 1073.86 tokens/s belongs to the Ascend engine, which uses W8A8. The paper separates projected A100 results from actual system tests. Its NVIDIA cache protocol pre-writes branch state, then broadcasts winning cache blocks. See Sections 2, 3.2–3.4, 4.1, and Appendix B. [LoPA paper](https://arxiv.org/html/2512.16229v2)

### S05: DistriFusion

DistriFusion divides an image into patches. Each device retains the model and computes fresh local regions against displaced remote features. Asynchronous AllGather prepares context for the next denoising step. The method also corrects stale GroupNorm statistics. It needs no retraining, but changes intermediate values. The default SDXL protocol uses 50 DDIM steps and five synchronous initial steps. The reported eight-A100 gain is 2.8× at 1024×1024 and 6.1× at 3840×3840. Few-step quality, small workloads, and interconnect cost limit transfer. See Sections 4–6 and Tables 1–2. [DistriFusion paper](https://arxiv.org/html/2402.19481v4)

### S06: AsyncDiff

This is the 2024 asynchronous-denoising paper, not an unrelated work with the same name. It assigns model components to devices. Components consume neighboring-step hidden features after exact initial denoising steps. Stride denoising also skips work and reduces communication frequency. The transformation is training-free and approximate. Tests use 50 DDIM steps and A5000 GPUs with NVLink Bridge. Four-device configurations do not all mean four equal components. For stride two, three components use four devices. Quality and warm-up cost remain part of the tradeoff. See Section 3.2, Section 4.1, and Tables 1–2, 5. [AsyncDiff paper](https://arxiv.org/html/2406.06911v3)

### S08: LoopSpec

LoopSpec batches branches at different recurrent depths through shared weights. An early readout starts future-token work before full-depth verification. A gated residual second proposal can preserve a fallback continuation. Rejection sampling preserves the target distribution under its stated mathematical assumptions. Branches retain separate recurrence-indexed KV. The paper reports single-batch BF16 tests on one RTX PRO 6000 Blackwell GPU. Table 1 reports GSM8K decode-time gains of 2.85× for Ouro-1.4B and 3.12× for Ouro-2.6B with proposal depths 1 and 2. These are not H100 distributed results. Original-gate adaptive serving is not established. See Sections 4, 6 and Appendix B. [LoopSpec paper](https://arxiv.org/html/2609.17184v1)

### S09: WaveFront Decoding

WaveFront co-batches shallow drafting and deep verification. Rejection discards the affected speculative suffix. The paper uses greedy verification and recurrence-specific KV for Ouro. BF16 output differences remain possible. The reported 2.42× Ouro-2.6B result measures aggregate output throughput, including prefill, on one A6000. A separate static-shape CUDA-graph test reports 38.51 versus 16.72 tokens/s, or 2.30×. Huginn's shared-KV extension is approximate and is not the Ouro configuration. Adaptive depth and P/R/C disaggregation remain empirical follow-ups in Appendix D. See Sections 3–4 and Appendices B.4, C.2, D. [WaveFront paper](https://arxiv.org/html/2609.23033v2)

### S10: FlashLoop

FlashLoop combines sparse token updates, sparse attention, and INT4 cross-loop KV residuals. It needs no retraining, but it is approximate. Its task-score language does not prove identical outputs. The paper reports single-A100-SXM4-40GB efficiency tests. The main Ouro gains combine sparsity with quantization. They cannot serve as BF16-only distributed headroom. Version 2 also states that token sparsity can increase decode latency before other components offset that cost. Retention schedules use 128 WikiText-2 sequences and a reconstruction-error criterion. See Sections 4–5 and Appendices E, G. [FlashLoop paper](https://arxiv.org/html/2609.29812v2)

### S11: Looped Latent Attention

LLA fits a cross-loop cache codec from teacher activations. It refines codec parameters with KL and attention-output losses. Frozen teacher weights do not make this path training-free. The exact scalar compression ratio does not imply exact reconstruction or faster decoding. Table 5 reports 74.1 versus 63.6 ms/token for the full-memory path. The latent-store path trades reconstruction compute for capacity. Single-H200 capacity tests fabricate decode caches and increase batch until OOM. These are not natural end-to-end serving tests. Main scale evidence uses Ouro-2.6B-Thinking, not the requested 2.6B Base target. See Sections 3, 6–7 and Appendix A. [LLA paper](https://arxiv.org/html/2607.15456v2)

### S12: Parallel Loop Transformer

PLT changes the computation graph during training. It shifts previous-loop states by one token and adds input embeddings. Later loops share first-loop global KV and use a trained gate for local sliding-window attention. This enables a diagonal schedule by changing dependencies, not by rescheduling unchanged Ouro. Algorithm 2 specifies the training procedure. Main latency tests use one GPU, FP8 attention, and W4A8 linear layers. They do not establish the campaign's BF16 systems comparison. The paper also describes dense OLMo and MoE experiments. See Section 2.2, Algorithms 1–2 and 4, Section 3.1.1, and Appendix A.2. [PLT paper](https://arxiv.org/html/2510.24824v1)

## Transfer worksheet

| Work | Original bottleneck | Enabling model property | Added work, state, or approximation | Ouro correspondence |
|---|---|---|---|---|
| LoPA | Confidence limits parallel token filling | Masked positions permit alternative filling orders | Extra branches, confidence evaluation, and winner-cache synchronization | Shared weights exist. Free filling-order choices do not exist in causal Ouro decoding. Recurrent depths are dependent states, not alternative orders. |
| DistriFusion | Synchronous context exchange removes patch-parallel gains | Nearby diffusion steps tolerate displaced patch context | Full stale feature buffers, asynchronous exchange, and normalization correction | Shared operators exist. Tolerance to displaced depth features is unestablished. R4 offers less initialization amortization than its many-step regime. |
| AsyncDiff | Sequential component inputs prevent device overlap | Neighboring denoising features support tested approximations | Initial exact steps, stale boundary features, broadcasts, and optional stride skipping | Component partitioning is possible. Stale Ouro boundary quality is not inherited. A new version needs a changed-assumption test against prior failure. |

These transfer statements are architectural inferences. They do not assert that a new approximation must fail. The descriptions above cite the corresponding source mechanisms.

## Official code findings

### LoPA: cache commit is real work

`_broadcast_kv_update` allocates key/value buffers for each affected layer. It broadcasts both tensors from the winning rank. `_commit_shared_cache` dispatches that update. Winner selection calls this path after choosing the best branch. See lines 1443–1520 and 2558–2614. The inspected score combines creation confidence with the lowest-confidence fraction of remaining positions. This differs from the paper's simple mean in Equation 2. A comparison must pin the actual score. [Pinned generation code](https://github.com/zhijie-group/LoPA/blob/433177a608dee8901bae0e7c02f9b6f341a6f6d0/lopa_dist_nv/generate_branch_parallel.py#L783)

The model's static-cache attention collects candidate chunks before attention and appends them afterward. `commit_kv_chunks` identifies the affected span. Winning blocks then overwrite peer state. A losing branch's pre-write is not an authoritative commit. See lines 1026–1124 and 1390–1415. [Pinned model code](https://github.com/zhijie-group/LoPA/blob/433177a608dee8901bae0e7c02f9b6f341a6f6d0/lopa_dist_nv/model_cache/dream/model_dream_bp.py#L1026)

### DistriFusion: fresh local KV does not remove remote context

`DistriSelfAttentionPP` uses synchronous AllGather during initial steps. Later calls replace the local slice of buffered KV and enqueue communication. The forward path waits for the preceding pending handle before reuse. This is a concrete stale-context protocol, not communication-free attention. See lines 107–195. [Pinned attention code](https://github.com/mit-han-lab/distrifuser/blob/8aebdd6d0db87884ba03dc596c98d28f3f38a1a3/distrifuser/modules/pp/attn.py#L107)

### AsyncDiff: compute assignment is not weight sharding

`AsyncDiff.__init__` sends the entire pipeline to each rank's device. `ModulePlugin` selects local execution or cached results. It broadcasts cached component outputs. The constructor requires `model_n + stride - 1 == world_size`. Thus this implementation does not demonstrate memory savings from physically sharded model weights. See lines 6–119. [Pinned AsyncDiff code](https://github.com/czg1225/AsyncDiff/blob/47cae2b021ec14a75eac8c37b14f69baf7f5cd07/asyncdiff/async_sd.py#L45)

### LoopSpec: separate branch state and fixed-depth scope

The branch manager shares prefix KV through mapping indices. Each branch owns new decode slots. Release excludes borrowed prefix slots. The manager explicitly describes an external batch-one scheduler. It clears shared allocator bookkeeping between requests. This is not evidence of concurrent-request isolation. See lines 106–144 and 207–250. [Pinned branch manager](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/branching/manager.py#L106)

The Ouro executor applies every physical block and then the loop norm. It retains the checkpoint exit gate but never evaluates it. See lines 85–98. [Pinned Ouro adapter](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/modeling/ouro.py#L85)

The server explicitly rejects LoopSpec when any supported parallel size differs from one. The check includes TP, PP, DP, EP, attention CP, and MoE DP. It also rejects PD disaggregation and PDMux. These restrictions establish the inspected implementation boundary. They do not prove that a distributed extension is impossible. See lines 88–118. [Pinned server CLI](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/cli.py#L88)

`LosslessLoopSpecExecutor.forward` advances all active lanes through one runner. It sends their depths, branch identities, and state slots into each batched recurrent call. Endpoint readouts then drive verification. See lines 293–339. [Pinned executor](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/executor.py#L293)

### WaveFront: adaptive code exists without matching-baseline validation

The scheduler verifies positions in order and rolls back after rejection. It also supports a temperature-sampling path. This is broader code scope than the paper's main greedy experiments. See lines 14–32 and 155–224. [Pinned scheduler](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/scheduler.py#L155)

The Ouro adapter contains an original-gate path. It accumulates halting probability and fills skipped depths with last-computed KV. It fills these slots immediately because later speculative positions can read them. Threshold, minimum depth, and baseline-equivalence tests still need inspection. See lines 84–132 and 375–397. This code is not evidence of validated adaptive H100 speedup. [Pinned Ouro blocks](https://github.com/summerbro-hhj/wavefront-decoding/blob/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/ouro_blocks.py#L375)

### FlashLoop: a nonquantized switch is available

The config can disable quantization or sparsity. Validation requires four loops, 24 or 48 physical layers, equal Q/KV head counts, and head dimension 128. The engine selects a BF16 cache builder when the config disables quantization. Decode accepts one token at batch size one. These switches permit a scoped BF16 ablation. They do not establish its quality or speed. [Pinned config](https://github.com/Superone77/FlashLoop/blob/aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6/flashloop/config.py#L14), [pinned engine](https://github.com/Superone77/FlashLoop/blob/aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6/flashloop/engine.py#L208)

## Constraints on candidate claims

- A mixed-depth frontier alone overlaps directly with LoopSpec and WaveFront.
- Versioned branch KV and prefix-index sharing already appear in LoopSpec's runtime.
- P/R/C disaggregation already appears as WaveFront future work. A candidate needs a more specific distinction.
- Sparse residual updates overlap with FlashLoop. Disable its quantization before drawing BF16-only conclusions.
- A fitted cross-loop basis overlaps with LLA and violates this campaign's no-new-training rule.
- Shifted token dependencies overlap with PLT and do not preserve the original Ouro function.
- A useful P2/P4 design must beat the closest single-GPU speculative method and the strongest matched distributed baseline.

### Strongest skeptical comparison for distributed speculation

LoopSpec already combines the relevant algorithmic ingredients. It creates causal speculative inputs, batches mixed-depth states, preserves separate depth KV, and verifies against the full-depth target. Its intended guarantee is greedy exactness or preservation of the sampling distribution. It does not substitute stale hidden states. Floating-point reordering remains a separate correctness issue.

A distributed extension must add rank placement, remote KV access, distributed lifetime control, and winner/prune messages. It must also define migration or stationary ownership for each branch's hidden state and depth KV. These requirements are missing from the inspected LoopSpec path. They are not individually new parallel axes.

Independent branch replicas can lose the weight amortization of the original shared-block batch. They also duplicate weights and may require prefix-cache transfer. Ordinary TP applied to the existing mixed-depth batch is therefore a necessary generic control. A recurrence-specific claim needs evidence that tied weights change the ownership or residency optimum. A diagram with several devices does not supply that evidence.

## Unresolved evidence

No assigned method ran in this worker. The worker inspected official repositories through pinned source URLs. Their dependencies, numerical correctness, and benchmark reproducibility remain untested. No inspected result establishes both Base sizes, both depth modes, and true H100 P2/P4 execution.

The search also returned a lead titled *Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion LMs*, identifier 2510.14961. This worker did not verify that lead. It must not enter the technical claim ledger without a primary-source check.

Reproducibility files: `artifacts/literature_sources.jsonl` and `artifacts/literature_searches.jsonl`.
