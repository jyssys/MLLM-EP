# Operator and KV ownership audit

Status: source audit complete for this branch. All GPU, runtime, and task-quality tests are `NOT_RUN`.

## Outcome

C01 and C02 overlap with established batching and attention–FFN disaggregation. C04 overlaps with generic layout changes and collective overlap. C07 overlaps with verified branch batching and shared-prefix cache management.

The strongest baseline is not one remote operator per GPU. It is a measured choice among local continuous batching, replicated engines, and TP of the same useful ready work. NanoFlow supplies a concrete operator-pipelined control. Sarathi supplies a concrete admission and token-budget control. These systems do not prove that either control wins for Ouro at concurrency 1–8.

KernelFlume is the closest new ownership prior. It separates weight-bearing dense computation from weightless KV/core-attention workers. Its routing protocol already distinguishes immutable historical ranges from live tail appends. A new Ouro design needs a more specific mechanism than separate compute and cache owners.

The [source registry](../artifacts/operator_ownership_sources.jsonl) preserves identities, versions, code revisions, searches, access intervals, and unsupported paths. This report extends the [earlier literature audit](literature_nearest_audit.md). It does not change that frozen report.

## Pinned primary sources

| ID | Work and latest arXiv version observed | Inspected code or availability |
|---|---|---|
| O01 | [Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving, 2507.06608v5](https://arxiv.org/html/2507.06608v5), 2025-08-07 | Official code `NOT_LOCATED` in this bounded search. |
| O02 | [FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling, 2504.03775v1](https://arxiv.org/html/2504.03775v1), 2025-04-03 | Official code `NOT_LOCATED`. This is not the multi-turn compression paper with the same name. |
| O03 | [NanoFlow: Towards Optimal Large Language Model Serving Throughput, 2408.12757v2](https://arxiv.org/html/2408.12757v2), 2025-05-25 | Official default branch `Nanoflow-python`: `f179a907828b87e042126e585880301ff5b2c62a`. C++ main: `d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57`. |
| O04 | [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve, 2403.02310v3](https://arxiv.org/html/2403.02310v3), 2024-06-17 | Official revision `96f9911790ecc00af12ee9fae47cb8fa9ba0d199`. |
| O05 | [Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding, 2507.07120v1](https://arxiv.org/html/2507.07120v1), 2025-07-07 | TensorRT-LLM revision `59eab99c898d1e3fdd78dc59373b1c360b7321ae`. This is not the max-flow serving paper named Helix. |
| O06 | [KernelFlume: Elastic Core-Attention Scaling for Agentic Long-Context Decoding, 2606.29207v1](https://arxiv.org/html/2606.29207v1), 2026-06-28 | Official code `NOT_LOCATED`, including the inspected author's publication page. |
| O07 | [Efficient Memory Management for Large Language Model Serving with PagedAttention, 2309.06180v1](https://arxiv.org/html/2309.06180v1), 2023-09-12 | Ownership mechanism audited from the paper. Its current code was not audited here. |
| O08 | [vLLM afd-plugin](https://github.com/vllm-project/afd-plugin/tree/8d55ed11f4ce19beea6cc2c5bdeca1ced8946623), official code | Revision `8d55ed11f4ce19beea6cc2c5bdeca1ced8946623`. Target: vLLM 0.30.0. |

The selected method and limitation sections were available for all seven latest arXiv versions. No latest arXiv body is unavailable in this branch. Missing official code does not mean that no code exists. No inspected repository ran.

## Concrete generic baseline: keep useful batches together

### NanoFlow: overlap operations without dedicating ranks to operators

NanoFlow splits independent batch rows into nano-batches. It overlaps compute-heavy projections/FFNs, memory-heavy attention, and network collectives within the same device. The search retains dependencies when both parent operations and row ranges overlap. It also considers equivalent collective/weight-layout choices. Profiling includes kernel interference.

Section 4.1 profiles dense batches from 128 tokens upward. The main serving evidence uses A100s and large token batches. It is not evidence of low-concurrency Ouro latency. Section 4.2 forms the next batch asynchronously. Delayed EOS retirement can add one useless decode token. That work must remain in any cost account. These are inference-runtime changes, not new model training. [NanoFlow, Sections 4.1–4.2](https://arxiv.org/html/2408.12757v2)

The C++ path makes the mechanism concrete. `setWeight`, lines 537–550, binds multiple nano-operators to the same `W_D`, `W_KQV`, `W_G`, and `W_U` pointers. `Pipeline::run`, lines 599–740, event-orders projections, attention, FFN, and collectives across streams. Line 632 selects physical weights with `iter % ModelConfig.model_layer`. This supports a generic weight-alias control. It does not establish correct Ouro cache, norm, or gate semantics. [Pinned C++ pipeline](https://github.com/efeslab/Nanoflow/blob/d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57/pipeline/src/pipeline.cu#L537)

The main scheduler retains request-local KV objects, then builds concatenated token and page metadata at lines 200–241. The newer default branch instead constructs an explicit operation DAG. Its executor topologically sorts dependencies and waits on events before each operation. These are two inspected branches, not one assumed reproducible artifact. [Main scheduler](https://github.com/efeslab/Nanoflow/blob/d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57/pipeline/utils/scheduler.py#L200), [default-branch executor](https://github.com/efeslab/Nanoflow/blob/f179a907828b87e042126e585880301ff5b2c62a/core/executor.py#L24)

### Sarathi: admit ready work under a token budget

Sarathi first schedules ongoing decodes. It then adds partial prefills and new prompt chunks within a profiled token budget. A chunk is large enough to improve dense-operation efficiency but small enough to limit decode stalls. Chunking also repeats reads of earlier prompt KV. Uniform batch work can reduce ordinary PP bubbles. Tests include single- and multiple-A100 deployments. This is a generic unchanged-model scheduler, not a recurrent-depth method. [Sarathi, Section 4 and Algorithm 3](https://arxiv.org/html/2403.02310v3)

The official `SarathiScheduler._schedule` adds decode rows at lines 126–160. It adds partial prefills at 162–187 and admits new requests at 196–235. Admission checks cache capacity and sequence count. `_get_seq_next_num_prefill_tokens`, lines 74–96, subtracts occupied tokens from the chunk budget. This is an executable scheduling mechanism, not only a batching slogan. [Pinned scheduler](https://github.com/microsoft/sarathi-serve/blob/96f9911790ecc00af12ee9fae47cb8fa9ba0d199/sarathi/core/scheduler/sarathi_scheduler.py#L98)

### Required comparison

For the campaign's same-P comparison, retain the same total concurrency, arrivals, prompts, token budget, precision, and quality policy.

1. Run local continuous batches on replicated complete engines when memory permits.
2. Run TP over one shared batch of the same ready rows.
3. Add dependency-correct nano-batch overlap without remote operator ownership.
4. Compare the candidate's worker routing with those controls.

For C07, use the same verifier, proposal budget, and branch work in all layouts. Compare a shared mixed-depth batch, ordinary TP of that batch, and split full replicas. Charge rejected work and shared-prefix state movement. The earlier audit already identifies LoopSpec and WaveFront as direct controls.

This comparison is an experimental prescription, not an observed winner. DP can fragment batches. TP can add collectives. Remote pooling can add queueing and round trips. Measure the best valid configuration for the actual concurrency.

At C1 without a valid speculative branch, a second ready request does not exist. A scheduler cannot overlap the next recurrent depth before its predecessor finishes.

## Ownership priors

### Nexus: phase isolation inside one engine

Nexus separates prefill and decode into coroutines with different streams and schedulers. They share the serving engine and its weights. CUDA Green Contexts partition SMs. A cost model selects the partition, and hysteresis limits frequent switches. Shared worker metadata needs guards. The implementation pre-creates partition layouts.

Section 6 tests Qwen and Llama on one or two L20s. This is neither a remote MLP pool nor a layer/depth ownership protocol. It supplies a generic resource-partitioning control for C01. Its model/workload-specific profiling also has a cost. The paper describes an implementation, but this audit did not locate official code. [Nexus, Sections 4–6](https://arxiv.org/html/2507.06608v5)

### FlowKV: state layout and transfer are not free

FlowKV changes KV layout from `(L,2,B,H)` to `(B,L,2,H)`. Here B denotes cache blocks, not request batch size. It allocates contiguous segments, merges adjacent free segments, and aligns source/destination block ranges before coalesced transfers. Its scheduler can change prefill/decode roles under load.

The transfer crosses inference phases. It is not a per-layer remote-operator protocol. Tests use A100/NVLink and heterogeneous L20/H20 deployments. The method does not remove KV bytes or certify shared states across recurrent depths. Its relevant lesson for C01/C07 is to charge allocation layout, packing, transport launches, and destination completion. [FlowKV, Sections 3.3–4.4](https://arxiv.org/html/2504.03775v1)

### vLLM AFD: C02's remote pooling pattern already exists

The CUDA `P2pNcclAFDConnector` maps one or more attention ranks to an FFN rank. It receives their hidden states and concatenates rows, lines 518–552. `send_ffn_output`, lines 554–618, splits the result by recorded row counts and returns each slice to its owner. The control plane communicates shapes and prepares graph-stable receive buffers. This is direct prior machinery for pooled stateless work. It does not require transferring historical KV. [Pinned connector](https://github.com/vllm-project/afd-plugin/blob/8d55ed11f4ce19beea6cc2c5bdeca1ced8946623/afd_plugin/connectors/gpu/p2p.py#L518)

The inspected CUDA connector is synchronous. It requires at least as many attention ranks as FFN ranks. Its graph/DBO path supports exactly two microbatches. The registered wrappers cover DeepSeek/GLM-MoE/Qwen-MoE families, not Ouro. GPU evidence includes H20, not this campaign's Ouro H100 matrix. Treat it as `OURO_ADAPTER_UNSUPPORTED`, not a drop-in baseline. A dense Ouro implementation would still need the correct residual, norm, layer, and cache contract. [Connector limits](https://github.com/vllm-project/afd-plugin/blob/8d55ed11f4ce19beea6cc2c5bdeca1ced8946623/afd_plugin/connectors/gpu/p2p.py#L3), [wrapper registry](https://github.com/vllm-project/afd-plugin/blob/8d55ed11f4ce19beea6cc2c5bdeca1ced8946623/afd_plugin/__init__.py#L96), [runtime scope](https://github.com/vllm-project/afd-plugin/blob/8d55ed11f4ce19beea6cc2c5bdeca1ced8946623/README.md#L31)

### KernelFlume: weight nodes and KV owners

Weight nodes retain projections and FFN. Weightless attention nodes retain token-range KV. Queries go to historical owners. Fresh KV goes only to the live tail. Owners return attention partials for an exact online-softmax merge. Route updates occur between token steps. Prepared UCX endpoints and graph-external readiness/completion signals keep routing outside the fixed CUDA Graph.

Query-first dispatch overlaps historical attention with remaining K/V projection. Separate microbatches overlap remote attention with another microbatch's dense work. Its filled-pipeline maximum formula therefore requires independent ready microbatches. It is not a C1 same-token dependency bypass.

Real tests use Llama-3.1-8B on A6000 and cross-node H100 systems. Larger-model results are separate simulations. Tests of kernel pipelining use batches 16–128. The source does not establish Ouro behavior at concurrency 1–8. Its distinction from C02 is that projections remain on weight nodes while core attention moves. Its weight/state ownership split is nevertheless a close generic prior. [KernelFlume, Sections 3–6](https://arxiv.org/html/2606.29207v1)

### Helix: temporal layout changes on the same ranks

Helix partitions context during attention, combines local softmax partials, and reuses the same GPU pool for dense TP or MoE FFN. HOP-B overlaps communication across independent batch rows. The paper's main GB200/FP4 performance study is simulation. It is not live H100 BF16 evidence.

This is a C04 control, not proof that arbitrary unreduced residuals can cross nonlinear functions. Its workflow still includes input/layout exchanges and output reductions. The current implementation documentation limits declared support to MLA DeepSeek models on Blackwell and targets long inputs above 64K. It does not establish Ouro H100 support. [Helix paper, Sections 2–3](https://arxiv.org/html/2507.07120v1), [pinned support document](https://github.com/NVIDIA/TensorRT-LLM/blob/59eab99c898d1e3fdd78dc59373b1c360b7321ae/docs/source/features/helix.md#L18)

Current MLA code computes partial outputs and combines them at lines 828–878. Lines 845–858 handle per-token KV visibility for speculative groups. Lines 2130–2132 still invoke an input AllGather. An older feature matrix's speculation exclusion cannot establish a universal current-code limit. These inspected paths do not validate Ouro or every feature combination. [Pinned MLA path](https://github.com/NVIDIA/TensorRT-LLM/blob/59eab99c898d1e3fdd78dc59373b1c360b7321ae/tensorrt_llm/_torch/attention/mla.py#L828)

### PagedAttention: existing branch-prefix ownership

PagedAttention shares physical blocks across parallel samples and beams. Reference counts govern block lifetime. Copy-on-write protects partially shared blocks when a branch appends. Completed immutable prefixes need not become private copies. This is a generic C07 ownership control. It does not make shallow Ouro KV equivalent to verified deep KV, or provide distributed winner synchronization by itself. [PagedAttention, Sections 4.2–4.6](https://arxiv.org/html/2309.06180v1)

## Candidate-specific decision screen

The following entries are architectural inferences from the inspected mechanisms and the frozen candidate definitions.

| Candidate | Already covered by close priors | Remaining testable distinction |
|---|---|---|
| C01 | Independent ready-row batching, shared weight pointers, operation DAGs, stream overlap, and fixed weight/KV ownership. | Show that grouping different logical depths by physical operator creates a better ready frontier than an aliased conventional executor. Retain each row's depth KV and gate semantics. |
| C02 | Many attention owners feeding one stateless FFN batch, followed by split returns. | Show a tied-weight-specific residency or batching benefit beyond generic AFD. Same-P local batching and TP must remain controls. |
| C04 | Per-stage layouts, exact partial-attention reduction, collective overlap, and local fused consumers. | Name the exact removed collective and its first real consumer. Show where all remaining statistics, gathers, and reductions occur. Preserve BF16 cast points or test their effect. |
| C07 | Shared-prefix block ownership, branch-private appends, mixed-depth verification, and accept/prune protocols. | Show useful distributed placement that preserves shared-batch efficiency and reduces exposed latency after rollback, transfer, and commit costs. |

The [semantics audit](ouro_semantics_audit.md) records 16 query and 16 KV heads in both Base models. Thus a TP2/TP4 layout does not cross the KV-head count. Helix's excessive-TP KV-duplication argument is not this campaign's immediate regime.

## Structural cost screen for C02

These counts are `STRUCTURAL_ACCOUNTING`, not measured communication or HBM traffic.

For H=2048 and BF16, one activation row contains 4096 bytes. Sending a row to a remote MLP and returning its result carries at least 8192 payload bytes. This excludes metadata, padding, packing, and protocol traffic.

If every MLP is remote, fixed R4 requires 96 round trips for L24 or 192 for L48 per token's full layer trace. Payload totals are 786432 or 1572864 bytes per row. Batching B ready rows multiplies payload by B but can amortize message launches. Selective remote placement reduces the applicable count.

The payload can be small while the dependency cost remains large. The next attention step waits for the remote result. A claimed overlap needs another ready request, branch, or microbatch. Compare the measured remote round trip against local batched MLP and TP MLP on identical inputs. Do not replace this chain with a maximum merely because two GPUs exist.

## Evidence limits and time

This audit establishes close prior mechanisms, not a performance ranking on Ouro. Official-code availability remains unresolved for Nexus, FlowKV, and KernelFlume. Runtime and adapter validation remain `NOT_RUN` for all sources.

The recorded worker segment ran from 2026-10-19T10:29:56Z to 2026-10-19T10:38:56Z. The observed duration is 540 seconds, or 9 minutes 0 seconds. This is the accounting cutoff. These minutes overlap the coordinator's elapsed time. They do not add campaign elapsed hours. No GPU work, installation, runtime change, or deliberate wait occurred. Final artifact stamping after the cutoff is outside this duration.

