# Parallel Track and breadth-first PP: transfer audit v1

Worker: `/root/independent_axes`.
Segment began: 2026-10-19 13:07:10 UTC.
This is a primary-source and code-transfer audit.
It contains no GPU, model-training, or new inference experiment.

## Outcome

Parallel Track Transformers change the model architecture and use newly pretrained weights.
Their synchronization reduction is not an exact execution rewrite of released dense Ouro.
Breadth-First Pipeline Parallelism changes the training schedule and layer placement.
Its word “looped” does not mean shared recurrent parameters.

The transferable breadth-first mechanism is a generic PP control.
Its weight-reconstruction advantage needs a matching nonresident-weight workload.
Its gradient-reduction advantage does not exist in forward-only inference.
Neither source supplies a new assumption that reopens C01, C04, or C14.
The candidate cards and earlier negative conclusions remain unchanged.

## 1. Primary sources and scope

| Source | Version and inspected locators |
|---|---|
| [Parallel Track Transformers](https://arxiv.org/html/2602.07306v1) | arXiv:2602.07306v1, 2026-02-07, sections 2.2, 3.1, 3.3, and 4 |
| [Apple project page](https://machinelearning.apple.com/research/parallel-track) | Official publication page, inspected 2026-10-19 |
| [Breadth-First Pipeline Parallelism](https://proceedings.mlsys.org/paper_files/paper/2023/file/24e845415c1486dd2d582a9d639237f9-Paper-mlsys2023.pdf) | Official MLSys 2023 PDF, sections 3.2, 4.1–4.3, 5, and appendices C/D |
| [PyTorch Looped BFS](https://github.com/pytorch/pytorch/blob/7baab7878598a7e3bfce1e982b2f3008fd89aef3/torch/distributed/pipelining/schedules.py#L3898) | Commit `7baab7878598a7e3bfce1e982b2f3008fd89aef3`, 2026-10-19 |

PDF page numbers below use one-based page indices.
The author code and generic framework code are distinct evidence sources.
This report does not label a generic implementation as the original paper's repository.
The structured observations preserve the official PDF and generic source SHA256 identities.

## 2. Parallel Track: architecture, not a drop-in schedule

PT uses independent smaller transformer tracks and periodic all-reduce fusion after groups of layers.
Section 2.2 defines this changed connectivity.
Section 3.1 uses eight tracks and GQA.
It pretrains 6B on 800B tokens and 13B/30B on 400B tokens.
These are different trained models, not the same dense checkpoint under a different schedule.

The serving comparison uses the 30B models on eight H100 GPUs.
Section 3.3 identifies the TensorRT-LLM implementation as an internal PT-enabled variant.
The paper also reports vLLM results and workload-dependent regressions.
Its model-quality tables do not establish unchanged-checkpoint equivalence.

The paper and Apple's page supplied no verified PT code link in this search.
Exact-title repository search returned zero repositories.
An AXLearn filename search also found no track-named implementation.
These lookup results cover only the inspected pages and queries.
They do not prove that no public implementation exists.
The PT runtime code therefore remains unaudited.

### Ouro transfer test

Released Ouro uses one dense residual stream, four norms per block, and depth-indexed KV.
Its MLP reads the residual after the completed attention update.
Its next block reads the completed MLP residual.
These dependencies do not become independent tracks because physical weights repeat across r.

Dividing a dense matrix into shards preserves its function only when the required partial results combine before their actual consumers.
Running several layers independently on those shards changes the function in general.
The existing [layout controls](layout_and_structured_residual_redteam.md) already reject separate nonlinear processing of unreduced additive contributions.

Likewise, running four independent copies of a pass from one hidden input does not reproduce Ouro's four recurrent passes.
Pass r+1 needs the actual output of pass r.
The source architecture removes cross-track dependencies by design.
It does not prove those dependencies absent in Ouro.

There is no supported weight conversion, original-gate preservation, or depth-KV mapping from PT to unchanged Ouro in the inspected evidence.
PT is relevant as a communication-aware trained architecture.
It is outside this campaign's no-new-training solution contract.
No PT percentage or eight-GPU result transfers to an Ouro speedup claim.

## 3. Breadth-first PP: what the paper actually changes

The paper targets training, not autoregressive serving.
“Looped” denotes distinct stages assigned cyclically across pipeline ranks, with several stages per rank.
Breadth-first ordering prioritizes a local stage across its microbatches.
Section 4.2 groups FSDP reconstruction and gradient work rather than repeating them per microbatch.
It also notes extra messages and requires more microbatches than pipeline ranks for the described transfer overlap.

The experiments use BERT training on 64 V100 GPUs.
Section 5 notes limitations in the baseline's communication overlap.
Appendix D describes a custom library and double-buffered weight reconstruction.
The inspected paper and author publication page do not expose that library's code.
The audit does not substitute training throughput for inference latency.

Relevant locators are PDF pages 4–6, 8–9, and 16–17.
The official PDF contains no matching “inference” text in the inspected text index.
That negative search is a scope check, not an inference impossibility theorem.

## 4. Actual generic code inspection

The inspected PyTorch source has SHA256:

`c0b7d3fd1f69474ff9272efefc00124d3c79e1570812dae9748339b62bafd973`.

`ScheduleLoopedBFS._calculate_single_rank_operations()` uses rank-strided logical stage indices.
For each local stage, it appends every forward microbatch.
It then appends reverse-stage backward work.
This code directly expresses stage placement and order, not equal parameter objects.

The generic runtime lowers actions into sends, receives, waits, unshards, reshards, and gradient operations.
Its public residency and lookahead controls already address staged parameter availability.
They do not identify equal recurrent weights or automatically preserve aliases in an Ouro builder.

The base class also exposes `eval()` for forward-only execution.
This audit did not check that entry point with the generated Looped BFS plan.
The class still generates backward actions in the inspected constructor path.
No claim of a framework failure or working autoregressive server follows from this source-only inspection.

Relevant code locators are lines 685–759, 3269–3485, and 3898–3991 at the pinned commit.
The [official API documentation](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html#torch.distributed.pipelining.schedules.ScheduleLoopedBFS)
also describes the earlier-stage scheduling rule.
This is a required generic control, not a candidate-specific implementation result.

## 5. What remains in forward-only Ouro inference

The following conclusions are this audit's transfer analysis.
They are not experiments reported in either paper.

### Weight scheduling

Inference removes gradients, optimizer updates, and training activation retention.
It does not automatically require repeated reconstruction of resident TP weights.
A breadth-first schedule cannot claim a reconstruction saving when the matched baseline performs no such operation.

One physical Ouro weight shard can serve all four passes on its owner.
An alias-aware logical unrolling can use that same shard.
Four copies in a deliberately materialized control would inflate memory and transfer costs.
That is not the strongest conventional baseline.

A genuinely nonresident-weight regime could make staged reconstruction relevant.
That requires a separately stated memory/admission constraint and actual transfer accounting.
Neither paper establishes that changed regime for the current two Ouro sizes and scoped workloads.
This audit therefore does not reopen C01 on weight streaming alone.

### Useful ready work

One non-speculative decode request supplies one recurrent chain.
Future pass inputs and later token inputs are not independent microbatches.
At system C=1, cyclic placement does not create a second ready request.
At C=2/4/8, independent active requests can occupy different stages.
Generic PP and CDB can use those same ready requests.

For a one-row-per-request transfer of the paper's microbatch condition, M cannot exceed ready active requests without another legal source of work.
The condition M>P therefore gives the following structural screen:

| P | C=1 | C=2 | C=4 | C=8 |
|---:|---|---|---|---|
| 2 | Below P | Equal to P | Greater than P | Greater than P |
| 4 | Below P | Below P | Equal to P | Greater than P |
| 8 | Below P | Below P | Below P | Equal to P |

This table does not predict utilization or speedup.
Packing requests into larger microbatches decreases M.
Adaptive exits can also shrink the ready frontier.
An original gate cannot reveal a future continuation decision before its actual loop output exists.

Do not substitute R=4 for four independent ready microbatches.
The paper's bubble formula is not a single-request latency theorem.
Its favorable schedule assumptions do not remove causal edges from the Ouro graph.

### Communication

Consider a simple physical partition into P consecutive chunks per pass.
Keep loop norm and gate at the final chunk owner.
The hidden path uses `R*(P-1)+(R-1)=R*P-1` inter-owner hops before final output processing.
At R=4 and D=2048, one full BF16 hidden payload contains 4096 bytes.

| P | Hidden hops per token | Aggregate logical payload |
|---:|---:|---:|
| 2 | 7 | 28 KiB |
| 4 | 15 | 60 KiB |
| 8 | 31 | 124 KiB |

This audit derives these layout counts without measuring traffic or timing.
They exclude embedding/head routing, metadata, control, synchronization, TP collectives, and any moved KV.
Finer stage granularity changes the hop count and message overhead.
It does not supply free communication overlap.
The same layout and aliases must remain available to the conventional comparison.

### State and numerics

KV stays indexed by request, logical depth, physical layer, position, and version.
Physical weight reuse does not merge those values.
Any stage migration must preserve actual cache ownership or charge its transfers.
An adaptive implementation also needs online gate routing and the pinned terminal-state propagation policy.

Neither a training schedule nor the generic stage API supplies those serving semantics automatically.
Different stage kernels or collective orders also need BF16 numerical qualification.
The current native asynchronous runtime remains the performance baseline.
No avoidable host barrier belongs only in the baseline.

## 6. Candidate reassessment

| Candidate | New source implication | Decision |
|---|---|---|
| C01 ready physical workers | Add Looped BFS and staged residency to the generic prior/control map | No new ready frontier or measured ownership optimum, no reopening |
| C04 loop-closed layout | PT's fewer fusion points require different architecture, not delayed exact Ouro reductions | No reopening |
| C14 causal chunk frontier | Breadth-first stage order is generic, same-prompt chunks still need their actual prefix KV | No reopening |

The conditional changed assumption would be a demonstrated nonresident-weight bottleneck or another previously unavailable legal ready frontier.
Neither follows from these papers.
Generic PP does not become a new axis because its name contains “looped”.

## 7. Limits, searches, and time

The [structured observations](../artifacts/parallel_track_breadth_first_observations_v1.json)
record source locators, code pins, bounded negative lookups, and tool limitations.
The worker did not locate a verified original PT runtime or BFS custom library for code inspection.
The concrete code inspection concerns PyTorch's generic implementation.

The host lacks `pdftotext` and the checked PDF Python modules.
No install followed.
The audit used the official PDF's indexed text and section locators.
It extracts no plotted performance value from a screenshot.

The segment began at 13:07:10 UTC on 2026-10-19.
Paper and code inspection reached the 13:12:03 UTC checkpoint.
Report synthesis and checks followed.
The final source and artifact checks ended at 13:15:56 UTC.
The evidence segment lasted 526 seconds.
The publication checkpoint was 13:17:12 UTC.
Count this worker interval once within concurrent campaign wall time.
Do not add overlapping worker intervals as extra elapsed research hours.
