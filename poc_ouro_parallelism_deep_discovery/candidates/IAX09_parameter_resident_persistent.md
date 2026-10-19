# IAX09: Persistent execution with recurrent weight residency

## 1. Identity

ID: `IAX09`. Revision: `1`. Family: shared-weight execution and persistent kernels.

Status: `GENERIC_ONLY` for streamed persistent execution.
The full-core residency premise fails the capacity screen below.
This result does not reject all persistent implementations.

## 2. Mechanism

Keep a GPU worker kernel active across recurrent passes, and retain reusable BF16 weight tiles when storage permits.

## 3. Relevant Ouro property

The physical stack repeats four times with the same weight pointers.
The inputs, depth-indexed KV, and adaptive control remain different.
The inspected installed Ouro implementation contains the outer recurrent loop.
Its source hash is `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff`.

## 4. Target scope

Models: Base Ouro-1.4B and Ouro-2.6B.
Modes: fixed R4 and the original adaptive gate.
The adaptive operating point must preserve q=.5, minimum R2, maximum R4, and the pinned KV policy.
Initial scope: decode, system-wide C=1/2/4/8, and contexts 1K/4K/8K.
P2/P4 need live tests after resource authorization.
P8 remains projected.
Prefill needs separate kernel shapes and timing.

## 5. Baseline and exposed cost

Compare with validated same-P TP using graph replay and unchanged BF16 weights.
Also compare with a generic persistent task runtime using aliased physical weights.
The possible savings include exposed dispatch, operator boundaries, and selected weight reloads.
No saved cost received a new measurement in this audit.

## 6. Operation sequences

AS-IS: replay a ready recurrent-stage graph, update depth state, consume the gate, and repeat when required.
The CDB runner already caches recurrent graphs by batch shape, gate flag, and KV slot.

TO-BE: enqueue equivalent ready tasks inside persistent workers.
Each task waits for its actual inputs and the required KV version.
Workers stream unavailable weight tiles and retain only a capacity-feasible subset.
The original gate controls continuation after its required computation.

## 7. Legal DAG and ownership

The dependency chain remains:

`embedding -> block(r,0) -> ... -> block(r,L-1) -> loop_norm -> gate/continue -> block(r+1,0)`

The output selection and LM head follow the same mode-specific contract.
The next generated token requires the current output decision.
This diagram does not create parallel recurrent passes for one token.

| State | Owner | Readiness and lifetime | Cost that remains |
|---|---|---|---|
| BF16 weight shard | TP rank, optionally selected SM workers | Model-load completion, immutable revision | Stream all tiles that cannot remain resident |
| Hidden activation | Producer task, then its consumers | Required request/token/r/l event | Publication, reductions, and layout changes |
| Historical KV | Rank owning its attention heads | Committed request/r/l/position version | Attention reads for each logical depth |
| Current KV | The current attention task | Projection and position processing complete | Write, commit, and adaptive missing-depth policy |
| Gate state | Policy owner | Current gate and prior survival state ready | Matching arithmetic and consistent TP routing |
| Scratch and task state | Persistent worker/runtime | Current task version only | Queues, semaphores, barriers, descriptors, and temporary tensors |

## 8. Useful concurrency

Ready tiles, ready communication fragments, and independent requests can overlap.
Prefetch can overlap with computation when resources and dependencies permit.
The same-token R1/R2/R3/R4 chain remains serial.
No future hidden state or exit decision enters the schedule.
These concurrency sources also exist in generic task runtimes.

## 9. Exactness and quality

The intended class is `Exact computation graph`.
Different fused reductions can instead produce `Numerically reordered` execution.
Bitwise equality is not established.
Tests must compare logits, KV versions, gate values, logical exits, and free-running token outputs.
Use mode-matched correctness smoke tests before task-quality pilots.
No model-quality test ran for this candidate.

## 10. Memory

For d=2048 and intermediate width 5632, one block has `4*d*d + 3*d*5632` dense parameters.
Its BF16 dense weights occupy exactly 98 MiB.
Norms, embedding, LM head, and gate add storage.

Actual physical GPUs 4 and 6 each report 132 SMs and 52,428,800 L2 bytes.
The existing evidence is `outputs/components_base14_P2_graph_v1/components.json`.
This worker did not initialize a GPU.

NVIDIA documents 64K 32-bit registers and 228 KiB shared memory per Hopper SM.
Treating all registers, shared memory, and L2 as one free pool gives 112.390625 MiB per chip.
That pool does not exist as a unified allocation.
It also leaves no resources for activations, accumulators, control, or communication.
Thus, it deliberately favors the residency hypothesis.
[Hopper Tuning Guide, sections 1.4.1 and 1.4.2](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

| Model | P | Dense core per rank, MiB | Other layers between revisits, MiB | Full-core residence |
|---|---:|---:|---:|---|
| 1.4B, L24 | 2 | 1176 | 1127 | Exceeds generous capacity |
| 1.4B, L24 | 4 | 588 | 563.5 | Exceeds generous capacity |
| 1.4B, L24 | 8 | 294 | 281.75 | Exceeds generous capacity |
| 2.6B, L48 | 2 | 2352 | 2303 | Exceeds generous capacity |
| 2.6B, L48 | 4 | 1176 | 1151.5 | Exceeds generous capacity |
| 2.6B, L48 | 8 | 588 | 575.75 | Exceeds generous capacity |

These are `ANALYTIC_BOUND` storage values, not HBM traffic measurements.
The revisit column counts dense weights of other physical layers.
It is not a cache-miss theorem.
An impossible full-core allocation does not imply that every tile misses L2.

Full depth-indexed KV remains 786,432 bytes/token for 1.4B and 1,572,864 bytes/token for 2.6B before sharding.
Persistent launch does not compress this state.
Queue and graph storage require separate accounting.

## 11. Communication

The candidate does not algebraically remove TP reductions.
With replicated hidden states, each row-parallel output reduction handles `C*2048*2` logical payload bytes.
The actual ring or tree moves additional bytes.
The attention-output and MLP-output reductions still precede their nonlinear consumers.
Fine-grained overlap needs validated fragment readiness and a device-callable communication implementation.
P8 also needs a specified topology.
No eight-rank run or bandwidth projection ran here.

## 12. Critical path and added costs

Use `T_candidate = T_required_ops + T_exposed_comm + T_device_control + T_residency_cost - T_legal_overlap`.
All terms describe the resource-constrained critical path.
They are not a sum of overlapping event durations.

Compare this value with the best same-P graph baseline.
Charge queue atomics, descriptors, grid synchronization, occupancy loss, spills, and startup.
Charge dynamic KV indexing, allocation policy, gate decisions, and output publication.
Device workers cannot directly call a host PyTorch linear function as a device task.
Attention and communication need compatible implementations too.

The local FlashInfer TinyGEMM source reserves a 192 KiB weight/activation pipeline in its 16-stage configuration.
Other storage and register use remain additional.
Keeping permanent weights in this kernel would compete with that pipeline.
This source fact is not a measured occupancy result.

If an exposed removable fraction is f, complete removal gives the conditional screen `S <= 1/(1-f)`.
This audit did not measure f for the full Ouro stack.
This expression is not a speedup estimate.

## 13. Nearest work

Persistent RNNs already retain recurrent weights in registers across timesteps.
Its capacity requirements matter for this transfer.
[Diamos et al., ICML 2016, sections 3–4](https://proceedings.mlr.press/v48/diamos16.pdf).

MPK is a closer full-stack generic control.
It supplies SM-level tasks, persistent workers, device scheduling, and communication.
Its paged shared memory streams tensor chunks rather than storing a complete model on-chip.
Its BF16 evaluation is not an Ouro result.
[Cheng et al., arXiv:2512.22219v2, sections 3–6](https://arxiv.org/html/2512.22219v2).

The installed FlashInfer source already includes persistent cuTile GEMM scheduling.
Its K loop still loads weight tiles.
The separate persistent-attention template also does not retain the whole Ouro stack.

FLUTE targets lookup-table quantized weights.
Its main mechanism is outside this campaign's BF16 contract.
It is not an exact BF16 full-stack resident-weight baseline.
[Guo et al., arXiv:2407.10960v4](https://arxiv.org/abs/2407.10960v4).

## 14. Prior NO-GO evidence

This candidate does not reuse stale hidden states or change projection residuals.
It does not reopen the failed stale/delta family.
The prior full-stack cache-capacity warning directly applies.
The new audit distinguishes impossible full-core residence from possible streamed persistent work.

## 15. Cheap falsifier and stop rule

First, inspect the existing graph baseline and measure its exposed dispatch/control intervals.
Separate host gate waits from GPU kernel-boundary cost.
Use both model sizes and both depth modes.
Do not infer these intervals from warm single-operator latency.

Next, compare a repeated aliased graph with a loop-capable generic control.
Preserve the exact depth-indexed KV and gate arithmetic.
If generic control explains the benefit, stop the recurrence-specific claim.
If the optimistic exposed-time saving cannot cover persistent scheduling and kernel costs, stop engine work.
No minimum speedup threshold replaces uncertainty analysis.

The full-core premise already fails without a GPU probe.
A partial-residency reopening needs a named tile set, allocated bytes, and a legal reuse schedule.
It must compare with the same cache policy on the graph baseline.

## 16. Results and uncertainty

Capacity: `ANALYTIC_BOUND`, complete core does not fit under the stated BF16 representation.
Implementation overlap: `SOURCE_REPORTED`, generic competitors cover the proposed dispatch mechanism.
Latency, task quality, HBM traffic, and achieved residency: `NOT_RUN`.
No new model inference or GPU experiment ran in this worker segment.

## 17. Compatibility matrix

| Model and mode | P2 | P4 | P8 |
|---|---|---|---|
| 1.4B fixed R4 | Full-core residency fails, streaming untested | Same | Projected capacity only |
| 1.4B original adaptive | Same, gate/control untested | Same | Projected capacity only |
| 2.6B fixed R4 | Same | Same | Projected capacity only |
| 2.6B original adaptive | Same, no new adaptive validation | Same | Projected capacity only |

## 18. Loop-specific controls

Compare aliased logical unrolling with the compact physical loop.
Use identical kernels and the same weight pointers.
Then compare separate physical copies with identical values, if memory permits.
Report the changed memory pressure for that control.
R=1 can isolate dispatch engineering, but it is not a quality-equivalent model.
An aliased generic compiler that matches performance defeats the proposed novelty.

## 19. Complexity and next PoC

A full engine requires device-callable attention, dense kernels, TP communication, KV management, and exact gate handling.
This is not a small fusion patch.
No reliable implementation-time estimate follows from the current audit.
The next scoped task needs only a baseline trace and a byte/dispatch ledger.
Do not install or build MPK to establish that first fact.

## 20. Verdict

Reject full-core BF16 on-chip residence at P2/P4/P8 for the audited dimensions.
Retain streamed persistence only as generic engineering with unmeasured headroom.
The most informative next action is to measure exposed costs after graph replay.
This audit establishes no new parallel execution axis.
