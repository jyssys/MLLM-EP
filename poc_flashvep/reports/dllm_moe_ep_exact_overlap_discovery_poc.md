# LLaDA2.0-Flash true-EP4 exact-overlap discovery PoC

## Final status: `NO-OVERLAP-SIGNAL`

The study found legal exact concurrency, but no candidate exceeds the
predeclared 5% **credible direct request-E2E** gate. The strongest same-request
analytical ceiling is only **3.50%**: even making the entire HumanEval shared
expert free cannot save more. The strongest live result is an exact
`dispatch(next) || expert(current) || combine(previous)` pipeline across
independent physical waves. Its measured sampled-shape service/throughput upper
bound is **4.35% GSM8K / 4.59% HumanEval**; even assigning the best observed
sample to every layer and wave yields only **4.84% / 6.73%**. These are not
single-request latency gains because refinement wave `t+1` depends on the
decision from `t`, and the strongest static substrate already executes one full
ready pool as a single physical wave.

Stage 2 also fails to create a dLLM-specific successor. Refinement changes the
communication fraction modestly, and cross-state overlap efficiency varies with
physical shape, but no phase-aware schedule demonstrably improves on the generic
complete-wave pipeline. The result therefore closes this exact-overlap branch
on the tested four-H100 substrate rather than recommending a production
scheduler or custom kernel.

## Working contract and evidence boundary

- Contract: `poc_flashvep/reports/dllm_moe_ep_exact_overlap_discovery_poc_spec.md`
- Isolated project branch: `flashvep/dllm-moe-ep-exact-overlap-discovery-poc`
- Runtime worktree base: dInfer commit `8c9561f5badf185b0ddcf38fc4753e3b2a49af88`
- Model: `/home/esjung/models/LLaDA2.0-flash-744c3f8`
- Hardware: physical H100 GPUs 0--3 only; all six links are NV18
- Runtime: dense TP4 + routed true EP4, 256 routed experts, 64 contiguous
  experts/rank, DeepEP normal dispatch, owner-rank fused BF16 expert, reverse
  combine, rank-local shared expert, final TP all-gather
- Controlled requests: bounded 32-request GSM8K and HumanEval pools
- Static configuration: submitted batch 32, mini-batch 32, generation/block
  length 32, threshold 0.9, BF16

The live overlap probe is deliberately post-layer and read-only. It captures
real hidden states, routes, expert ownership, resident weights, and attention
inputs, then replays isolated exact operations after the production layer. It
does not alter decoding. This gives trustworthy dependency/contention evidence,
but a replay-derived independent-wave upper is not called request latency.

GPU performance counters were unavailable because Nsight Systems returned
`ERR_NVGPUCTRPERM`; Nsight Compute was absent. TensorCore/SM/HBM/NVLink labels
are therefore source-derived resource classes. All quantitative claims use
same-rank CUDA-event timing and observed concurrent slowdown, never fabricated
counter values or cross-rank timestamps.

## Strongest baseline

| dataset | clean independent restarts | median BCT / pool wall | range | median throughput | NFE | peak HBM |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 3 | **5.8449 s** | 5.7624--6.0893 s | 455.25 token/s | 66 | 77,693 MiB |
| HumanEval | 3 | **7.2621 s** | 7.1177--7.4404 s | 455.65 token/s | 86 | 80,945 MiB |

Every fresh clean and instrumented run reproduced the same answer hash within
its dataset. The bounded anchor scores remain 5/32 GSM8K and 6/32 HumanEval,
matching the previously validated identical outputs. These short-generation
scores are substrate anchors, not a claim about model quality. Replayed expert
outputs have minimum cosine **0.99999988** and maximum relative L2 **0**.

Clean observer-derived component shares establish the economic mass:

| dataset | router | dispatch | routed expert | combine | shared expert | total MoE |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 6.43% | 12.07% | 29.47% | 6.41% | 2.98% | 59.52% |
| HumanEval | 7.22% | 13.32% | 29.55% | 6.66% | 3.50% | 62.81% |

Dispatch and routed expert have enough raw mass to matter; shared expert does
not. This motivated live pairwise testing rather than assuming all MoE stages
could overlap ideally.

## Stage 1 — exact opportunity atlas

### Runtime dependency fact

The current `forward_deepep` contract is:

```text
router/top-k
  -> DeepEP dispatch issue + wait
  -> owner fused expert (whole wave)
  -> DeepEP combine issue + wait
  -> shared expert
  -> routed + shared residual
  -> TP all-gather
```

The source exposes asynchronous DeepEP events, but the model wrapper waits at
each boundary. The wait is not automatically removable: downstream tensors
usually depend on communication completion. Legal exceptions are:

- shared-expert compute after the common hidden state versus routed EP work;
- local source branches versus remote branch dispatch after routing;
- operations belonging to distinct already-prepared physical waves/requests;
- partial expert-output combine only if the fused expert produces externally
  visible completed tiles, which the current whole-wave invocation does not.

The final TP all-gather and same-request next refinement remain strict under the
current tensor and decoding contracts.

### Pairwise results

Each state used five warmups and at least 30 measured repetitions on each rank.
The table reports critical-rank medians across the six sampled states unless a
range is more informative.

| exact pair | dependency/scope | measured saving | interpretation |
|---|---|---:|---|
| `dispatch || attention` | independent requests/replay | **-0.050 to -0.130 ms** | contention; slower in all six states |
| `remote dispatch || local expert` | same source wave | **-0.020 to -0.092 ms** | algebraically independent, economically negative |
| `dispatch || shared` | same request | negative in all sampled states | DeepEP communication competes with shared GEMM |
| `combine || shared` | same request | -0.008 to **+0.146 ms** | sometimes useful, but whole shared ceiling is <=3.50% E2E |
| `dispatch(next) || expert(current)` | independent waves | **+0.023 to +0.043 ms** | repeatable but service-only and small |
| `expert(current) || combine(previous)` | independent waves | **+0.033 to +0.070 ms** | repeatable exact pipeline component |
| three-stage D/E/C | independent waves | **+0.106 to +0.186 ms/layer** | strongest live exact diagnostic |

DeepEP is not pure “free NVLink work”: its dispatch kernels use communication
SMs and the memory system. That is why apparently complementary attention,
shared GEMM, and local expert work slow down when paired with dispatch. The
three-stage pipeline works only because independent complete waves offer enough
issue/progress slack; pairwise savings cannot simply be added.

### Scope-correct economic map

| candidate | optimistic or measured upper | credible direct request E2E | gate |
|---|---:|---:|---|
| routed EP + shared expert | 2.98% GSM8K / 3.50% HumanEval perfect ceiling | same | KILL <5% and prior-art adjacent |
| remote dispatch + local expert | negative live saving | 0% | KILL contention |
| communication + attention | negative/mixed live saving | 0% | KILL contention |
| independent-wave three-stage pipeline | 4.35% / 4.59% sampled-median service upper | **0%** in one request | CHARACTERIZATION only |
| extreme best saving at every wave | 4.84% / 6.73% service upper | **0%** in one request | optimistic boundary, not a result |
| same-wave early combine/tile send | not supported by current fused output contract | unknown, no valid prototype | prior-art collision/high implementation cost |

The independent-wave projection multiplies a measured per-layer saving by the
31 MoE layers and observed denoising-wave count, then divides by clean pool
wall. It assumes the independent waves already exist and ignores scheduler
overhead, so it is an upper bound on service throughput, not request latency.

## Stage 2 — refinement/timestep-aware overlap

Stage 2 was entered because Stage 1 had no 8% candidate. The exhaustive
same-substrate trace shows:

| dataset | phase | dispatch | expert | combine | communication fraction of D+E+C | whole MoE |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | early | 0.420 ms | 1.028 ms | 0.250 ms | 39.47% | 1.961 ms |
| GSM8K | middle | 0.392 ms | 1.036 ms | 0.214 ms | 36.90% | 1.904 ms |
| GSM8K | late | 0.312 ms | 0.787 ms | 0.146 ms | 36.81% | 1.545 ms |
| HumanEval | early | 0.415 ms | 1.081 ms | 0.270 ms | 38.79% | 1.998 ms |
| HumanEval | middle | 0.396 ms | 0.902 ms | 0.187 ms | 39.25% | 1.742 ms |
| HumanEval | late | 0.318 ms | 0.622 ms | 0.137 ms | 42.29% | 1.510 ms |

HumanEval becomes modestly more communication-heavy late; GSM8K does not.
Physical wave shape is not a pure function of normalized phase: sampled source
rows were 248/40/8 for GSM8K waves 0/30/60 and 144/200/16 for HumanEval waves
0/40/80. Ready-pool drain, block progress, and request completion therefore
confound any rule based on `early/middle/late` alone.

All ordered cross-state pairs were replayed. `combine(A) || expert(B)` saved a
median **0.0598 ms** GSM8K and **0.0759 ms** HumanEval across pairings;
`dispatch(A) || expert(B)` saved **0.0536/0.0608 ms**. Shape affects overlap
efficiency, but the best cross-state two-stage savings (0.120/0.109 ms) do not
beat the generic same-state three-stage maximum, and no complete phase-aware
pipeline has a demonstrated positive increment over the generic policy.

Next-timestep exact preparation also fails the mass gate. DeepEP buffers and
workspaces are persistent, while routing, send offsets, token counts, and
destination metadata depend on the next hidden state. Pre-posting only static
descriptors would remove no measured recurring >=5% component. Same-request
`t+1` backbone execution was not pursued because it is speculative rather than
exact.

## Prior-art attack

The remaining generic implementation space is crowded:

- [DeepEP](https://github.com/deepseek-ai/DeepEP) already exposes asynchronous
  event-overlap mechanisms and persistent buffers.
- SGLang's current
  [batch-overlap operation strategy](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/batch_overlap/operations_strategy.py)
  splits dispatch/combine phases and schedules shared-expert work between
  communication halves on supported paths.
- [COMET](https://proceedings.mlsys.org/paper_files/paper/2025/hash/e27ea0cd50b798ff8942caf9203f0992-Abstract-Conference.html)
  integrates fine-grained communication/computation overlap for MoE.
- [StreamEP](https://github.com/evolutionaryscale/StreamEP) implements tile- and
  token-granular producer/consumer readiness rather than whole-wave barriers.
- Recent [tile signaling](https://arxiv.org/abs/2607.19539) and
  [X-Stage](https://arxiv.org/abs/2607.23264) directly cover streaming return
  communication and inter-wave expert pipelines.
- [DICE](https://arxiv.org/abs/2411.16786) is diffusion-MoE adjacent but changes
  freshness/staleness semantics; it is not an exact-overlap baseline.
- [Sangam](https://arxiv.org/abs/2607.04206) and recent
  [dLLM serving work](https://arxiv.org/abs/2608.23807) make refinement-aware
  service multiplexing adjacent, but the present measurements do not show an
  incremental phase-aware advantage.

Thus the only positive live direction is generic, below the economic gate, not
direct request latency, and overlaps mature mechanisms. A new paper would need
either a larger unexploited same-request dependency break or a demonstrable
dLLM phase advantage; neither appears here.

## Required questions

1. **Which sub-operations are actually independent?** Shared expert versus
   routed EP after the common hidden state; local branches versus remote
   dispatch after routing; and D/E/C work from independent prepared waves.
2. **Which use complementary resources?** Communication versus expert/shared/
   attention is nominally NVLink-versus-TensorCore, but live contention shows
   shared SM/HBM pressure. Only complete-wave expert versus another wave's
   communication is robustly positive.
3. **Strongest pairwise exact overlap?** `combine || shared` reaches 0.146 ms in
   one state, but its entire same-request E2E ceiling is 3.50%. Across
   independent states, combine/expert reaches 0.120 ms.
4. **Does contention destroy ideal overlap?** Yes. Dispatch plus attention,
   shared expert, or local expert is consistently slower than serial.
5. **Is there a legal D/E/C pipeline?** Yes, across independent prepared waves;
   it saves 0.106--0.186 ms/layer. It is not legal across causally sequential
   refinement steps of one request.
6. **Can local expert overlap remote dispatch?** Algebraically yes; economically
   no on the tested H100/DeepEP shapes (0.020--0.092 ms regression).
7. **Can shared expert overlap routed EP?** Some combine/shared overlap works,
   but dispatch/shared is negative and the full shared mass is <3.51% E2E.
8. **Does cross-request communication-compute overlap work?** Expert versus
   another wave's communication works weakly; dispatch versus attention does
   not. The best complete pipeline remains below 5% sampled-median service
   headroom.
9. **Does refinement phase change resource intensity?** Modestly on HumanEval,
   not robustly across both datasets; physical ready-pool shape matters more
   than the phase label.
10. **Do complementary phases beat blind pairing?** No complete phase-aware
    schedule beats the generic three-stage pipeline in the measured atlas.
11. **Is there a phase-dependent best overlap policy?** Shape-dependent pair
    efficiency exists, but no positive request-level policy increment was
    demonstrated.
12. **Can next-step input-independent preparation matter?** No recurring >=5%
    component exists; static state is persistent and expensive metadata is
    next-input dependent.
13. **Strongest credible request-level oracle?** **3.50%**, the full shared
    expert ceiling on HumanEval.
14. **Generic or dLLM-specific?** The only positive signal is generic
    independent-wave MoE pipelining. No dLLM-specific successor survives.
15. **Difference from DeepEP/StreamEP/X-Stage/TBO?** This PoC measures current
    whole-wave contention and scope-correct economics; it does not introduce a
    new mechanism beyond their asynchronous, tile-streaming, or inter-wave
    overlap spaces.

## Reproducibility, limitations, and decision

The repository contains the immutable summary tables, 16 required figures,
analysis scripts, a fail-closed invariant test, per-run GPU accounting, and a
format-patch for the dInfer diagnostic changes. The decisive limitations are
explicit: representative layer 16 rather than a full per-layer live sweep,
three physical shapes per dataset, no privileged GPU counters, and a post-layer
diagnostic rather than integrated scheduling. Those limitations could change
fine detail, but they do not rescue the decision: the direct ceiling is already
<5%, the strongest service upper remains <8% even under an extreme assumption,
and the surviving mechanisms face direct prior-art adjacency.

Total task-owned live GPU wall was **1,268.477 s** including two fixed setup
failures, or **1.4094 four-GPU hours**. Passing evidence runs account for
1,140.562 s / 1.2673 four-GPU hours.

**Final decision: `NO-OVERLAP-SIGNAL`.** Do not implement a production overlap
scheduler or custom kernel from this branch.
