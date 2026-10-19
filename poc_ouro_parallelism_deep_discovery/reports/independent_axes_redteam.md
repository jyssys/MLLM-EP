# Independent execution axes and skeptical screen

Status: `HYPOTHESES_AND_CPU_SCREENS_ONLY`.

This work found eight candidate mechanisms. It did not establish an Ouro speedup. It did not establish a qualifying new execution axis.

Three candidates start outside seed areas A–F: proof-driven decisions, state rematerialization, and redundant quorum computation. Their mechanisms have generic prior art or substantial novelty risks. A fourth candidate uses query-independent attention summaries. That candidate overlaps area F after closer analysis.

Six candidates examine changed execution units or dependency transformations. Two candidates provide practical controls for speculative continuation and tied operator workers. This allocation favors execution axes over placement tuning. It is not a claim about elapsed time per idea.

The companion artifact is [independent_candidate_seeds.json](../artifacts/independent_candidate_seeds.json). It contains the search queries, source versions, CPU commands, outputs, and candidate fields.

## Scope and actual activity

The worker started at `2026-10-19T09:55:29Z`, according to the first shell timestamp after timezone conversion. The first explicit UTC clock reading was `2026-10-19T09:55:58Z`. These timestamps describe this worker only. They do not establish ten hours of campaign activity.

This worker read the full campaign specification. It also read the reporting skill and its required writing-rules reference. It accessed public primary sources through web reads and public metadata requests. It ran three small CPU derivations. It did not load model weights or use a GPU.

Only the two assigned files belong to this worker. No prior report, running process, environment, package, commit, or remote branch changed.

## Outcome and cheapest next tests

No candidate merits a distributed implementation from this screen alone.

1. Preserve the finite-R4 numerical screen. The scalar counterexample already rejects a general claim of exact four-depth Jacobi acceleration.
2. Inspect baseline dispatch traces before building persistent workers. Test aliased unrolling and competent graph execution first.
3. Test head certificates on existing saved logits and hidden states, if such artifacts exist. Charge certificate work and all head reads.
4. Reject residual rematerialization as a novelty claim unless a new recurrence-specific difference survives the generic restoration control.

The most informative unresolved numerical fact is the number of charged solver rounds on real Ouro states. This fact remains `NOT_RUN`.

## Common semantics and scope

The pinned public Base configurations contain the following values. These are metadata checks, not locally loaded-runtime checks.

| Field | Ouro-1.4B | Ouro-2.6B |
|---|---:|---:|
| Public revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | `1ed04250da1a9936042725d302e81c8fa2ab5abd` |
| Physical layers L | 24 | 48 |
| Hidden width d | 2048 | 2048 |
| MLP width | 5632 | 5632 |
| Attention/KV heads | 16/16 | 16/16 |
| Recurrent passes R | 4 | 4 |
| Vocabulary | 49152 | 49152 |
| RMSNorm epsilon | 0.000001 | 0.000001 |
| Model dtype | BF16 | BF16 |

The [1.4B config](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json) and [2.6B config](https://huggingface.co/ByteDance/Ouro-2.6B/blob/1ed04250da1a9936042725d302e81c8fa2ab5abd/config.json) support these fields.

The inspected public implementation reuses a physical layer stack. It applies RMSNorm after each pass. It computes a linear exit gate from that normalized state. Its LM head has no bias. The public forward computes all configured passes. This observation does not establish the local adaptive runtime's behavior. [Ouro source, `OuroModel.forward` and `OuroForCausalLM.__init__`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py).

For one decode token, define `F_r(h, H_r)` as one complete physical stack plus its boundary normalization. `H_r` contains committed historical KV for depth r. Each evaluation also creates provisional current-token KV. The finite equations are `h_r = F_r(h_(r-1), H_r)` for r=1…4.

The maps share weights. They generally do not share historical KV. Each speculative evaluation requires a distinct scratch version. A final hidden state does not certify the scratch KV from an earlier, different input.

All eight cards target both Base sizes. Fixed R4 remains the required fixed mode. Adaptive cards require the unchanged original gate and pinned KV policy. Adaptive 2.6B remains `ADAPTIVE_2P6B_UNVALIDATED` in this worker's evidence.

The initial operating points are C=1/2/4/8 and contexts 1K/4K/8K. These are proposed tests, not measured cells. P2/P4 remain `NOT_RUN`. P8 remains `PROJECTED_ONLY`. A candidate can require a narrower regime, as stated below.

## Primary numerical audit

### DEER and quasi-DEER

[DEER v3](https://arxiv.org/html/2309.12252v3), Sections 3.4–3.5, evaluates trajectory residuals and Jacobians before an associative affine scan. Its dense formulation requires quadratic Jacobian storage and cubic scan work in state width. Those costs concern the solver, not only model evaluation. The paper does not establish Ouro inference performance.

[Gonzalez et al. v3](https://arxiv.org/html/2407.19115v3), Proposition 1, gives a stronger finite-chain statement. Undamped Newton reaches the discrete trajectory within at most T iterations in exact arithmetic. This guarantee does not require a contracting transition. It does not guarantee fewer than four expensive rounds for R4. Section 4.1 replaces dense Jacobians with diagonal approximations. Appendix B.1 warns that ordinary autodiff can still construct dense Jacobians before diagonal extraction. A direct diagonal derivative requires separate implementation. Numerical stability and iteration counts remain empirical concerns.

Thus, “not contractive” does not prove numerical parallelization impossible. Conversely, finite convergence does not prove useful latency reduction.

### ParaDiGMS

[ParaDiGMS v3](https://arxiv.org/html/2305.16317v3), Algorithm 1, evaluates several guessed diffusion states. It then computes a prefix update and advances a sliding window under an error threshold. Appendix C describes producer/consumer multi-GPU execution. Table 4 reports 1000-step DDPM with 2504 model evaluations and 50 parallel iterations. Its reported time falls from 50.0 s to 14.6 s, or 3.4×. The quality statistic is CLIP score on 1000 COCO2017 captions. These are source-reported diffusion results. They are not an exact-output guarantee, an R4 result, or an Ouro measurement.

The [authors' repository](https://github.com/AndyShih12/paradigms) recommends its multiprocessing path. It also states that large denoising-step counts suit the method. No code from that repository ran here.

### R4 cost screen

Let `F_1` denote one complete native pass on one GPU. Let `F_P` denote one pass under the best correct same-P baseline. These quantities are unknown here.

An idealized P-way trajectory round costs at least `ceil(4/P) * F_1` under one full-model replica per worker. This expression ignores contention and unequal work. It is not a measured runtime model.

The necessary inequality is:

```text
T_init + K * T_round + T_validate + T_KV_commit < 4 * F_P
```

For a one-pass exact warm-up, charge that pass to `T_init`. Three further exact-prefix rounds can still reproduce the original four-pass span. Parallel useful work does not remove this accounting.

At d=2048, four dense FP32 Jacobian blocks contain 67,108,864 bytes, or 64 MiB. Three coupling blocks alone contain 48 MiB. Storage alone does not reject single-token DEER.

An illustrative `4*d^3` expression equals 34,359,738,368 scalar multiplications. This is a scaling illustration, not an exact scan operation count. Jacobian formation adds cost beyond this expression.

For a prompt-wide dense state, `D=N*d`. Four dense FP32 blocks require 64 TiB at N=1024. They require 1024 TiB at N=4096. Causality creates zero blocks, but does not make a naive dense implementation feasible. These are structural allocations, not measured memory or a lower bound for all solvers.

### CPU counterexample

The executed scalar test used `F(x)=tanh(1.4*x+0.3)` and `h0=0.2`. It initialized all four unknown states to zero. It evaluated all depths with synchronous Jacobi updates. Each round corrects one additional prefix state in this example.

| Round | Exact prefix depths, tolerance 1e-12 | Absolute h4 error |
|---|---:|---:|
| 1 | 1 | 0.6199160840025685 |
| 2 | 2 | 0.30190960278910706 |
| 3 | 3 | 0.09246781421654404 |
| 4 | 4 | 0.0 |

This result illustrates exact-prefix propagation. It does not measure Ouro convergence. It rejects a general guarantee that four-depth Jacobi needs fewer than four rounds.

## Generic competitors that can remove apparent novelty

The following official interfaces were inspected live. Their documented availability does not establish local support, installation, correctness, or achieved performance.

| Official component | Documented mechanism | Required control |
|---|---|---|
| [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html), Section 4.2.4 | Device-evaluated IF/WHILE/SWITCH graphs | Compare a persistent loop with generic conditional execution |
| [NCCL Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html), documentation 2.32.3 | GPU-side communication and fused building blocks | Compare custom synchronization with supported generic primitives |
| [FlashInfer cascade](https://docs.flashinfer.ai/tutorials/recursive_attention.html), documentation 0.7.2 | Attention states merge across KV subsets | Compare partial historical attention with generic split-K/CP |
| [FlashInfer communication](https://docs.flashinfer.ai/api/comm.html), documentation 0.7.2 | Fused reduction interfaces and parallel communication primitives | Compare custom layout/collective fusion with existing interfaces |
| [vLLM parallelism](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/) | TP, PP, and hybrid deployment | Compare a cyclic worker graph with aliased logical unrolling |
| [TensorRT-LLM disaggregation](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html) | Separate prefill/decode pools and KV transfer overlap across independent requests | Do not claim request-independent KV transfer as an Ouro discovery |

CUDA conditional bodies reside on one device. A multi-device protocol still requires explicit messages and completion rules. NCCL CFT examples list SM_100 and CUDA 13.3 requirements. They cannot establish H100 capability. [NCCL CFT requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cft.html).

## Candidate register

Each verdict concerns this screen only. `CONDITIONAL_MECHANISM` does not mean that a positive speedup or acceptable quality exists.

| ID | Mechanism family | Outside seeds | Initial category | Current verdict |
|---|---|---|---|---|
| IAX01 | Zeroth-order finite-trajectory correction | No, E | Potential new axis | `NO_GENERAL_EXACT_HEADROOM` |
| IAX02 | Matrix-free Newton correction | No, E | Potential new axis | `CONDITIONAL_MECHANISM` |
| IAX03 | Proof-driven token decision frontier | Yes | Potential decision axis | `CONDITIONAL_MECHANISM`, generic risk |
| IAX04 | Historical-state rematerialization frontier | Yes | Representation/execution axis | `GENERIC_ONLY` at present |
| IAX05 | Redundant quorum linear computation | Yes | Redundancy axis | `GENERIC_ONLY`, BF16 risk |
| IAX06 | Query-independent attention summaries | Independent origin, F overlap | Approximate feature axis | `CONDITIONAL_MECHANISM`, generic risk |
| IAX07 | Exact-input continuation beside gate work | No, D/A | Speculative continuation | `GENERIC_ONLY` at present |
| IAX08 | Shared physical-operator task workers | No, A/B | Recurrence hybrid | `CONDITIONAL_MECHANISM`, generic risk |

### IAX01 — Synchronous trajectory correction

Mechanism: evaluate several depth maps on versioned guesses, then correct the finite trajectory in parallel.

- Legal useful concurrency: different `F_r` evaluations can execute together. Their inputs are guesses, not predecessor outputs from the current round.
- Readiness: immutable weights and historical depth-indexed KV must exist before each evaluation. Each evaluation creates separate current-token KV.
- AS-IS: `F1 -> F2 -> F3 -> F4 -> head`. TO-BE: initialize four states, evaluate all residuals, refine, check, and commit matching state versions.
- Exactness: complete convergence can recover the intended graph in real arithmetic. A finite residual threshold is approximate inference without a stronger certificate.
- Baseline: native BF16 R4 and the best correct TP2/TP4 implementation. Compare identical context, precision, and state semantics.
- Saved path: fewer than four charged parallel rounds could reduce recurrence span. The number of evaluations may increase.
- Added costs: replicated weights, solver state, historical KV access, scratch KV, inter-depth state exchange, convergence checks, and final consistent KV generation.
- Stop test: reject exact acceleration if the first decisive real-state pilot needs four full rounds plus overhead. Reject earlier if an optimistic bound already exceeds baseline time.
- Generic/unrolled control: an aliased unrolled graph admits the same residual formulation. Shared weights alone do not make this axis novel.
- Prior relation: this is not the old stale-state execution with unverified outputs. It requires correction and charges correction work. No old failed quality arm ran again.
- Scope: C1 is possible in principle. Adaptive R2/R3 reduces the available span further. P2/P4 and both model sizes remain unmeasured.
- Next action: inspect one token's complete current-KV scratch contract before any quality run. Implementation burden is high. Do not start distributed code first.

### IAX02 — Matrix-free Newton–Krylov trajectory correction

Mechanism: solve the finite trajectory residual using Jacobian–vector products instead of materialized dense Jacobians.

- Legal useful concurrency: depth-wise residual and derivative products can run on separate workers for a shared solver vector.
- Readiness: each product requires frozen historical KV and a versioned state guess. Autodiff must not mutate authoritative current-token KV.
- AS-IS: four sequential forward passes. TO-BE: initialize, evaluate residuals, solve a correction system, update states, check, and commit.
- Exactness: this targets the finite R4 function, not an infinite-depth equilibrium. Inexact linear solves and early termination create approximation.
- Baseline: correct TP2/TP4 and IAX01 under the same memory budget. Use a single-GPU solver as an algorithmic control.
- Saved path: a small number of effective Newton rounds might correct multiple depths. A Krylov inner solve creates its own serial reductions.
- Added costs: derivative products through complete stacks, Krylov vectors, reductions, line search or damping, residual reevaluation, and consistent KV output.
- Sharp test: first count JVP/VJP calls and global reductions. Reject if an optimistic charged schedule cannot beat four matched passes.
- Generic/unrolled control: Newton–Krylov applies to an ordinary finite feed-forward graph. Tied weights do not remove derivative work.
- Prior relation: exact current inputs drive each derivative product. No learned coarse model or stale activation replacement is allowed.
- Scope: both sizes share d=2048, but 2.6B doubles physical stack length. Adaptive gate branches require a pinned solver/commit rule.
- Next action: derive an identity-preconditioned correction on a CPU nonlinear toy. Do not equate this proposal with the audited diagonal quasi-DEER implementation.

This candidate remains a bounded theoretical option. The parent should allocate a strict screening budget before approving model autodiff.

For the four-depth correction, write the block Jacobian as `J=I-B`, where B is strictly lower triangular. Then `J^-1=I+B+B^2+B^3`. A matrix-free implementation still needs nested derivative products. Symbolic function composition does not shorten those products. A logarithmic scan needs an efficient representation for composed linear maps. Without that representation, memory savings do not establish reduced span.

### IAX03 — Certified token decisions from partial head results

Mechanism: certify the greedy winner before every output-head tile completes, then release the next token without an unverified guess.

- Legal useful concurrency: independent head tiles produce logit contributions. A certifier checks bounds while remaining tiles execute.
- Readiness: use the exact final normalized hidden state for the selected depth. Do not invent an uncomputed hidden-coordinate norm.
- AS-IS: finish all logits, reduce argmax, publish token. TO-BE: accumulate bounded partial logits, certify a winner, publish token, and finish required state work.
- Exactness: a valid interval proof can preserve greedy output. Sampling needs a separate distribution-preserving method. BF16 rounding needs an explicit error envelope.
- Baseline: fused vocab-parallel head plus argmax under the same resources. Keep recurrent passes and the original gate unchanged.
- Saved path: only the unfinished head tail can disappear from token issuance. No pass becomes independent because of this certificate.
- Added costs: row-norm metadata, partial sums, bound updates, winner checks, synchronization, possible duplicate weight reads, and scratch buffers.
- Sharp test: apply certificates to saved real head inputs. Reject if certificates arrive only after most tiles or cost more than the removed tail.
- Generic/unrolled control: any autoregressive decoder with a linear head admits this method. Recurrence-specific value is currently absent.
- Prior relation: the method does not reuse a stale hidden state. It preserves the selected depth and all current-token KV.
- Scope: both fixed and adaptive modes can use the same final-head contract. C1 is relevant. P2/P4 remains `NOT_RUN`.
- Next action: test certificates on existing artifacts before collecting new model states. Implementation burden is moderate, but proof and baseline costs dominate risk.

For known coordinates S, let `p_v = sum_(i in S) W_vi*h_i`. The missing contribution obeys `|z_v-p_v| <= ||W_v,U||2 * ||h_U||2` in real arithmetic. A winner a is certified when its lower bound exceeds every other upper bound.

The CPU toy certified a large-margin winner after 1/4 features. A near-tie toy required 4/4 features. These are constructed algebra checks, not an acceptance-rate estimate.

RMSNorm's positive common denominator cancels from real-arithmetic greedy ranking after scale folding. This shortcut does not prove equality to rounded BF16 normalization. Epsilon also prevents an unconditional claim of scale homogeneity. The primary test should retain the native normalized head input.

### IAX04 — Parallel historical-KV rematerialization

Mechanism: store per-layer, per-depth projection inputs and regenerate historical K/V chunks on auxiliary workers before attention needs them.

- Legal useful concurrency: historical positions are known. Their K/V projections can execute independently once the matching stored residuals exist.
- Readiness: each stored residual needs its own `(request, token, r, l)` identity. Workers need K/V weights, positions, normalization rules, and rotation state.
- AS-IS: retain all K/V and read them during decode. TO-BE: retain projection inputs, regenerate chunks, publish exact-version K/V, consume attention, and discard temporary chunks.
- Exactness: this preserves the intended algebra. Changed GEMM shapes can change floating-point results. Input normalization, RoPE, and cache policy require checks.
- Baseline: native full KV, CP with stationary KV, and a generic restoration engine with the same checkpoint representation.
- Saved path: capacity savings might avoid paging. Regeneration could replace exposed transfer time. It does not automatically accelerate an in-memory B1 attention read.
- Added costs: historical projection GEMMs, temporary KV, weight residency, chunk publication, attention-state merging, and synchronization.
- Sharp test: use the arithmetic screen below. Reject a latency claim if even optimistic reprojection exceeds avoided traffic time.
- Generic/unrolled control: identical state rematerialization works for untied transformers. No recurrence-specific advantage survives this screen yet.
- Prior relation: this changes stored state, not whole-state prediction. It does not reopen the failed distributed-attention operator claim.
- Scope: memory-constrained long contexts are the relevant regime. C1 short context is a poor first target. Both sizes require distinct capacity accounting.
- Next action: retain as a generic comparator unless a new reuse fact appears. Implementation burden is high and should not precede a favorable roofline screen.

The [Residual Stream paper v1](https://arxiv.org/html/2603.19664v1), Sections 3.1–3.2, derives K/V from the matching layer residual. It also proposes more aggressive checkpoint storage. Those claims do not establish that one Ouro residual reconstructs every depth without replay. The conservative design here stores separate residuals at every logical layer/depth. It does not inherit the paper's end-to-end performance or bitwise claims.

[CacheFlow v2](https://arxiv.org/abs/2604.25080v2) is a strong generic restoration lead. Its abstract describes token-, layer-, and GPU-level restoration with boundary activations. This worker checked metadata and the abstract only. A full method audit remains necessary before treating it as an executable comparator.

For d=Hkv*dhead=2048, the stored input has half the elements of K+V. This is structural storage accounting. It is not measured traffic reduction or speedup.

| Size | Context N | Full R4 KV | Per-logical-layer residuals | Added full-history reprojection per decode token |
|---|---:|---:|---:|---:|
| 1.4B, L24 | 1024 | 0.75 GiB | 0.375 GiB | 1.649267441664 TFLOP |
| 1.4B, L24 | 4096 | 3.0 GiB | 1.5 GiB | 6.597069766656 TFLOP |
| 1.4B, L24 | 8192 | 6.0 GiB | 3.0 GiB | 13.194139533312 TFLOP |
| 1.4B, L24 | 16384 | 12.0 GiB | 6.0 GiB | 26.388279066624 TFLOP |
| 2.6B, L48 | 1024 | 1.5 GiB | 0.75 GiB | 3.298534883328 TFLOP |
| 2.6B, L48 | 4096 | 6.0 GiB | 3.0 GiB | 13.194139533312 TFLOP |
| 2.6B, L48 | 8192 | 12.0 GiB | 6.0 GiB | 26.388279066624 TFLOP |
| 2.6B, L48 | 16384 | 24.0 GiB | 12.0 GiB | 52.776558133248 TFLOP |

The formulas are `KV=2*L*R*N*d*2` bytes and `residual=L*R*N*d*2` bytes. Reprojection uses `4*L*R*N*d^2` FLOP. The FLOP convention counts a multiply and an add separately. It excludes RoPE, attention, traffic, and synchronization. Sliding regeneration can reduce temporary capacity, not the full-history arithmetic shown here.

### IAX05 — Redundant quorum projections

Mechanism: execute redundant linear transforms and recover a required output after a sufficient worker subset completes.

- Legal useful concurrency: several encoded transforms of one ready activation can execute concurrently. The decoder waits for a valid quorum.
- Readiness: workers need encoded weight blocks and the current exact activation. Historical/current KV remains authoritative outside this linear subproblem.
- AS-IS: every TP shard must finish. TO-BE: redundant workers compute coded shards, then a decoder reconstructs the linear result before any nonlinearity.
- Exactness: the code identity is exact in real arithmetic. Encoded BF16 weights and finite-precision decoding do not generally reproduce native BF16 results.
- Baseline: competent TP and uncoded redundant replication under the same P2/P4 compute and memory budgets.
- Saved path: the slowest required rank can leave the critical path when redundant capacity overcomes a real straggler.
- Added costs: encoded storage, redundant GEMV work, input fan-out, output collection, decoding, and error control.
- Sharp test: reject if ordinary baseline traces show no removable straggler tail larger than those added costs. Do not introduce artificial GPU contention.
- Generic/unrolled control: static weights also permit pre-encoding in an ordinary transformer. R4 does not create a unique amortization advantage.
- Prior relation: no learned weight update is proposed. Exactness still fails if the auxiliary encoding changes the effective function through rounding.
- Scope: P4 offers more redundancy choices than P2. Current low-C dedicated-device inference lacks evidence of the required straggler regime.
- Next action: leave this as `GENERIC_ONLY`. Reopen only after a real tail-latency cause and a valid BF16 error contract appear.

[Short-Dot](https://arxiv.org/abs/1704.05181v1), by Dutta, Cadambe, and Grover, already studies coded short dot products and quorum recovery. Its cluster result is not an Ouro or H100 result.

### IAX06 — Query-independent attention summaries

Mechanism: maintain fixed, non-learned feature summaries of historical K/V and distribute query-time feature contractions.

- Legal useful concurrency: once a historical K/V version commits, workers can update independent feature summaries before the next query exists.
- Readiness: every summary belongs to one request, layer, depth, head, feature-map seed, and committed prefix. The current token adds its own contribution.
- AS-IS: query every historical K/V. TO-BE: update `S=sum phi(k)*v^T` and `z=sum phi(k)`, then evaluate `phi(q)^T*S / phi(q)^T*z`.
- Exactness: finite feature maps generally approximate the softmax kernel. No exactness claim applies. Random features must remain fixed and untrained.
- Baseline: exact native attention, generic CP/split-K, and the same feature method on a logically unrolled tied graph.
- Saved path: long-context attention reads can shrink to feature-state reads. Background summary updates can use slack after KV publication.
- Added costs: feature construction, summary updates, state replication/sharding, query broadcasts, output reductions, numerical stabilization, and approximation error.
- Sharp test: first compare real Ouro attention outputs at fixed seeds and declared feature sizes. Reject if the required feature state approaches native KV cost.
- Generic/unrolled control: feature attention is not recurrence-specific. Separate depth summaries preserve the R factor rather than eliminate it.
- Prior relation: this does not reuse old q, k, v, or hidden values as current values. It changes the attention kernel and needs new quality tests.
- Scope: only a new long-context profile can establish enough attention share. The old reported 7.3% fraction does not authorize a new speed claim.
- Next action: mark quality `NOT_RUN`. Require free-running fixed/adaptive task pilots before distributed code. Implementation burden is high.

[Performers v4](https://arxiv.org/abs/2009.14794v4) provides the nearest feature-kernel mechanism. Its feature approximation guarantees do not certify a frozen Ouro replacement. This worker checked the abstract and source metadata, not its full experimental protocol.

Changed-assumption statement: the old mechanism replaced a full stage with stale or delta state. This candidate retains current q/k/v computation and changes only the historical attention representation. The first test must measure attention error and causal KV updates. Budget at most 30 minutes for algebra and existing-artifact error checks. Stop before GPU work if those checks fail. This is a different approximation, not a claim that the old quality failure vanished.

A later model-quality pilot requires a separate budget and resource approval. No reopening pilot ran in this worker.

### IAX07 — Gate computation beside exact-input continuation

Mechanism: start the next pass from a completed exact state while another worker evaluates the original exit gate.

- Legal useful concurrency: `gate(h_r)` and `F_(r+1)(h_r,H_(r+1))` share a ready input. The continuation is provisional until the gate decides.
- Readiness: exact normalized h_r and correct historical KV must exist. Continuation KV uses scratch, never committed cache.
- AS-IS: finish h_r, evaluate gate, then continue. TO-BE: evaluate gate and continuation concurrently, then discard or retain continuation state.
- Exactness: this can preserve gate decisions and model outputs with a correct commit protocol. Shape-dependent BF16 numerical differences still require checks.
- Baseline: GPU-resident gate evaluation and generic conditional execution. Do not compare only with a host round-trip baseline.
- Saved path: at most the exposed gate/control delay on continuing tokens. The next recurrent pass does not become independent of h_r.
- Added costs: activation transfer, duplicated residency, wasted continuation work, cancellation boundaries, and scratch lifetime management.
- Sharp test: measure gate/control delay first. Reject if an optimistic overlap saves less than the activation/message overhead.
- Generic/unrolled control: conditional-depth networks admit identical speculation. Fixed R4 has no gate-wait requirement to remove.
- Prior relation: no stale input or trained gate appears. This does not reuse the old predictive-stage mechanism.
- Scope: `MODE_SPECIFIC` for adaptive inference unless a separate fixed-mode cost exists. It cannot satisfy both-mode headroom alone.
- Next action: keep this as a baseline diagnostic, not a finalist. Implementation burden is moderate and unnecessary without exposed control delay.

### IAX08 — Shared physical-operator task workers

Mechanism: dispatch ready tasks from different logical depths to workers that retain one copy of a physical operator's weights.

- Legal useful concurrency: ready tasks from independent requests or causally ready prompt chunks can share a GEMM batch. Same-token depths remain sequential.
- Readiness: each task needs its predecessor activation, correct depth KV, resident weight shard, layout, and gate decision when applicable.
- AS-IS: request-local stack execution or ordinary PP. TO-BE: publish ready operator tasks, group compatible tasks, execute, and route outputs.
- Exactness: the intended graph remains unchanged. Grouped batch shapes and communication reductions require numerical checks.
- Baseline: DP+CDB, grouped GEMM, graph capture, tied-aware PP, and logically unrolled graphs with aliased weights.
- Saved path: possible weight traffic, memory replicas, dispatch cost, or bubbles. None of these savings is measured here.
- Added costs: queueing, activation messages, packing, result routing, state-owner access, startup/drain, and constrained batching opportunities.
- Sharp test: enumerate the ready frontier at C1/2/4/8. Reject if useful batches require unavailable predecessor activations or remove the latency benefit through waiting.
- Generic/unrolled control: an unrolled executor with aliases can schedule the same nodes. Physical sharing must change a measured optimum to support a recurrence-hybrid claim.
- Prior relation: this does not reopen the old prefill-wavefront novelty claim. An ordinary causal-PP DAG remains ordinary causal PP.
- Scope: no C1 recurrence concurrency is assumed. Both modes may support multi-request grouping, but adaptive gates reduce predictability.
- Next action: audit a tiny legal trace against the aliased control. Implementation burden is high. Require a concrete saved cost before runtime development.

## State ownership contract

| State | Creation and owner | Readers | Mutability and commit rule |
|---|---|---|---|
| Original weights | Load/setup, baseline shard or worker | Ready operator tasks | Immutable, pinned model revision |
| Historical KV | Earlier committed token, authoritative state owner | Matching request/r/l attention or restoration tasks | Immutable after commit |
| Current KV | Current evaluation, scratch owner | Only consumers of that scratch version | Commit only after correct input and policy validation |
| Hidden state | Predecessor operator, task owner | Named successor, gate, or head | Immutable after publication |
| Solver vectors | Solver round, solver worker | Residual and correction tasks | Mutable only between named rounds |
| Head intervals | Partial head tile, certifier | Greedy decision protocol | Publish a token only after valid bounds |
| Residual checkpoints | Matching layer/depth projection input | Restoration workers | Immutable, depth-indexed |
| Encoded weights | Fixed algebraic setup | Quorum workers | Immutable, but numerical error remains unresolved |
| Attention summaries | Committed K/V append | Query feature workers | Prefix-versioned, never shared across unequal depths |

For IAX01/IAX02, copying scratch KV is not free. For IAX04, one residual per token does not replace all logical-layer checkpoints without replay. For IAX05, a nonlinear successor must wait for a complete decoded linear result. For IAX06, an updated summary cannot include an uncommitted branch.

## Scope matrix

`U` means unknown performance and quality. `N` means not run. No cell denotes measured headroom.

| Candidate | 1.4B fixed | 1.4B adaptive | 2.6B fixed | 2.6B adaptive | P2/P4 | P8 | Single request |
|---|---|---|---|---|---|---|---|
| IAX01 | U | U, shorter span | U | Unvalidated runtime | N/N | Projected only | Possible solver work, no established gain |
| IAX02 | U | U, gate contract missing | U | Unvalidated runtime | N/N | Projected only | Possible solver work, no established gain |
| IAX03 | U | U | U | Unvalidated runtime | N/N | Projected only | Head-only opportunity |
| IAX04 | U | U, policy-specific | U | Unvalidated runtime | N/N | Projected only | Capacity regime only at present |
| IAX05 | U | U | U | Unvalidated runtime | N/N | Projected only | Requires real straggler headroom |
| IAX06 | U, approximate | U, approximate | U, approximate | Unvalidated runtime | N/N | Projected only | Long-context attention hypothesis |
| IAX07 | No fixed gate-wait headroom | U | No fixed gate-wait headroom | Unvalidated runtime | N/N | Projected only | Adaptive-only overlap |
| IAX08 | U | U | U | Unvalidated runtime | N/N | Projected only | No new exact depth frontier |

## Rejection boundaries and evidence gaps

The scalar test rejects a universal exact-Jacobi acceleration claim. It does not reject every numerical solver on Ouro.

Generic prior art currently defeats a recurrence-specific novelty claim for residual restoration and coded quorums. This conclusion does not prove those methods lack practical value.

No candidate used a newly trained component. No candidate changed q, output length, precision, or the KV policy to manufacture a comparison. No quality evaluation or distributed timing ran.

Public-source metadata and local-runtime behavior remain separate. In particular, public gate code does not establish online adaptive 2.6B correctness.

The report's independent-origin count concerns how the candidate mechanisms were formulated. It does not prove that each mechanism lies outside every broad reading of areas A–F. The overlap risks appear explicitly in the cards.

The next coordinator decision should select a small number of discriminating CPU or existing-artifact tests. GPU implementation should require a legal schedule and a favorable charged cost screen first.

Final validation checkpoint: `2026-10-19T10:10:41Z`. The recorded worker span to that checkpoint was 912 seconds. This span is wall time, not measured active CPU time. Timestamp annotations followed that checkpoint.
