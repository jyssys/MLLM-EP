# Looped-MoE Recurrent-Aware Expert Parallelism PoC

**File name:** `looped_moe_ep_coalescing_replication_poc_spec.md`  
**Deployment location:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_coalescing_replication_poc_spec.md`  
**Project root:** `/home/esjung/MLLM-EP-github`  
**Output directory:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_coalescing_replication/`  
**Date:** 2026-10-26  
**Status:** Discovery-first, accuracy-preserving, TRUE-EP4/EP8 PoC. No efficacy claims before measurement.

## 0. Executive decision and scope

We test exactly **three official, full-size research checkpoints** throughout every principal PoC stage:

1. **IFM Ouro MoE 8.78B** — `IFM/LoopedLM-P1-moe-ouro-500b`.
2. **IFM Huginn MoE 8.78B** — `IFM/LoopedLM-P1-moe-huginn-500b`.
3. **Foil-1** — `ShourenWSR/how-to-loop-moe`, subfolder `Foil-1`.

No small surrogate, extra baseline model, substitution of checkpoints, fine-tuning, weight quantization, precision reduction, expert skipping, output reuse, layer/loop parallelism, or algorithmic accuracy trade-off is allowed. The goal is **loop-specific, multi-GPU expert parallelism** rather than generic loop pipeline parallelism.

- **Track A (primary): Cross-Loop Expert Coalescing.** Coalesce ready expert work from independent requests at *different recurrent passes of the same physical MoE layer and same physical expert* into a single, correctly unpacked GPU expert batch. Measure whether fewer small GEMMs / communication launches improve end-to-end latency without blocking the requests too long.
- **Track B (primary): Loop-Conditioned Expert Replication and Placement.** Exploit pass-conditioned routing/load statistics to pre-place a *small* number of weight-identical expert replicas and steer exact router-selected tokens toward a favorable resident replica without weight migration at each pass. Measure rank imbalance, remote traffic, and actual E2E improvement.
- **Track C0 (bounded diagnostic only): Cross-request compute–communication overlap and next-pass lookahead feasibility.** Trace dependency-respecting overlap opportunities between already-ready work from different requests; separately measure non-authoritative next-pass route predictability. Do not implement predictive routing or change model outputs as part of this PoC.

Required conditions: **TRUE EP4 and TRUE EP8**, on GPUs **0–7**, across all three models; batches/concurrency **B1, B2, B4** with limited B8 stress tests if affordable. Focus on low-request latency. End-to-end, tail latency, TTFT, TPOT, decode-step and prefill measurements must be real GPU measurements where labeled "measured".

**Research question:** Does recurrent sharing of physical expert weights across distinct logical depths afford EP execution opportunities beyond ordinary layer-wise MoE batching and load balancing?

## 1. Scientific hypotheses, causal attribution, and hard constraints

### H-A: cross-loop physical expert identity

Let each expert-work item be

`w = (request_id, token_id, logical_pass, physical_layer, expert_id, expert_input, routing_weight, ready_timestamp)`.

Two items may share one expert GEMM **only if** `physical_layer`, `expert_id`, model checkpoint, and weight version are identical and **both are ready independently**. Their `logical_pass` may differ. Inputs/outputs are NOT reusable: the combined batched GEMM computes each input separately using the unchanged expert weights. Gather and scatter must exactly preserve each request's top-k routing weights, token order and semantics. A request's own future recurrent pass is never ready until its actual dependency chain completes.

Hypothesis: multiple low-concurrency requests whose loop positions diverge create a sufficient supply of same-physical-expert work to offset queueing, packing, and communication overhead. **B1 has no intrinsic cross-loop request coalescing**; a B1 advantage from this mechanism is not expected and must not be fabricated.

### H-B: loop-conditioned expert demand

For physical MoE layer `p`, pass `r`, expert `e` and rank `g`, collect exact token routing counts and real elapsed execution costs, not only expert IDs. A pass-conditioned load/placement policy may outperform an equally provisioned pass-agnostic generic EP policy. Replica copies must occupy real measured GPU memory; their construction/initialization costs must be disclosed. Default is static replicas allocated **once before inference** with no dynamic BF16 expert migration within the loop. The model router always chooses logical experts; the runtime merely chooses which identical resident replica executes an assignment.

We require the **incremental improvement attributable to pass information**, over generic best-of-baselines under equal memory, to avoid mislabeling generic EP optimization as novel. No claim of gain if the loop-conditioned predictor does not improve upon a pass-agnostic dynamic placement baseline.

### H-C0: allowable overlap and lookahead

**Exact overlap:** an already-computed router for request B/loop s may dispatch B's ready activations while request A/loop r computes its ready experts, subject to GPU link/SM contention and dependency synchronization. B and A must be independent requests or demonstrably independent work units. Stream-level overlap is only useful if measured critical path actually shrinks.

**Not exact:** one request's next pass has not produced its input hidden state, so the next pass's exact routing IDs, dispatch payloads and expert outputs are unavailable. A previous-pass routing pattern is at best a prediction. Pre-gated MoE in prior literature can train a gate and alter selection semantics; that is out of scope here.

Lookahead diagnostics may measure next-pass top-k recall, predicted expert set size, precision/recall, bytes that *could* be proactively staged as immutable metadata or replica weights, and penalty of false positives, but **must never use predictions to alter ground-truth routing or send nonexistent activation payloads**. No claimed speedup from a predictor without a validated full dependency-respecting execution path.

### Quality constraints

- Maintain checkpoint BF16 weights, existing recurrence count and topology, original top-k router and mixing weights, attention/MLP definitions and model outputs within reproducible native-vs-native numeric tolerance.
- For the same backend, seed, deterministic greedy decoding, and inputs, require token agreement wherever attainable; report first divergence and max/mean logits difference even if nondeterministic kernels prevent bitwise equality. Never silently accept accuracy regression.
- Track native fidelity separately from distributed backend fidelity (earlier IFM backend differences are a known concern). Do not attribute backend-port numerical differences to A/B algorithms.
- If true distributed baseline cannot match native within a defensible numerical range, mark `BLOCKED_BACKEND_FIDELITY`, not GO.
- No fake EP, single-process trace simulation described as distributed run, unsynchronized CUDA-clock arithmetic, unrealistic concurrent inputs, free replicas, or Oracle passing future decisions into an online algorithm.

## 2. Official sources, exact model properties and loader rules

### IFM Ouro and Huginn

- Ouro: `https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-500b`
- Huginn: `https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b`
- Official runtime: `https://github.com/ifm-ai/xllm-loop`
- Official models require the `xllm.paper_part1.native_inference.load_native_model` loader; **not** `transformers.AutoModel.from_pretrained`. Their native format has a verified artifact manifest. Use official `hf download --local-dir` and do not break manifest integrity or symlink restrictions.
- The official native path requires CUDA + FlashAttention 3. Check supported device architecture, drivers, PyTorch/CUDA and build before claiming baseline. If FA3 cannot run on available GPUs, assess a correctness-preserving supported backend/port and label changed kernels and fidelity; never falsely call it the untouched native baseline.
- Published architecture for these Part I models: 28 routed physical layers, 25 experts/physical layer, top-2. Ouro recurrence = **28 shared layers × 4 passes**; Huginn = **8 prelude + 12 shared layers × 8 passes + 8 coda**. Confirm these against the downloaded artifacts; only Huginn's recurrent core is eligible for cross-pass *same-physical-layer* merging; prelude/coda are unshared one-off calls and remain in E2E timing.
- Parameter counts: Ouro 8,780,889,600; Huginn 8,783,251,968. Weight BF16; ~17.6 GB decimal total before runtime overhead. Do not assume available VRAM/replica budget without actual measurement.

### Foil-1

- Official checkpoint/card: `https://huggingface.co/ShourenWSR/how-to-loop-moe`, `subfolder="Foil-1"`; code: `https://github.com/SR-A-W/how-to-loop-moe`.
- **563.2M parameters** (not 553.7M); one shared-core routed physical MoE layer reused **16 passes**, **64 experts** in that core, **top-2** routing. Two additional unlooped MoE layers (one before, one after) with 8 experts each. Attention weights are untied by pass; Expert and Router weights are shared for the core.
- Use `AutoModelForCausalLM.from_pretrained(..., subfolder="Foil-1", trust_remote_code=True, dtype=torch.bfloat16)` and official tokenizer. Follow official model's input/padding restrictions. Do not use model-provided `loss` indiscriminately as a pure LM NLL: official labels must be shifted, and returned loss may include router auxiliaries. Evaluate cross entropy on logits against correct shifted tokens.

### General

- Record exact Git commits, model revision/SHA, config JSON, routing top-k, pass count, tokens, Python, CUDA, NCCL and `nvidia-smi` in manifests.
- Read and reuse any existing project instrumentation, including the last looped expert-headroom report, while keeping this a fresh, independent PoC with separate raw artifacts.
- Respect checkpoint license and storage quotas. Never overwrite another PoC's report or checkpoint files.

## 3. Hardware and execution matrix

### Mandatory topology discovery

Run and log before changing code:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id --format=csv
nvidia-smi topo -m
nvidia-smi pmon -c 1
python - <<'PY'
import torch
print('torch',torch.__version__,'cuda',torch.version.cuda)
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i,p.name,p.total_memory,p.major,p.minor)
PY
```

Check GPU ownership with `nvidia-smi` and do not kill other users' jobs. Determine whether NVLink, PCIe switch, NUMA locality, NCCL P2P/GPUDirect assumptions are valid; capture link/device contention and usable HBM. Adjust launcher to topology, not arbitrary assumed H100s. If EP8 ranks require reshuffling, record exact mapping and justification.

### Deployment plan

- **EP4 group G0**: CUDA visible IDs `[0,1,2,3]`; **EP4 group G1**: `[4,5,6,7]`. In a process configured with `CUDA_VISIBLE_DEVICES=4,5,6,7`, local ranks 0..3 refer to physical GPU 4..7; log both IDs.
- Use two EP4 groups concurrently for *independent functional tests, data capture and oracle analysis* if contention is tolerable. **Run sequentially / alone for final latency** and repeat on G0/G1 where feasible to estimate topology variance.
- **EP8**: GPUs `[0,1,2,3,4,5,6,7]` exclusively, with 8 processes and one real EP group. Must test actual communication and completion, not just construct groups.
- **EP1**: one physical GPU when native checkpoint/runtime fits. This is a reference, not an EP4/EP8 substitute. If EP1 impossible because of VRAM/runtime, document why, and still implement TRUE EP4/EP8.
- For Foil-1, EP8 might be much slower than native due to small work units; still profile it. Avoid interpreting "EP8 worse" as a correctness failure.
- All three models must have EP4 **and** EP8 entries. If a backend cannot support one configuration after a bounded honest implementation effort, report the specific blocker (OOM, unsupported FlashAttention, distributed MoE gap, etc.), never fill with simulated speedup.

### Full experiment matrix

| Axis | Mandatory values | Notes |
|---|---|---|
| Model | Ouro 8.78B, Huginn 8.78B, Foil-1 | Exactly these three |
| EP degree | TRUE EP4, TRUE EP8 | EP1 optional reference |
| Concurrency | B1, B2, B4 | B8 diagnostic only, not headline |
| Prompt length | 128 or 256, 1024, 2048 tokens | Add 4096 only if supported and memory safe; Foil max context 4096 |
| Output length | 32 and 128 greedy tokens | Record stop/EOS, prefill and decode separately |
| Request arrival | synchronized, staggered, asynchronous low-QPS trace | Never assume all loops ready together |
| Placement | native/static balanced EP; generic replication; loop-conditioned | Equal memory/replica budget |
| Coalescing | off; same-pass generic only; cross-pass exact | Identical base executor |
| Window | no delay, small bounded windows | e.g. 0/50/100/250/500 microseconds; adapt to actual measured work cost |
| Replicas | 0, tiny feasible count, feasible medium count | Real resident weight bytes; no moving weights per pass |
| Phase | prefill, decode, E2E | report P50/P90/P95 and mean |

The matrix is factorial only for cost-feasible cells: stage discovery then narrow to measured promising cells; nevertheless every **model × EP4/EP8 × B1/B2/B4** must have verified baselines and A/B oracle records. Use the same prompt/token streams across techniques within each cell. Do not pad or silently truncate Foil inputs beyond supported maximum.

## 4. Runtime baseline: TRUE expert parallelism and fidelity

**P0 deliverable:** a maintained baseline capable of actually executing the model's unchanged MoE experts across EP ranks with Dispatch → Expert Compute → Combine at every applicable MoE call.

1. Inspect current repository, previous EP code, official loaders, architecture, and GPU capabilities. Reuse existing validated distributed kernels if applicable, but confirm their tensor semantics against these new models.
2. Implement/adapt a thin distributed EP layer at the native MoE call boundary. Preserve non-expert residual, attention, norms, loops, prelude and coda. Define rank-local routing and expert mapping explicitly; do not confuse expert replicas with unique logical experts.
3. Partition experts stably by physical layer and expert ID across ranks; top-2 routes to the selected logical experts. Preserve router scores, token multiplicity, combine weights, and token order. Ensure correctly functioning E2E generation, not merely expert microbenchmarks.
4. Verify EP4 and EP8 functionality for each model on short prompts; compare native logits, greedy tokens and NLL for matched inputs. Distinguish backend port numerical error from optimization error.
5. Verify repeated physical-layer identity across loops via a runtime weight-storage identity check. Ensure Foil's **untied attention is not coalesced** and Huginn's prelude/coda remain distinct physical layers.
6. Capture TTFT, TPOT, request E2E and per-pass per-layer per-rank dispatch/compute/combine, plus all-to-all bytes and launches. Use real CUDA event/stream profiling with explicit causal sync; compare GPU events within a stream without bogus global timestamp assumptions. NVTX / Nsight Systems are recommended for representative subtraces.
7. Establish warmup and steady-state reproducibility (≥10 warmups plus ≥20 timed repetitions per key cell if feasible; target ≥30, document deviations). Control GPU clocks, thermal/P2P interference and CPU contention. Record confidence intervals/bootstrap for paired speedup comparisons.

**Fatal P0 blockers:** missing TRUE EP communication; missing full recurrent execution; intolerable fidelity divergence; incorrectly reusing output tensors across passes; FP16/INT8 substituted without explicit user authorization. Mark stages blocked rather than deriving measured speedups from surrogates.

## 5. Instrumentation and traces

For each **routed physical MoE invocation**, collect:

- `model_id, request_id, sequence_id, token_id, prefill_or_decode, logical_pass, physical_layer, call_index, is_recurrent_core, arrival_time, ready_time, completion_time`;
- `expert_id, topk_slot, router_weight, assigned_rank, source_rank, selected_replica_rank, expert_input_shape, tokens_per_expert, bytes_in, bytes_out`;
- per-rank `dispatch_pack_us, dispatch_comm_us, expert_queue_wait_us, gemm_us, combine_comm_us, combine_unpack_us, synchronization_us, idle_us`;
- `cuda_stream, dependency_event, kernel_launch_count, NCCL/DeepEP version, achieved_link_bandwidth, GPU HBM peak/resident/repl_bytes`;
- distinct time to *become ready* vs time *chosen to execute*, which is essential to avoid inventing overlap or coalescing from a post-hoc trace.

Store light aggregate CSVs and independently inspectable trace samples. Instrument with sampling where full per-token traces would materially slow execution. Record instrumentation overhead separately; final latency must use minimized tracing.

Trace validity: prefill and decode have different token counts and GEMM dimensions; record true expert inputs, request progress and sequence-ready events; do not infer readiness from loop index alone.

## 6. Track A: Cross-Loop Expert Coalescing

### A1. Opportunity analysis

Compute, for every cell, the number and volume of ready work items with identical `(model, physical_layer, expert_id)` but **different logical_pass** belonging to independent requests. Report:

- fraction of expert executions/activations eligible to merge within **causally valid queue windows**;
- fraction of merges that are **strictly cross-pass**, not ordinary same-pass batching;
- per-expert batch-size distributions before/after, GEMM occupancy, kernel launch count, packing bytes, communication invocation count;
- cross-loop pass skew probability by B1/B2/B4, short vs long prompt, and synchronized vs staggered arrivals;
- number of cases where queues would have to block critical-path requests and added wait distributions.

No "merge everything across all passes" clairvoyant oracle. An *unconstrained offline oracle* is allowed as a separate ceiling, but cannot justify GO; the *readiness-constrained oracle* is the primary ceiling.

### A2. Counterfactual ablations

1. Static standard EP (no waiting/coalescing).
2. Generic coalescing limited to same physical layer **and same loop/pass** with equivalent wait budget.
3. Exact cross-pass coalescing using identical physical expert and two independent ready requests.
4. Optional weighted case: combine A with B only after both independent mechanisms show headroom.

Matched arrival schedules and same memory footprint. Distinguish **additional cross-pass gain** from generic batching/continuous batching and from service-time reductions due to request reordering alone.

### A3. Cost model and practical minimum implementation

For candidate queue group `Q_e,p`, measure or fit:

`T_coal = T_wait + T_pack + T_dispatch + T_gemm(batch(Q)) + T_combine + T_unpack + T_extra_sync`.

Compare against actual baseline schedule's *critical-path time*, not sum of independent operation durations. Include EP collective participation requirements and possible NCCL collective order deadlocks: global process groups must use a well-defined consistent collective ordering, or correctly isolated communicators. A coalescing plan that cannot preserve distributed collective order is infeasible.

Implement the minimum ready-queue executor only if readiness-constrained oracle reaches the stage gate; support request metadata, duplicate top-2 entries, exact scatter/aggregation and fallback to standard EP on low work. No cross-pass output cache and no `wait forever for a partner` policy.

## 7. Track B: Loop-Conditioned Expert Replication

### B1. Diagnose loop-specific stragglers

Measure the distribution of expert work *across GPU ranks* for each physical layer/pass, with exact token counts, GEMM time, dispatch time and observed idle time. Compare:

- per-pass expert/rank work coefficient of variation (CV) and max/mean, not only routing entropy;
- loop-dependent hotspot changes: `top_e(r,p)` vs `top_e(r+1,p)` and demand shift metrics;
- request-conditioned vs prompt-conditioned hotspot stability (train/eval split to avoid oracle leakage);
- baseline EP4 vs EP8: fewer tokens per local expert often make communication and kernel overhead more severe at EP8;
- fixed non-recurrent prelude/coda cost (Huginn/Foil) that this mechanism cannot optimize.

### B2. Policies and memory parity

Mandatory policies under the *same GPU memory cap*:

- `B0`: static balanced expert partition, no replicas;
- `B1`: generic, pass-agnostic replica selection (global hot experts / measured generic EP policy);
- `B2`: loop-conditioned replicas and exact router-driven per-invocation replica selection;
- `B3`: a non-clairvoyant runtime policy based on pass ID, past counts and current exact router output (not future hidden state), if B2 favorable;
- `B_omniscient`: offline per-invocation fully informed oracle, clearly marked **unimplementable upper bound**.

The meaningful contribution is `B2/B3 vs B1` under equal replica memory, not `B2 vs B0` alone. Model static load/memory trade-offs and actual replica startup cost. No self-claimed gains from unaccounted replica copies; report base weights and peak HBM.

### B3. Benefit and cost decomposition

Candidate savings: maximum per-rank expert critical-path time and dispatch/combine topology (locality and destination selection). Candidate costs: synchronization, replicated weight VRAM, replica selection/scheduling, increased communication due to token origin or fragmented small GEMMs, initialization amortization, and potential new slow ranks. **Do not assume that a workload may be reassigned to a rank where the weight is absent.**

Measure both best static mapping and heterogeneous but fixed mapping; assess whether pass-conditioned choices remain good under different prompts, B1/B2/B4, and at EP4/EP8. Keep Expert GEMM full BF16 and output exact up to baseline numerical variability.

### B4. Minimum implementation gate

Only implement real EP replica-aware token dispatch after oracle analysis verifies incremental loop-conditioned headroom, a feasible memory budget, and predictable online policy. Correctness first. Validate actual live E2E performance over static/generic replication; report any regressions (especially B1).

## 8. Track C0: optional diagnostic for overlap / pre-gate hypothesis

This is **measurement and bounded feasibility testing**, not a third standalone optimization build.

1. Create a causal DAG from real event trace: attention/router readiness → dispatch → expert work → combine → residual → downstream layers/passes; relate different requests' independent ready nodes.
2. Record actual concurrency between (a) dispatch B / compute A, (b) combine B / compute A, (c) two independent requests at different passes and the same physical expert. Analyze GPU SM–DMA–NVLink/NCCL contention. A timeline with two streams is not proof of effective overlap.
3. Report `available_overlap_us`, `achieved_overlap_us`, and `critical_path_saved_us`, and compare to a realistic replay schedule with bandwidth/compute constraints and collective ordering. No mathematical full-overlap assumption unless justified by operator-level observations.
4. Route lookahead: from already available preceding-pass or preceding-layer **observable** features, compute expert-set precision/recall, top-k recall, false-positive replica byte cost. Separate same-request prediction from exact other-request ready work. Do **not** predispatch next-pass activations or alter next-pass authoritative gate.
5. A full implementation of learned/pre-gate prediction requires an explicit separate research decision, accuracy qualification, and prior-work novelty analysis; out of scope here.

## 9. Oracle hierarchy (do not conflate with speedup)

For every track and every model/EP degree:

- **O0 Analytical absolute upper bound**: idealized full workload redistribution or unconstrained grouping; ignores overhead and readiness. Useful only to reject weak ideas.
- **O1 Readiness-/memory-constrained oracle**: preserves DAG order, ready times, rank capacity, expert residency, queue wait budget, device communication resources, collective order, and actual measured GEMM scaling; uses future knowledge only for offline assignment, labeled oracle.
- **O2 Practical online simulation/replay**: policy uses only data available at decision time; replay validated against measured baseline traces and bounded model error.
- **M Real distributed implementation**: correctness-verified running GPU code on TRUE EP4/EP8, and only level eligible to claim *measured E2E speedup*.

Require an error bar / sensitivity range for oracle predictions; use directly measured microbench curves for expert GEMMs of different token counts and for NCCL / dispatch at EP4 and EP8 (including launch latency and bandwidth). If simulation reproduces baseline with large error, do not trust its claimed headroom.

Use Amdahl sanity checks: if affected fraction `f` is small and local speedup is `s`, `S_total <= 1 / ((1-f)+f/s)` without extra overhead. Never infer overall headroom from just a reduction of dispatched experts or all-to-all calls.

## 10. Quality and experimental design

- Evaluation corpus: reusable fixed public samples spanning GSM8K, MATH-500, ordinary text, and long prompts; at least 32 examples/domain for engineering sanity where time permits; larger matched samples if near GO. A tiny 8-example pilot cannot establish downstream quality preservation.
- Log exact prompt IDs, input token counts, truncation choices, output tokens, tokenization/normalization differences and EOS outcomes. Deterministic greedy generation, temperature zero, fixed seeds and fixed recurrence policy.
- Quality checks: native vs distributed logits/CE, then distributed EP baseline vs A and vs B, and downstream answer accuracy where runtimes are faithful. **Use Foil's proper next-token CE, not its auxiliary-inclusive training loss.**
- Performance: mean, P50/P90/P95 request E2E, TTFT, TPOT, end-to-end tokens/s, decode latency/token, prefill time, CPU overhead, per-rank idle, dispatch/compute/combine wall costs, GPU HBM and replica bytes. Distinguish latency improvement from aggregate throughput improvement.
- Each measurement labeled with real GPU topology, concurrency, EP group, model, precision and backend. Different runtimes are not compared by raw absolute times; comparisons stay within each model/backend.
- Include statistical confidence (bootstrap paired 95% CI), warmups, trials, inter-run variance, and conditions causing latency regressions.

## 11. Stage plan and stop rules

### P0 — native/TRUE EP feasibility (mandatory all models, EP4 + EP8)

Environment, official checkpoints, fidelity bridge, actual distributed Dispatch/Combine and profiling baseline. Each cell clearly `PASS`, `FAIL`, `BLOCKED` or `NOT_RUN` with reason. Do not mark an unexecuted checkpoint `PASS`.

### P1 — Discovery (mandatory all models, EP4 + EP8)

Collect realistic B1/B2/B4 traces, recurrent core vs one-off layer distinction, per-rank imbalance, ready cross-loop expert pairs, causal DAG and (limited) C0 overlap measurements. Raw data and code required.

### P2 — Oracles (mandatory all models, EP4 + EP8)

Compute and report A coalescing O0/O1/O2, B replication O0/O1/O2, memory and quality feasibility, generic vs loop-specific ablations. Explain every untested condition. Only **O1/O2** with measured microbenchmark calibration can support a GO recommendation.

### P3 — Minimal implementation (conditional; do not force weak direction)

Implement A or B only in combinations passing GO headroom. If one direction passes in only one model, implement it there and run correctness-preserving negative/control experiments for other models where feasible; do not invent full implementation on models failing headroom. The discovery and oracle coverage must still include all three models.

### Research GO criteria

To call **`GO_A`** or **`GO_B`**, require all:

1. TRUE EP4 or EP8 fidelity-verified baseline exists; no architecture/quality compromise.
2. **Readiness- and overhead-constrained E2E oracle predicts ≥1.15×** improvement against a strong, *memory-/arrival-matched generic baseline*; target ≥1.20× for implementation enthusiasm.
3. The **incremental benefit from recurrent pass / tied-weight information** is material (e.g. ≥5% E2E beyond generic same-pass coalescing or generic replication) and evidenced by ablation; this is a heuristic research threshold, not a measured fact.
4. Novelty audit finds a contribution distinct from generic expert scheduling/replication; implementation feasible within available HBM and EP collective constraints.

`CONDITIONAL` if signal exists but below gates or O2 poorly calibrated. `NO_GO_A` / `NO_GO_B` if robustly insufficient headroom; `BLOCKED_TRUE_EP` / `BLOCKED_RUNTIME` if measurement unavailable (not a negative scientific claim); `NO_GO_EP_DIRECTION` if both genuinely fail after adequate measurement. Always state whether B1, B2 and B4 conclusions differ.

### Prior-work novelty controls

- Cross-layer EP scheduling/load balance: EasyBalance (ICML 2026); separate generic cross-layer dispatch gains from true *cross-pass same-weight GEMM* savings.
- Shortcut-architecture communication overlap: ScMoE (ICML 2025) and FarSkip-Collective (MLSys 2026); do not claim a fundamentally new generic compute/communication overlap primitive.
- Pre-gated MoE (ISCA 2024): learned early selection changes/extends the routing mechanism, not equivalent to exact next-loop predispatch.
- Generic expert replication, load balancing and grouped GEMM: use as **strong baselines**, not as novelty.

## 12. Required artifacts and naming

All under `poc_flashvep/reports/looped_moe_ep_coalescing_replication/`:

```
README.md
report.md                        # full methods, quantitative tables, evidence, caveats
go_no_go.md                      # per-track × per-model × EP4/EP8 explicit verdict
manifest.json                   # model revisions, GPU topology, backend, commits
configs/                        # immutable run configs
logs/                           # environment/runtime/launcher/NCCL diagnostics
raw/                            # machine-readable traces, timings, routing counts
summary/                        # aggregate CSVs and confidence intervals
figures/                        # per-pass skew, heat maps, timelines, break-even plots
scripts/                        # reproducible launch, extraction and analysis scripts
quality/                        # matched native / EP / A / B fidelity checks
oracle/                         # O0/O1/O2 parameters, calibration and predictions
```

**Minimum report tables:**

1. Environment and checkpoint loader/fidelity for 3 models × EP4/EP8.
2. Baseline E2E/TTFT/TPOT/dispatch/compute/combine by model, EP, B1/B2/B4.
3. Per-pass EP rank imbalance and recurring-core versus non-recurring share.
4. A: actual eligible coalescings, wait budgets, same-pass control, O0/O1/O2, if implemented M.
5. B: pass-conditioned vs generic replication, replica memory, load balance, O0/O1/O2, if implemented M.
6. C0: valid ready-work overlap windows and speculative lookahead predictability clearly marked diagnostic.
7. Per-model verdict with sensitivity, CI, failures and untested cells; practical multi-GPU architecture recommendation.

Capture plots and raw data sufficient for independent re-analysis; keep a one-page executive summary first. Do not claim commit/push until commands actually succeed.

## 13. Work discipline and reproducibility

- Inspect the existing repository and dirty/staged files first. Preserve all unrelated working tree changes. No `git reset --hard`, force push, destructive cleanup or process termination on shared GPUs.
- Use a fresh research branch if safe; avoid conflicting modifications and record existing branch/HEAD.
- Use bounded/time-boxed feasibility checks when distributed backend work becomes unexpectedly large. Report honest partial status rather than replacing TRUE EP with simulator results.
- Work in small stages with unit tests for (a) mapping parity, (b) top-2 scatter/add, (c) pass/physical-layer identity, (d) collective ordering, (e) native-vs-EP logits/greedy fidelity, (f) duplicate-expert token handling and replica identity.
- Make scripts idempotent; save failure logs and use unique run IDs so parallel EP4 groups cannot clobber each other.
- Summarize code changed, exact commands, measured vs estimated results and GO decision. If permitted by repository policy and credentials, commit and push only after validation; verify remote SHA. Don't assert success without confirming it.

## 14. Source URLs (current at spec creation)

- IFM Ouro model card: https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-500b
- IFM Huginn model card: https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b
- IFM official runtime: https://github.com/ifm-ai/xllm-loop
- IFM architecture notes: https://huskydoge.github.io/husky-blog/posts/recursive_models/towards-looped-models-done-right/
- Foil official checkpoint: https://huggingface.co/ShourenWSR/how-to-loop-moe
- Foil official implementation: https://github.com/SR-A-W/how-to-loop-moe
- EasyBalance (cross-layer workload co-scheduling): https://proceedings.mlr.press/v306/wu26o.html
- ScMoE (shortcut architecture to overlap communication): https://proceedings.mlr.press/v267/cai25c.html
- FarSkip-Collective (overlap with modified model): https://proceedings.mlsys.org/paper_files/paper/2026/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html
- Pre-gated MoE (ISCA 2024): https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf

---

**Decision rule:** Treat recurrent weight sharing as a **hypothesis about scheduling freedom**, not proof of speedup. Track A and Track B are the only main optimization tracks; C0 is a diagnostic. An architecture is worth building only if latency-critical, budget-correct, quality-preserving headroom survives strong generic EP baselines in TRUE EP4/EP8.
