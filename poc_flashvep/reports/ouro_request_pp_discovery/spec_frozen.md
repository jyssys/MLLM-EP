# Ouro Request-Level Pipeline Parallelism: Layer-PP vs Loop-PP Discovery PoC

**Status:** Implementation specification (not experimental results)  
**Version:** 1.0 — 2026-10-20  
**Model:** `ByteDance/Ouro-1.4B`  
**Hardware:** physical NVIDIA GPUs **4, 5, 6, 7** (four GPUs total)  
**Required remote specification path:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_request_pipeline_pp_discovery_spec.md`  
**Primary objective:** Implement and measure two **request-level**, **no-prompt-chunking** forms of pipeline parallelism with identical Ouro model semantics and a comparable runtime, then identify the causal source of differences in throughput, latency, communication, memory, and pipeline utilization.

> **Research stance:** This is a discovery experiment, not a claim that Loop-PP outperforms Layer-PP. Record negative outcomes and avoid designing a new scheduling policy before a measured bottleneck justifies one.

---

## 1. Research questions and hypotheses

### Q1. Fixed-depth execution

When `R ∈ {2, 3, 4}` is fixed, how do conventional **physical-layer partitioning** and **recurrent-loop partitioning** compare under the same request-level pipeline workload?

- **H1 — communication:** Loop-PP reduces inter-rank activation handoffs for a full recurrent forward pass.
- **H2 — utilization:** Fewer, heavier Loop-PP stages may increase stage bubbles and lower attainable overlap at low concurrency; the net result is **unknown**.
- **H3 — memory:** Both placements can keep their own KV slots local to the executing rank, but Loop-PP replicates the physical decoder weights on every rank. Neither placement intrinsically compresses the total logical KV cache.
- **H4 — serving crossover:** Pipeline throughput/latency rankings may vary with in-flight request count, prompt length, decode length, and interconnect cost. Do not assume a crossover exists.

### Q2. Variable recurrent depth

What changes when requests/tokens exit after different numbers of loops?

- Both schemes can skip **entire downstream recurrent loops** after a true exit decision. Layer-PP does **not** have to execute skipped loops.
- Loop-PP shortens the route taken by an early-exiting request/token, but assigns the mandatory early loops to particular ranks, potentially creating a permanent R1/R2 hotspot and underusing late-loop ranks.
- Layer-PP revisits all physical-layer partitions for every executed loop; cumulative compute can be more evenly distributed across ranks, yet variable-sized active sets, re-entrant traffic, queue contention, and bubbles still create real utilization loss.
- A fixed-depth sweep is **not** equivalent to true token-level dynamic exit. Measure and label these separately.

### Q3. Future research opportunity (not Phase-1 implementation)

Does the measured trade-off between **recurrent communication**, **per-rank active-loop load**, and **KV affinity** suggest an architecture beyond static Loop-PP or Layer-PP? Candidate future work might include flexible rank roles or exit-aware stage placement, but only if profiling demonstrates sufficient headroom.

---

## 2. Scope and exclusions

### Mandatory

1. Inspect the existing project, model code, checkpoints, model-license requirements, server environment, and past relevant PoCs. Reuse tested kernels, attention implementations, distributed helpers, and workload tooling where appropriate.
2. Provide a correct, single-GPU Ouro reference for fixed depths 2/3/4 (the primary default reference is R4).
3. Implement **Layer-PP4** and **Loop-PP4** on the same physical GPUs 4/5/6/7.
4. Execute the **whole prompt of each request as one prefill unit** at a pipeline stage. Do **not** split a single prompt into token chunks, sequence partitions, or pipeline microchunks.
5. Allow **different requests** (or separate eligible decode steps from different requests) to occupy different ranks at once: e.g., request A in R2 on GPU5 while request B runs R1 on GPU4.
6. Keep exact recurrent-iteration / physical-layer KV slots, correct RoPE positions, per-request state, and input/output semantics. Use the same KV policy in all compared arms.
7. Record correctness, runtime timeline, communication, memory footprint, per-rank activity, and serving metrics using reproducible scripts.

### Excluded from the first implementation

- Prompt chunking, token-level intra-request sequence pipeline, context/sequence parallelism, expert parallelism, KV compression/quantization/offload, speculative decoding, training/fine-tuning, custom CUDA kernel development, RL scheduling, speculative gate-based skipping, and an adaptive partition-selection algorithm.
- Do **not** treat post-hoc exit selection as true early-exit compute reduction.
- Do **not** change attention correctness merely to make distributed execution faster.
- Do **not** claim a complete dynamic-depth result if ragged token KV semantics have not been validated.

### Ordered phases

- **Phase 0 (environment and architecture audit):** verify source/weights/GPU topology and reference semantics.
- **Phase 1 (required):** fixed-R4 correctness and two genuine asynchronous request-level pipelines.
- **Phase 2 (required if Phase 1 succeeds):** fixed R2/R3/R4 performance and concurrency/length sweeps.
- **Phase 3 (conditional):** true dynamic exit and/or fixed exit-trace replay; explicitly distinguish actual model execution from synthetic scheduling replay.

Finish Phase 1 with a report even if Phase 2/3 cannot be completed. Do not let Phase 3 delay the core deliverables.

---

## 3. Hardware and environment contract

**Allowed physical GPUs:** `4,5,6,7` only. No other GPU may be allocated or used.

Recommended environment launch (adapt to the actual repository and container runtime):

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
# Within the process group: logical cuda:0/1/2/3 map to physical 4/5/6/7.
nvidia-smi -i 4,5,6,7
nvidia-smi topo -m
```

- Confirm model support, current distributed stack, GPU types/capacities and topology; capture `torch`, `transformers`, CUDA, driver, NCCL versions, model revision, git SHA, and flags.
- Check that the four GPUs are available before starting. Do not kill others' processes, alter global installations, or modify unrelated artifacts.
- Prefer a 4-rank single-node `torchrun` launch. Use NCCL P2P or equivalent device-to-device send/receive with traceable message lifetimes; **do not silently stage all activations through host RAM**.
- Use BF16 inference and the same attention backend on both implementations where possible. If kernels/backends differ due to implementation constraints, disclose and isolate the effect.
- Warm up model and communication paths. Synchronize the whole distributed run at workload boundaries, but **do not insert global barriers inside every stage/handoff** (this would destroy pipeline overlap).
- Save all detailed outputs under a dedicated new directory in the repository, e.g. `poc_flashvep/reports/ouro_request_pp_discovery/`; do not overwrite unrelated prior reports.

---

## 4. Model assumptions; verify against checked-out revision

Official model reference: <https://huggingface.co/ByteDance/Ouro-1.4B>.

The publicly available config has **24 physical decoder layers**, **4 recurrent iterations by default** (`total_ut_steps=4`), hidden size 2048, 16 attention heads, 16 KV heads, head dimension 128, and BF16 configuration. Pin the actual model/checkpoint revision in the report. Treat the downloaded model code as authoritative if it differs from this spec.

Core semantics: for each executed recurrent iteration `r`, run physical decoder blocks `L1,...,L24`, sharing the corresponding decoder weights across iterations. Cache slots must preserve the logical identifier `(request_id, recurrent_iteration, physical_layer, token_position)`.

**IMPORTANT: validate the placement and usage of embeddings, RMSNorm/final norm, exit gate, and LM head.** They are not uniformly part of every partition's decoder-block subset. Preserve the exact placement/timing of norm, exit decisions, and logits. If a module is replicated to enable a terminal stage to produce logits, record the duplication, compute, and memory cost on **both** arms. Avoid making only one arm pay a head-transfer penalty.

### Fixed-depth correctness rule

- R4: four actual recurrent iterations, no early-exit skipping.
- R2/R3: execute exactly two/three recurrent iterations and project the correctly normalized hidden state to logits (if supported by the reference). Merely picking a post-hoc output from a full R4 pass **does not** validate saved compute or fixed-R2/R3 timing.
- Pre-existing `early_exit_threshold` logic may compute all recurrent iterations before selecting a hidden state. Explicitly inspect code before treating it as a runtime stopping mechanism.
- For correctness, use the same checkpoint, tokenizer, attention mask, position IDs, generation policy, requested sequence lengths, and per-loop KV-cache semantics.

---

## 5. Exact pipeline placements

### Arm A — Naive / physical-layer PP4 (`layer_pp4`)

Use four ranks. Divide the **24 physical layers**, not the 96 unrolled invocations:

| Logical rank | Physical GPU | Owned decoder weights | Every recurrent iteration |
|---|---|---|---|
| rank 0 | GPU 4 | L1–L6 | execute its L1–L6 of R1,R2,R3,R4 |
| rank 1 | GPU 5 | L7–L12 | execute its L7–L12 of R1,R2,R3,R4 |
| rank 2 | GPU 6 | L13–L18 | execute its L13–L18 of R1,R2,R3,R4 |
| rank 3 | GPU 7 | L19–L24 | execute its L19–L24 of R1,R2,R3,R4 |

For an R4 request:

```text
R1: GPU4 -> GPU5 -> GPU6 -> GPU7
                                   |
R2: GPU4 -> GPU5 -> GPU6 -> GPU7
                                   |
R3: GPU4 -> GPU5 -> GPU6 -> GPU7
                                   |
R4: GPU4 -> GPU5 -> GPU6 -> GPU7
```

The return from rank 3 to rank 0 between loops is an actual P2P activation handoff. The logical route contains **16 decoder partition visits** and **15 inter-rank hidden-state transfers** for one recurrent model forward at R4, excluding transfer of final token IDs/control and explicit logits/head placement. At depth `R`, the corresponding counts are `4R` visits and `4R-1` transfers.

**KV ownership:** each rank keeps local KV for its six physical layers **for each executed recurrent iteration**. For a full R4 history, each rank stores 6×4 = 24 logical layers' KV. KV data should not be repeatedly copied across stage boundaries; only the necessary activation/control messages cross ranks.

**Re-entrant scheduling:** rank 0 receives both new R1 requests and already-progressed R2/R3/R4 requests from rank 3. Implement explicit request/loop/layer-range identifiers; no hidden global model-loop state or request mix-ups. The four ranks must remain able to work on different requests simultaneously.

### Arm B — Loop-wise PP4 (`loop_pp4`)

Each rank holds the **full 24-layer recurrent-core weights** and owns one recurrent iteration:

| Logical rank | Physical GPU | Owned decoder weights | Assigned iteration |
|---|---|---|---|
| rank 0 | GPU 4 | L1–L24 | R1 |
| rank 1 | GPU 5 | L1–L24 | R2 |
| rank 2 | GPU 6 | L1–L24 | R3 |
| rank 3 | GPU 7 | L1–L24 | R4 |

For an R4 request:

```text
GPU4: R1 (L1--L24)
         |
GPU5: R2 (L1--L24)
         |
GPU6: R3 (L1--L24)
         |
GPU7: R4 (L1--L24)
```

The R4 request makes **4 stage visits** and **3 inter-rank hidden-state transfers** for one recurrent model forward. At depth `R`, it makes `R` visits and `R-1` transfers, excluding token/control/head transfers. Do **not** count a missing R3/R4 visit as a benefit unique to Loop-PP: Layer-PP also skips iterations after an actual exit.

**KV ownership:** GPU4 stores all R1 KV (24 layers) for its assigned requests; GPU5 stores all R2 KV; and so forth. Past R2 KV remains local to GPU5 for subsequent decode steps; executing R2 needs prior **R2/L** KV for each layer, **not** prior R1 KV from GPU4. GPU4 passes the current request's **hidden state**, not its cache. At full R4 history, Loop-PP also stores 24 logical layers' KV per rank.

**Dynamic-depth structural implication:** early iteration ranks process all active tokens; late ranks see only those that continue. This can impose a stage-capacity bottleneck in steady state even if the total amount of work drops.

### Common stage semantics

- Define a *stage work item* as `(request_id, decode_step/prefill, iteration_id, layer_start, layer_end, sequence_metadata, KV_handle, ready_event)`.
- A request's **prefill** is its complete prompt, not a chunk. A request's **decode** is sequential in generated-token order; different requests may advance concurrently when their dependencies are satisfied.
- The next token cannot be decoded for a request until the previous token has produced the required logits / token ID and routing to the model entry has completed.
- Each route is a correctness dependency, not a compulsory synchronization of all four GPUs.
- State stays separate across requests and decoding iterations: KV slots, RoPE offsets, masks, RNG, and request completion status.
- Stage queues are per-rank and event-driven. Avoid serial host loops that dispatch request A through all ranks before admitting request B.

---

## 6. Request-level overlap: mandatory functional test

**No prompt-token chunking is allowed.** Each request's prompt is handled as one unit at a stage.

With four independent requests A/B/C/D and Loop-PP4, the following idealized schedule must be *possible* when dependencies permit:

| Timeslot | GPU4 (R1) | GPU5 (R2) | GPU6 (R3) | GPU7 (R4) |
|---|---|---|---|---|
| 1 | A | idle | idle | idle |
| 2 | B | A | idle | idle |
| 3 | C | B | A | idle |
| 4 | D | C | B | A |

The same principle must hold for Layer-PP4, albeit with a cyclic, re-entrant route. This table is only an ideal schedule—not measured timing, and not an assertion that the latency of stage work is equal.

**Required proof of genuine pipelining:** produce a GPU activity / NVTX / CUDA event timeline showing overlapping stage execution on separate GPUs for **different request IDs**, plus numerical busy/idle timestamps. Running all ranks but processing one request to completion before the next is **not** a successful pipeline implementation.

### Scheduler fairness and comparability

- Use a common ready-queue discipline (start with FIFO; identical admission ordering and max in-flight count); both arms must obey the same per-request causal dependencies.
- Because stage granularity differs (6 vs 24 layers), same FIFO policy does not guarantee equal batching efficiency. Keep per-stage batching disabled initially (one request per stage invocation) and **report** its limitations rather than hiding them.
- Avoid artificially forcing all requests to synchronize at loop boundaries. Per-request readiness should determine progress.
- Explicitly count queue waiting time versus compute and P2P time. If request-level one-at-a-time kernels severely underutilize GPUs, that is an observation, but identify potential confounding from batching before making a general architecture claim.

---

## 7. Attention, KV, prefill, and decoding correctness

### KV addressing

For each request and executed recurrent iteration, each physical layer has distinct K and V indexed by past token position:

```text
KV[request_id][iteration_id][layer_id][position]
```

The upstream stage's cached KV is **not** an input to the downstream stage. The downstream stage uses its own cache plus upstream current hidden states.

### Prefill

- Each request's entire prompt is forwarded through all stages for its target number of recurrent iterations.
- Preserve causal masking and complete prompt positions without prompt-token chunking.
- For multi-request execution, ensure no cross-request cache aliasing or accidental sharing.
- Report prefill wall time and TTFT separately from decode.

### Decode

- Maintain persistent per-request KV on its owner rank across successive generated tokens.
- After the terminal iteration, produce logits / deterministic next-token ID, then make the ID available to the entry side for the request's next forward pass.
- Include **return/control communication** and projection/head work in actual latency/bytes totals, even though the analytic stage-handoff counts above exclude these items.
- Compare exact greedy token IDs first. Use a fixed output-token count for throughput experiments (disable EOS stop or otherwise control stopping consistently); separately test normal EOS termination. Record actual generated token counts in either case.
- Two requests on different stages must be able to make independent decode progress without a global token-step barrier.

### Handling normalization and heads

Implement exactly one agreed terminal-head plan in both arms. Suggested simple implementation: keep identical embedding and LM-head/norm replicas where necessary for dynamic terminal-rank support; record the replicated weights and associated runtime compute separately. If using a distinct head owner instead, include its activation sends and scheduling in **both** arms. Make the chosen plan and fairness trade-off explicit in results.

### Correctness thresholds

- Fixed R4: compare a representative set of exact generated token IDs to the trusted single-GPU reference, and compare terminal logits/hidden states with defined BF16 tolerance.
- Fixed R2/R3: compare to an *actually executed* R2/R3 single-GPU reference, not a post-hoc R4 selection masquerading as saved compute.
- Multiple concurrent requests: compare outputs with sequential per-request reference on the same prompts, seed, and generation configuration. Repeat under interleaved scheduling to detect KV aliasing.
- Record max/mean absolute and relative logit differences; decide numeric tolerance based on observed BF16/kernel differences. Flag any greedy-token mismatch and root-cause it; never silently ignore it.
- Verify no device-to-host staging of KV and no unexpected rank-wide barriers.

---

## 8. Dynamic depth: separate research phase

**Do not ship an apparent Dynamic-Exit benchmark without actual execution skipping.** The Hugging Face Ouro path may collect states/gates for multiple loops and choose a result post-hoc. Inspect the checked-out implementation and prove skipped kernels are absent from traces.

### Recommended progression

1. **Fixed R2/R3/R4:** deterministic, real truncated executions. This isolates placement effects and gives an early indication of late-loop rank idle.
2. **Optional trace-driven scheduling-only replay:** prepare a controlled distribution of exit depths; mark it **replay/synthetic** if no model kernels actually skip. Never report it as end-to-end dynamic-depth inference.
3. **True request-level exit (if compatible with model semantics):** a whole request takes its actual chosen depth when possible; preserve the correct KV semantics. If a request's generated tokens have different depth requirements, request-wide early exit alone is insufficient and must be explicitly labeled a simplification.
4. **True token-level exit (advanced, conditional):** support token-dependent iteration counts, active-token packing/masks, per-loop/layer ragged KV positions, cache updates for previous shallow-exit tokens, and faithfully implemented fallback semantics. Use the same exit-policy and model outputs in both arms. Refer to the original model and CDB-type implementations as necessary; do not substitute previous-loop KV without validation.

### Dynamic measurements

- Exit depth histogram per decoded token and, if applicable, per prompt token.
- Active token counts per iteration; per-rank scheduled work / active GPU time / idle GPU time.
- Host/rank queue occupancy, completion rate, early-exit control messages, and skipped stage count.
- Quality / output agreement under equal exit semantics.
- Distinguish **compute savings** from **communication savings** and **unused GPU capacity**.

**Important:** Even when late iterations have fewer active tokens, Layer-PP4 is not immune to bubbles or loss of kernel efficiency. The hypothesis is only that its *nominal compute distribution* is more balanced across ranks under comparable physical-layer costs.

---

## 9. Workload matrix and benchmarking protocol

Start small; avoid running a huge Cartesian product before the pipeline is correct.

### Tier 0: correctness/smoke

- Fixed depth: R4 first; then R2 and R3.
- In-flight requests: 1 and 4 (and 2 to test partial fill).
- Prompt lengths: 16/128 tokens.
- Decode lengths: 1/8 tokens initially.
- Output: exact IDs, cross-rank stage traces, cache lengths, no deadlocks, and resource sanity.

### Tier 1: primary fixed-depth performance

| Variable | Values |
|---|---|
| GPU set | physical 4,5,6,7 |
| Placement | `layer_pp4`, `loop_pp4` |
| Depth | R4, then R2/R3 |
| Concurrent requests / in-flight limit | 1, 2, 4, 8, 16, 32; 64 only if memory/time permit |
| Input length | 128, 512, 2048 tokens |
| Output length | 32, 128 tokens |
| Generation | deterministic greedy, identical tokens/prompt and stopping policy |
| KV semantics | complete loop×layer exact cache (unless a separately labeled experiment changes it) |
| Precision | BF16 |

**Priority runs:** R4 × concurrency {1,4,8,16} × input {128,512} × output {32,128}; then depth {2,3,4} × concurrency {2,8,16} at a representative length. Expand according to observed bottlenecks.

Include at least two workload types if practical:

- Uniform request lengths (controlled mechanism analysis).
- Mixed request lengths / realistic prompts (robustness test) with pinned tokenized inputs.

### Procedure

1. Same warm-up policy and number of repetitions across arms; begin with at least 3 measured repetitions after warm-up (increase to 5 when feasible).
2. Alternate arm execution order to mitigate temperature/frequency bias.
3. Record batch admission time, terminal times, iteration/transfer timestamps, and generation token accounting.
4. Distinguish **latency under load** (request TTFT/TPOT/E2E) from **capacity** (steady-state tokens/s and requests/s). Track both throughput and tail latency.
5. Report mean/median/P90 and variability; use uncertainty intervals where justified, not fabricated significance claims.
6. On OOM, record actual peak allocation and failure case; do not silently lower batch size in only one arm.
7. For comparison at the same number of GPUs, always state the memory capacity used, whether an arm fails, and whether head replication changes per-rank memory.

### Metrics

**Performance:** request/s, committed output tokens/s, TTFT P50/P90, decode TPOT P50/P90, E2E latency P50/P90, end-to-end workload makespan, GPU utilization and per-rank throughput.

**Compute and pipeline:** CUDA-event stage compute durations, per-rank busy/idle (with precise definition), queue waiting, stage arrival/departure times, bubble fraction, active token counts, pipeline overlap interval, norm/head time, prefill vs decode components.

**Communication:** count and payload size of P2P transfers, sender/receiver rank pair, time in enqueue/wait/synchronization, effective bandwidth, turnaround rank3→rank0 events, logits/token-ID/control transfers, NVLink/PCIe topology evidence. Do not sum overlapping kernel/event times and call the result wall time.

**Memory:** per-rank model-weights estimate, actual allocated/reserved and peak HBM, per-request per-loop/layer KV allocation, fragmentation metadata where possible, KV lifetime and cache cleanup, OOM limits.

**Correctness:** per-request greedy token agreement, logits errors, cache slot/length checks, fixed-depth loop count, exit trace equality in true dynamic mode.

### Analytical controls

For a single model-forward at fixed `R`:

```text
Layer-PP4: physical partition visits = 4R, hidden-state handoffs = 4R - 1
Loop-PP4:  loop-stage visits         = R,  hidden-state handoffs = R - 1
```

These exclude the decode return path, embedding/head movement, and control messages. Derive observed P2P counts from tracing rather than hard-coding these as measured values.

In a simplified compute-only steady-state model with identical physical-layer cost `c` and token iteration-depth random variable `D`:

```text
Layer-PP4 work / rank ≈ 6 * c * sum_r P(D >= r)
Loop-PP4  work / rank r ≈ 24 * c * P(D >= r)
```

This expresses only a possible load imbalance, **not actual throughput**. GPU kernel efficiency, batch formation, communication, queue order, and varying prompt/decode cost matter.

---

## 10. Implementation architecture and module boundaries

Adapt filenames to the actual existing repository. Prefer a small, inspectable PoC rather than a full serving framework.

Suggested structure:

```text
poc_flashvep/
  ouro_request_pp/
    README.md
    model_adapter.py          # checked, version-pinned Ouro decoder/module helpers
    partition.py              # layer_pp4 / loop_pp4 ownership mapping
    kv_manager.py             # per-request, per-(loop,layer) cache handles
    transport.py              # traced asynchronous rank-to-rank messages
    scheduler.py              # event-driven request-level stage queues
    runtime.py                # prefill/decode state transitions
    reference.py              # single-GPU reference and correctness comparisons
    workloads.py              # pinned tokenized request sets
    benchmark.py              # CLI flags and measurement runner
    trace.py                  # GPU/NVTX/P2P trace collection
    tests/
  reports/
    ouro_request_pipeline_pp_discovery_spec.md
    ouro_request_pp_discovery/
      environment.md
      correctness.md
      benchmark_summary.md
      analysis.md
      results.csv
      traces/
      commands.sh
```

Suggested CLI (examples; actual implementation may use other flags):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  poc_flashvep/ouro_request_pp/benchmark.py \
  --model ByteDance/Ouro-1.4B --placement layer_pp4 \
  --depth 4 --concurrency 8 --prompt-len 512 --output-len 32 \
  --precision bf16 --seed 42 --no-prompt-chunking

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  poc_flashvep/ouro_request_pp/benchmark.py \
  --model ByteDance/Ouro-1.4B --placement loop_pp4 \
  --depth 4 --concurrency 8 --prompt-len 512 --output-len 32 \
  --precision bf16 --seed 42 --no-prompt-chunking
```

**Process/rank caution:** physical GPUs 4/5/6/7 appear inside the launcher as logical CUDA devices 0/1/2/3. Never use `cuda:4` inside a four-device `CUDA_VISIBLE_DEVICES` namespace.

### Code safety rules

- Inspect existing code and repository state first; do not clobber user's in-progress files.
- Do not run destructive cleanup, kill unrelated processes, or require `sudo`.
- Keep modifications modular and reversible; preserve the default execution path.
- On failure, save logs and explain the blocker with reproducible commands rather than reporting guessed measurements.

---

## 11. Acceptance criteria / stage gates

### Gate A — executable and faithful (must pass)

- Model is loaded on physical GPUs 4/5/6/7 with the correct placement and BF16.
- `layer_pp4` indeed partitions **24 physical weights 6 per rank** and revisits ranks for each loop.
- `loop_pp4` indeed replicates **all 24 physical decoder weights** and executes a distinct full loop per rank.
- No prompt token chunking; all stage work is request-level.
- Fixed-R4 distributed outputs agree with a single-GPU reference within justified numeric tolerance; greedy IDs match on the tested requests.
- No request KV corruption with interleaved requests and decode steps.

### Gate B — genuine pipelining (must pass)

- With at least four in-flight requests, independent requests execute concurrently on different GPUs, demonstrated by rank timelines and event timestamps.
- Asynchronous communication and scheduling are real; no per-request global barrier that serializes the four-rank system.
- Full forward + token-return/control path is accounted for.

### Gate C — comparable discovery measurements (must pass)

- Fixed-R4 performance runs yield paired metrics on both arms for at least concurrency 1/4/8/16, plus a subset of R2/R3.
- Stage busy/idle, P2P bytes/count/time, per-rank KV/weights, and TTFT/TPOT/E2E/throughput are reported.
- Conclusions distinguish placement benefit from different weight replication, stage granularity, and kernel inefficiencies.
- Report the possibility of **no performance gain**. Do not overclaim novelty or absolute speedups.

### Gate D — dynamic depth (conditional)

- Exit depth is selected and work is **actually skipped** by the distributed runtime; active-token traces and omitted kernel invocations demonstrate this.
- Both placement arms use the same validated KV/exit policy and produce matching results.
- If token-level dynamic execution is not implemented correctly, explicitly mark it as future work; fixed R2/R3 are still useful outputs.

### Suggested empirical decision (not prespecified scientific truth)

Continue beyond discovery only if a repeatable, practically meaningful advantage or a robust and explainable regime-specific crossover is observed **and** the effect persists with reasonable scheduler/backend controls. If performance gaps are dominated by accidental Python serialization or unequal implementations, fix the confounder before drawing conclusions.

---

## 12. Required deliverables and reporting format

1. **Implementation:** reproducible runnable code for both four-rank placements, single-GPU reference, and at least one test verifying request-level overlap.
2. **Environment and reproducibility:** exact GPUs, topology, versions, git/model revisions, command lines, seeds, trace instructions, and checked-out model assumptions.
3. **Correctness report:** fixed R4 (and if possible R2/R3), examples of golden outputs, comparison metrics, KV indexing/length assertions.
4. **Performance tables and plots:** concurrency vs tokens/s, TTFT/TPOT/E2E; GPU rank busy/idle and queue time; P2P count/bytes/time; peak HBM and KV occupancy. Include raw CSV/JSON.
5. **Timeline evidence:** representative 4-in-flight-request trace showing overlap for both arms and explicit recurrent GPU revisits in Layer-PP4.
6. **Final analysis:** state exact cases where Layer-PP wins, Loop-PP wins, or neither wins; explain the measured contribution of communication, bubbles, and loop-depth imbalance. Clearly identify whether results are live GPU measurements, analytically estimated values, or scheduling-only replay.
7. **Repository state:** provide changed-file list and reproducible commands; commit changes if the working tree and local workflow permit, but do not push or alter external branches without instruction.

**Stop and report early if:** GPUs 4/5/6/7 are unavailable; the model/reference cannot be faithfully loaded; a necessary unsupported KV feature prevents correctness; or the purported pipeline does not overlap requests. Preserve code, traces, and blockers so work can resume.

---

## 13. External references (for implementation context only)

- Ouro-1.4B model/config/source: <https://huggingface.co/ByteDance/Ouro-1.4B>
- Ouro model implementation (verify current repository revision before using): <https://huggingface.co/ByteDance/Ouro-1.4B/blob/main/modeling_ouro.py>
- CDB-style true early-exit / ragged-KV implementation reference, for **conditional** Phase 3 only: <https://github.com/kschwethelm/continuous-depth-batching>

These are pointers, not evidence that the target server already contains these versions. Verify license, runtime compatibility, and actual model code locally.
