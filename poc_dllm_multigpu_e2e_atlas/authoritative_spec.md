# dLLM Multi-GPU End-to-End Critical-Path Atlas and Architecture Discovery PoC

**Status:** Experiment specification — characterization before optimization.  
**Server destination:** `/home/esjung/MLLM-EP-github/poc_flashvep/reports/dllm_multigpu_e2e_critical_path_spec.md`  
**Permitted physical GPUs:** **4, 5, 6, 7 only**; use `CUDA_VISIBLE_DEVICES=4,5,6,7`.  
**Preferred repository output:** `poc_dllm_multigpu_e2e_atlas/`  
**Suggested branch:** `flashvep/dllm-multigpu-e2e-atlas`  
**Date of design:** 2026-10-14.  
**Primary objective:** Produce a trustworthy, *end-to-end and per-refinement-iteration* latency atlas for a real dLLM running across four GPUs, identify quantitatively meaningful synchronization, memory-movement and pipeline/host gaps, and rank **non-TP** architectural alternatives using physical evidence. Do not assume MoE, EP, CP, PP, or a particular method is the answer.

## 0. Executive research question

> At each step of a block-diffusion language model's generation, **where does real request latency go**, and which recurring multi-GPU dependency, collective, memory movement, or launch/scheduling bottleneck can be reduced while preserving inference semantics?

Two independent deliverables are required:
1. A **measured, validity-tagged E2E timeline / critical-path decomposition**, broken down by request phase, block, refinement iteration, layer family, GPU rank and communication/memory categories.
2. An **architecture decision matrix** that evaluates tensor parallelism (TP), pipeline parallelism (PP), data/replica parallelism (DP), and feasible hybrids under **the same four physical GPUs**, distinguishing latency for one request from throughput for concurrent requests.

This is a discovery PoC. A novel multi-GPU architecture is optional and follows only if a compelling physical bottleneck and realistic end-to-end gain are established.

## 1. Project-specific empirical starting point

Do not repeat a closed hypothesis without new evidence. Prior project PoCs found:

- **MultiBD × EP4:** logical dispatch/combine bytes per committed token rose substantially, but physical EP milliseconds per committed token did *not* degrade; PCE remained ~0.93–1.01. Structural bytes were not a latency target.
- **Looped MoE × EP4:** static load balance improved in a ~0.55B model, but clean wall latency was statistically unchanged. A load oracle was not a latency oracle.
- **dLLM prefix–canvas CP:** same four H100 GPUs, non-preview `LLaDA2.0-mini` native **TP4/EP1/DP1/PP1/CP1**, four KV heads/16 query heads, 20 attention layers, 19 sparse layers; prefix KV already cached. At P16,384, component times per 20 layers × 3 refinement passes (60 calls): optimized native attention **4.094 ms**, exact local canvas-update control **2.880 ms**, candidate exact CP **7.364 ms** excluding prefix preparation. The CP candidate was ~1.80× slower than native. It was NOT integrated into E2E.
- The same earlier *single heavy profile* classified about **123.25 ms of `memcpy128` GPU kernel service time over 560 events**, **22.10 ms of NCCL kernels**, **10.10 ms of split-KV FlashAttention kernels**; these are **not mutually exclusive portions of request wall time**, and `memcpy128` is NOT proof of a hardware D2D/H2D transfer or an actual KV HBM read. The graph/trace-based attention fraction was only a **superset/stationarity estimate**, not a measured exact critical path.
- Previous workload had P16,384 cold E2E ~868 ms and warm decode span ~672 ms across a small calibration set. Do **not** treat absolute historical timings as the new baseline, pool them with new data, or infer a precise 70% attention bottleneck from those estimates.

Relevant prior internal report: `poc_dllm_prefix_canvas_cp/report.md` at commit `5f39e26798d7b600aa0853a72271c845417dae11`. Read its measurement limitations and prior-overlap analysis before instrumenting.

### Mandatory correction: do not invent All-to-All

Under the **current TP4/EP1 native baseline, no EP Dispatch/Combine All-to-All exists**. The native model has TP reductions and may have other collectives; instrument and identify the **actual** primitives. Mark:

- `TP_all_reduce`, `TP_all_gather`, `reduce_scatter`, `broadcast`, `barrier` only when observed/verified;
- `EP_dispatch_all_to_all = NOT_PRESENT` and `EP_combine_all_to_all = NOT_PRESENT` in baseline if indeed absent;
- `OTHER_NCCL` when unclassified, not fictional `A2A`;
- `D2D`, `H2D`, `D2H`, `in-device copy kernel`, `KV logical update`, `cache staging`, and `HBM-read proxy` as **different labels**.

The goal is truthful topology and timeline accounting.

## 2. Research hypotheses (to falsify)

**H1 — Repeated synchronization cost.** One or more barriers/TP collectives or CPU–GPU synchronization boundaries recur at every layer/iteration/block and cause **actual idle time on the critical path** rather than merely high summed NCCL service time.

**H2 — Repeated memory movement cost.** CUDA Graph static-buffer copies, out-of-place current-canvas KV updates, layout conversions, or hidden-state reformatting consume a sizable exclusive fraction of the critical path. In particular, check native `slice_scatter` versus exact safe in-place updates **within the full model**, not only in an isolated component.

**H3 — Pipeline/host gaps.** Model-compute intervals are separated by measurable CPU launch, graph-replay, sampling/token-selection, inter-block refresh, GPU idle, or rank-skew gaps. These may motivate a different execution schedule.

**H4 — Parallel architecture mismatch.** TP4's per-layer collective or small-matrix costs may exceed benefits in some regimes, while PP / hybrid TP×PP / multi-request DP might reduce collective frequency or improve throughput. This is **not automatic**: for B1, PP has genuine pipeline bubbles and loop-to-loop dependencies; for B2/B4, additional independent work may fill them.

**H5 — No substantial target.** Native TP4 is already efficient, measured overhead is small, or claimed optimizations are covered by strong prior systems. The correct verdict can be NO-GO.

## 3. Model, checkpoints, hardware and execution contract

### 3.1 Primary model and runtime

- Primary: **`inclusionAI/LLaDA2.0-mini`** (non-preview), preferably checkpoint revision `dad945cac317da394b390f82c7b40691d8a881ed`, if it exactly matches the prior validated environment.
- Reuse existing pinned **dInfer/SGLang** environment without modifying other PoCs. Freeze complete git revision, interpreter, torch/CUDA/NCCL/DeepEP/FlashAttention/transformers versions, launch flags and pre-existing patches.
- Baseline: **physical 4-GPU TP4/EP1/DP1/PP1/CP1**, unless validation proves another topology. Record each rank's actual operations and KV/parameter ownership. The TP4 baseline should not silently switch to the earlier EP4 bridge.
- H100 80GB, physical 4,5,6,7, observed NVLink/NVSwitch connections; recheck topology and free memory in this session. Do not disturb other users or any pre-existing burners beyond user-authorized own processes.
- Native model precision, attention backend, CUDA Graph policy, block size, thresholds, sampling, EOS, cache logic and max context must remain unchanged in the baseline.

### 3.2 Input and configuration matrix

**Primary discovery cohort:** 8 frozen calibration requests: code (3), math (3), long-context/document (2) or preserve the already-frozen manifest if accessible. **Held-out:** 24 frozen, disjoint request IDs; run only once the measurement contract is validated and a meaningful bottleneck is detected.

Primary regime: **single request B1**, `block_size=32`, default native threshold (~0.9), native sampling setting (temperature 0 if supported), EOS, output budget 64. Freeze random seeds, input text, token IDs, checkpoint and decode parameters before profiling. Report *actual* output lengths and NFE; do not call bounded snippets benchmark accuracy.

Context: P=512, 4,096, 8,192, 16,384 (if supported without unintended cap); optionally P=1,024 / 2,048 for continuity. Skip lengths exceeding the checkpoint's 32,768 logical limit including generated output. Record eager versus graph prefill and physical padded KV capacity. Keep an apples-to-apples graph policy for comparisons where possible; report exceptions explicitly.

Optional second axis: output budgets 32, 64, 128 or block sizes 32/64, **only if safe and native-supported**. Do not turn this into a giant sweep.

**Secondary concurrency sensitivity:** two or four **independent** requests B2/B4 to assess throughput-oriented architectures. Distinguish batch size, concurrent requests, and replica count. Keep B1 as primary latency target.

### 3.3 No misleading cross-regime claims

- Cold E2E (uncached prefix) is different from warm-prefix generation span.
- Prefill-inclusive wall time is different from pure refinement.
- TTFT in a block dLLM may require a definition tied to observable committed token delivery. Record the exact definition.
- TPS must count **final emitted useful tokens**, never fixed physical rows, dummy slots, tentative mask predictions, or NFE.
- B1 TPS/latency cannot be equated with aggregate goodput/QPS under B2/B4.

## 4. Stage 0 — Establish a trustworthy timing contract

**Required before detailed profiling.**

1. Establish clean **CPU monotonic request wall-clock** brackets and per-rank GPU execution boundaries. No per-operation `torch.cuda.synchronize()` in headline execution.
2. Verify physical rank→GPU mapping and process-group topology. In this baseline, list actual TP NCCL operations and **absence** of EP A2A; count calls with evidence.
3. Inspect CUDA Graph capture/replay keys and static input buffer staging. Distinguish capture time, graph preparation, and replay time. The primary clean E2E starts after model loading/capture warm-up unless measuring an explicitly labeled cold-start lane.
4. Audit prefix KV lifetime, actual padded physical key capacity, canvas update mechanism, sampler sync points and cross-block refresh.
5. Check tool availability: `nsys`, Nsight Compute (`ncu`, only if scoped), torch profiler/CUPTI, CUDA Graph node trace support, NVTX. **Do not block the PoC** if high-overhead tooling is unavailable.
6. Validate that final token IDs, acceptance decisions, NFE, EOS and stop reason match between clean and every timing/trace lane **for identical configurations**. Capture at least 8 small smoke requests across representative lengths, all 4 ranks.
7. Evaluate observer perturbation from paired clean ↔ coarse-timed runs. Prefer <=5% mean overhead, reject quantitative fine attribution when >10% or strong parity failure. Full node profiling is for **diagnostic classification**, not headline wall fractions.
8. Establish sufficient overlap semantics: CPU activity, device kernels, and NCCL can run concurrently. Never sum service time of overlapped activities and call it E2E.

**Stage 0 exits:** `RUNTIME_BLOCKED`, `TIMING_INVALID`, or proceed.

## 5. Stage A — Request/block/iteration E2E time atlas

### 5.1 Three complementary measurement lanes

**Lane A — CLEAN:** minimum-perturbation wall, prefill span, block spans, per-iteration wall or replay span, token delivery, output length, NFE, TPS. No expensive layer hooks. Run ≥3 warm-ups plus ≥3 measured runs per request, keeping request as independent statistical unit.

**Lane B — COARSE/EVENT:** timing for actual per-block and per-iteration GPU path; coarse CUDA events / NVTX scopes; per-rank phase intervals. Record *all* actual replay occurrences, not only the **last CUDA Graph replay for each graph key**. Use a preallocated record queue, separate per-replay start/end events or a validated low-overhead host/GPU timestamp correlation method. If the runtime's graph mechanism prevents safe per-replay attribution, label it `UNRESOLVED` rather than multiply the last observation by the call count and claim measured percentages. Perform post-request event resolution.

**Lane C — DIAGNOSTIC PROFILE:** short, bounded Nsight Systems trace with CUDA+NVTX+OSRT+NCCL where supported. Start `--cuda-graph-trace=graph` for low overhead and use **node** tracing only for 1–2 representative short requests if node support and perturbation permit. Correlate graph nodes with application phases by capture metadata/validated NVTX projection. Use `ncu` selectively for suspicious kernels. Do not compute headline E2E from node-profile wall.

### 5.2 Required timeline granularity

At minimum represent each request as:

```text
REQUEST (cold prefill start -> last returned output)
  tokenizer / queue / launch overhead (if in measured boundary; label separately)
  PREFILL (masked or nonmasked; include actual context/graph mode)
  BLOCK 0
    block entry / KV refresh / physical-cache preparation
    REFINEMENT iteration 0
      scheduler + static-graph input staging / H2D/D2D as observed
      LAYER 0..19:
        norm + QKV + RoPE
        attention kernel + attention-specific layout/KV read/canvas update
        attention output projection + TP collective(s)
        norm + router / MoE (or dense MLP), shared expert if applicable
        TP collective(s)
        residual / layer bookkeeping / memory copies
      LM head / logits / softmax-confidence / token-selection / acceptance
      stream/event synchronization / CPU waiting / graph boundaries
    REFINEMENT iteration 1 .. N
    block commit / prefix-cache refresh / next-block preparation
  BLOCK 1 .. B
  request finalization / output transfer
```

This is an **annotation taxonomy**, not an assertion that the real execution serializes exactly in this order. Read source/code and collect physical traces. If a layer has a different order or fused operator, record that faithfully.

### 5.3 Required categories per refinement iteration

Record, where separable:

- `attention_compute_kernel` (FlashAttention/split-KV/CUTLASS etc.)
- `attention_QKV_RoPE_norm_and_proj`
- `routed_expert_or_FFN_compute` (plus shared expert if present)
- `router_and_permutation`
- `TP_all_reduce` and other **observed** collectives (type, call count, bytes proxy, elapsed time)
- `EP_A2A` **NOT_PRESENT** for native TP4/EP1 if verified
- `KV_canvas_update`, `KV_prefix_refresh`, `graph_static_buffer_copy`, `activation_layout_or_reformat`
- `GPU_memcpy_kernel` and physical `cudaMemcpy` H2D/D2H/D2D as different categories
- `sampling_logits_confidence_acceptance` and device↔host scalar sync
- `kernel_launch_graph_replay`, `CPU_wait`, `device_stream_wait`, `rank_wait`, and `GPU_idle_gap`
- `misc_fused_or_unattributed` (never force total percentage to 100 by mislabeling).

Keep per-layer label, per-rank, physical CUDA stream, parent phase, block ID, refinement index, meaningful token count, physical padded key count, and graph replay ID where observable.

### 5.4 Three distinct accounting systems: do not conflate

**(A) Disjoint request wall-time accounting.** Partition CPU-monotonic request wall time into mutually exclusive measurable intervals: prefill, refinement blocks/iterations, block commit/refresh, sampler/host and unaccounted/gaps. Report wall-time percentages whose denominator is **that request's clean or bounded-coarse wall**, and require near-closure (target ±5% with explicit measurement gaps). Distinguish GPU execution that extends past host-return boundaries.

**(B) GPU work/service time.** Sum active kernel durations **per kernel family, per rank/stream**. These are not request-wall percentages: concurrent kernels, concurrent ranks and trace overlap can make total work time > wall. Present absolute service time, counts and utilization, not percentages purporting to partition E2E.

**(C) Critical-path / stall time.** Reconstruct a partial-order graph using stream order, CUDA graph dependencies, GPU event wait, NCCL dependencies and host→device launch/sync. Identify the longest causally necessary path (or validated conservative bounds) per request and per refinement. Classify GPU idle gaps and non-overlapped synchronization separately from NCCL service time. Cross-rank timestamps must be aligned/calibrated or handled through boundaries; **never sum unrelated ranks' maxima**. When incomplete, label attributed fraction `PARTIAL` or `UNRESOLVED`.

**Illustrative requested output (not measured):**

| Phase, per refinement | Critical-path ms | % of refinement | Validity |
|---|---:|---:|---|
| attention + local projections | 2.1 | 18% | measured |
| expert/FFN | 4.3 | 37% | measured |
| TP collective non-overlapped | 1.5 | 13% | measured |
| KV update / graph staging | 1.0 | 9% | bounded |
| sampling/host/sync | 1.2 | 10% | bounded |
| other / unresolved | 1.5 | 13% | remainder |

The example is entirely **fictional**; produce percentages only when the actual non-overlapping critical-path classification supports them. A simple kernel-service histogram does not satisfy this deliverable.

### 5.5 Diagnosis of repeated overhead

For every candidate overhead, report:

1. frequency per **request / block / refinement / layer / rank**;
2. payload: real vs logical bytes, affected token rows, KV key length, tensor shapes, graph key;
3. CPU time, GPU kernel service time, non-overlapped critical-path time (each separately);
4. overlap with compute and stream/rank idle windows;
5. change across prefix lengths and early/mid/late iterations;
6. simple removal/amortization **upper bound** and independently measurable exact micro-control;
7. what native runtime or prior already optimizes.

For `memcpy128`, check kernel callers, source/destination, whether it is graph input staging, out-of-place `slice_scatter`, layout, or unrelated. **Do not assert that it is a PCIe or HBM bandwidth problem solely from its name.**

### 5.6 Per-iteration / per-block visualization

Required:

- **Four-rank Gantt timeline** for a representative request (host launch, GPU compute, NCCL, memory copies, idle/waits, block/refinement boundaries); color categories only if trace mapping is reliable.
- **Block × refinement heatmap** of actual wall/replay latency; include number of committed tokens per iteration, current block index and actual NFE.
- **Layer × category latency heatmap**, explicitly labeled kernel service or exclusive critical-path category.
- **Per-iteration stacked wall-critical chart** with `unknown/overlap` separated, *not* double counting.
- **Prefix length vs** clean cold E2E, warm generation, prefill, attention/comm/copy/sampling critical-path cost and conditional headroom.
- **GPU 4/5/6/7 utilization and idle-gap timeline**; hottest rank/wait rank and synchronization frequency.

## 6. Stage A gate — Is there a real research-sized cost?

Do not invent methods yet. Promote exactly one top bottleneck only when:

- ≥ **15% of clean E2E** is a credible *non-overlapped* removable target, OR a conservative realistic twofold improvement gives ≥ **10% clean E2E** expected reduction;
- the repeated overhead is stable across multiple requests/configurations rather than one anomalous graph/profile;
- precise cause and physical operation can be identified;
- optimized native runtime / existing work does not already provide equivalent removal;
- a low-risk implementation or architecture mechanism is plausible within four GPUs.

Thresholds are exploratory kill-gates, not measured findings. If timing permits only an upper bound, use `CONDITIONAL_MEASUREMENT` rather than GO.

Hard stop if: stage attribution untrustworthy, no persistent relevant cost, or candidate headroom <10% cold E2E and <10% warm-span reduction (unless a strong, explicitly stated throughput-oriented problem exists). Preserve profiling result as a reusable infrastructure contribution, but do **not** present it as a novel systems method.

## 7. Stage B — Non-TP architecture screening (required, but mostly analytical first)

The user explicitly invites **new multi-GPU inference architectures**. Do not prematurely lock into TP. First screen architectures using real Stage A timings and model weight/HBM feasibility.

| Candidate | Physical 4GPU layout | Mechanism | Main risk | Best comparison regime |
|---|---|---|---|---|
| Native TP4 | TP4×DP1×PP1, EP1 | Baseline: head/model tensor sharding | Layer collectives / small GEMMs | B1 and B2/B4 |
| TP2×DP2 | 2 independent TP2 replicas, each on 2 GPUs | fewer TP collectives per replica; independent request parallelism | 2× model replication footprint; may not fit; B1 gains not guaranteed | concurrency B2/B4, aggregate TPS/QPS |
| TP1×DP4 | 4 standalone complete replicas | eliminates inter-rank TP communication within request | model/HBM/cuda-graph replication; very different service objective | B4 throughput only if fits |
| PP2×TP2 | 2 pipeline stages each using TP2 | lower TP degree / fewer intra-layer collectives; layer-group communication | B1 bubble, KV/cache boundary, stage balance, model fit | B1 only if measured; B2/B4 pipeline |
| PP4×TP1 | 4 layer pipeline stages | layer-boundary activations rather than TP reductions | B1 cannot overlap successive dependent refinements; severe bubble | B2/B4 or other independent microbatches |
| Existing hybrid DP/TP/PP | only supported exact variants | practical baseline | invalid performance if implementation changed | appropriate matched regime |
| Context/sequence sharding | CP/SP (non-assumed) | distribute long KV or tokens | previous CP prototype slower; repeated collectives | only if **new** critical-path evidence |
| Decoupled control/sampling vs GPU compute | scheduler/control plane optimized across CPU/GPU, no free concurrent work | remove host waits and replay gaps | hidden sync, stale data, semantics | B1 if host-stall critical |
| Split execution for independent requests or blocks | multi-GPU microbatch/phase overlap | fill pipeline idle slots | same-request iteration dependency; data movement | B2/B4 and truly independent work |

### 7.1 Feasibility math and constraints

- On four GPUs, conventional orthogonal dimensions satisfy **TP × PP × DP × CP = 4** (except explicitly documented rank-reusing techniques); **EP** may reuse TP ranks only if the runtime really supports that layout. Do not assume TP4×PP4 fits on four GPUs.
- Check BF16 weights, KV and CUDA Graph workspace under every stage/replica. The model might fit on one GPU in theory but fail after graph capture; **actually verify**.
- A plain PP4 pipeline with **one request / one dependent refinement** has effectively no useful pipeline fill across successive iterations. Show this dependency explicitly; do not present ideal 4× stage throughput as B1 latency speedup.
- PP can reduce inter-layer communication frequency but B1 activations must still traverse all stages serially, often with stage bubbles. Use measured layer stage times, activation-boundary transfer and achievable microbatch counts, not zero-cost PP.
- TP2×DP2 and DP4 mainly address **multi-request throughput**. For B1 they can leave GPUs idle; do not label higher aggregate B4 throughput as lower latency for the same B1 request.
- Probe realistic simultaneous requests only on frozen input and exact model semantics. Distinguish *per-request latency, aggregate TPS, completed requests/s, P50/P95/P99,* and memory.
- Prioritize existing runtime configurations when supported; otherwise run tiny physically validated prototypes and honest oracle bounds. Avoid a large new engine for Stage B.

### 7.2 Three architecture filters

**Filter A: capacity & correctness:** actual weights, KV, cache graphs fit; no unauthorized precision change; checkpoint identical; outputs/acceptance/NFE parity pass under same settings, or precisely explain documented numeric divergence.

**Filter B: calibrated performance model:** recompute candidate estimate using observed stage/collective/transfer duration; include graph/copy costs, pipeline bubbles, inter-rank transfers, request schedulers, HBM and startup amortization. Validate at least two predictions against a real control if feasible. Bounds that omit communication are named `UNREALIZABLE_ORACLE`.

**Filter C: measured end-to-end benefit:** only test a candidate with conservative E2E or aggregate-throughput improvement potential ≥10–15% and meaningful novelty beyond existing TP/PP/DP serving. Choose **one** candidate, not all.

### 7.3 If screening ranks a feasible candidate first

Run minimal physically correct experiment with one factor changed (e.g. TP4 vs supported TP2/DP2 across B2 requests, or baseline vs safe graph staging). Validate clean E2E/throughput and exact or justified numerical/functional parity. If runtime lacks the layout, do not build an engine by default. Report `ARCHITECTURE_NOT_AVAILABLE` or `PREDICTION_ONLY`, not GO.

## 8. Stage C — One bounded causal intervention (optional)

Run only after Stage A & Stage B gates. Candidate classes:

- **Exact local KV canvas-write/cache-staging fix** if event+DAG confirm real full-model critical-path impact, memory semantics and graph lifetime permit safe in-place writes. Earlier isolated 2.880 vs 4.094 ms is *not* E2E evidence and is not inherently multi-GPU novel.
- **Reduce a measured CPU synchronization or graph-replay idle boundary** without changing acceptance / NFE / mask; must verify no hidden asynchronous race.
- **Physically supported alternative parallel layout** from Stage B whose gains survive communication/PP bubbles and memory cost.
- **Work-preserving inter-request or independent-block pipeline fill** if independent work is actually available. A single request's successive iterations are causally dependent and cannot be arbitrarily overlapped.

Forbidden without independent justification: speculative/proxy stale KV, approximate sampling, expert replication/migration, synthetic graph skipping, invalid precision reduction, cross-step stale hidden states, or a new CP prototype simply to rescue the closed result.

**Causal proof** must include: intervention-only change, same prompts/checkpoint/seeds, clean and coarse lanes, exact/carefully justified output+acceptance+NFE parity, buffer ownership correctness across CUDA Graph replay, no extra graph capture included as hidden free work, clean cold E2E and warm span, representative held-out, and memory.

## 9. Statistical and validation protocol

- Paired comparisons on the **same request IDs**, request as bootstrap unit; >=2,000 request bootstrap resamples after any calibration; report CI and per-request scatter. Three repeated timings per request are technical repeats, **not independent samples**.
- Randomize/counterbalance arm order within same session when valid, measure temperature/clocks/interference; do not mix different GPU sessions or alternate GPU IDs.
- Instrumentation ON/OFF on the same configuration: exact token IDs, acceptance timeline, NFE, EOS and quality. For a distinct valid parallel architecture, use strict parity where numerical reduction ordering permits; otherwise prespecify numeric/functional tolerances and evaluate quality; **do not hide legitimate divergence**.
- GPU asynchronous timing: don't include sync waits as GPU kernel compute. Record device graph completion before declaring request done. Avoid accumulating per-layer CUPTI overhead into clean timing.
- Prefix lengths may change content, NFE and output length; their E2E difference is **not** solely attributable to sequence length without matched-input controls.
- Keep profiler/tracer overhead and timestamp cross-rank alignment error visible.
- `all-to-all %` should be `0 / NOT_PRESENT` under TP4/EP1 if verified; do not relabel all-reduce as A2A.

## 10. Concrete outputs

```text
poc_dllm_multigpu_e2e_atlas/
  scripts/
    inspect_runtime_and_collectives.py
    freeze_workloads.py
    run_clean_baseline.py
    profile_coarse_timeline.py
    profile_nsys_scoped.sh
    parse_timeline.py
    analyze_critical_path.py
    analyze_repeated_overheads.py
    evaluate_architecture_options.py
    run_optional_physical_control.py
    validate_parity.py
  tests/
  manifests/
    environment.json
    model_geometry.json
    workload_calibration.json
    workload_heldout.json
    timing_validity.json
  results/
    per_request.csv
    per_block.csv
    per_refinement.csv
    per_layer_or_graph_node.csv
    per_rank_collectives.csv
    gpu_kernel_service.csv
    copy_and_memory_classification.csv
    critical_path_intervals.csv
    e2e_phase_budget.csv
    architecture_screening.csv
    bootstrap_cis.json
    raw/  # compact, bounded evidence; large nsys files may be external
  figures/
    four_rank_timeline.png
    per_iteration_wall_budget.png
    block_refinement_heatmap.png
    layer_category_heatmap.png
    sync_idle_gaps.png
    prefix_scaling.png
    architecture_tradeoff.png
  timing_contract.md
  prior_overlap.md
  reproduce.md
  report.md
```

Create only legitimate populated files. If a stage did not run, retain `NOT_RUN` rather than placeholder success files. Preserve prior branch / staged / dirty files.

### Required summary tables

**A. Full request budget:** P, request length, output, NFE, prefill ms/%, refinement ms/%, block refresh ms/%, sampling/control ms/%, GPU/CPU gap ms/%, unresolved ms/%, cold E2E, warm span, TPS, timing validity.

**B. Per-refinement budget:** request, block, iteration, physical P, committed tokens, TP collectives (count/non-overlapped time), actual A2A count, attention+projection, expert/FFN, KV/copy, sampling/control, GPU idle, total, attribution confidence.

**C. Repeated-cost suspects:** operation, calls per iteration, aggregate service ms, critical-path ms (or UNRESOLVED), size, skipability/cost, estimated *achievable* max gain, closest prior.

**D. Architecture options:** layout, GPU count, actual supported/feasible, B1 latency model, B2/B4 throughput model, PP bubble, communication, HBM, measured or optimistic, novelty issue, verdict.

### Report's final answers

1. Was the native TP4/EP1 topology and absence/presence of A2A **physically verified**?
2. What exact portion of **clean E2E** is prefill, refinement, block-refresh, sampling/control, and residual?
3. For an *average and representative* refinement: how many ms and what defensible percentage are attributable to attention, MoE/FFN, TP collectives, KV/copy, host launch/sync, and idle? Which are `UNRESOLVED` because of overlap?
4. Does timing vary by iteration phase (early/mid/late), physical context length, blocks, and GPU rank?
5. Are `memcpy128` kernels graph staging, KV mutation, or something else? What fraction of **critical path**, not summed service, do they explain?
6. What are exact physical NCCL calls and non-overlapped waits? Is a reported "communication overhead" a true critical-path cost?
7. Which single bottleneck offers the biggest **measured, realistically removable E2E headroom**?
8. Do TP2×DP2, PP2×TP2 or PP4 have a feasible operating regime with four GPUs, after bubbles/memory/correctness? Compare B1 latency separately from B2/B4 aggregate throughput.
9. Is the most promising opportunity already solved by CUDA Graphs, DCP, FlashBlock, Epoch, BlockServe, Denoising Surface, FastPP or another prior?
10. If an intervention ran, what is the measured clean E2E/TPS gain and parity/accuracy? If none, why?
11. Give final verdict and *one* best follow-up, or close the direction explicitly.

## 11. Verdict taxonomy

- `GO_B1`: measured valid physically plausible B1 E2E improvement >=~10% with clean requests and correctness, distinct enough from prior for research.
- `GO_THROUGHPUT_ONLY`: independent multi-request throughput/goodput improvement is compelling, while clearly **not** claiming B1 latency.
- `CONDITIONAL`: a credible measurable critical-path bottleneck with robust realistic upper bound, but method/physical implementation still unproven.
- `PROFILE_ONLY`: a useful validated atlas but no convincing independent optimization direction.
- `NO_SIGNIFICANT_BOTTLENECK`: candidate removable critical-path share too small.
- `GENERIC_ONLY`: issue known/handled by existing serving/graph/PP/TP optimizations without dLLM-specific gap.
- `RUNTIME_BLOCKED`: model/hardware layout unsupported without an engine rewrite.
- `TIMING_INVALID`: observer/graph/timestamp validity prevents attribution.
- `NO_GO`: physically tested intervention slower or no meaningful E2E/throughput improvement.

No structural-volume oracle, kernel count, or service-time share alone is sufficient for GO.

## 12. Prior and source checklist (must confirm at runtime)

- NVIDIA **Nsight Systems CUDA Graph tracing / per-node profiling / event-tracing caveats**: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- NVIDIA **CUDA Graph performance / static input copy / idle-gap guidance**: https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html
- PyTorch **CUDA Graph static storage and replay requirements**: https://docs.pytorch.org/docs/stable/notes/cuda
- **Revisiting Pipeline Parallelism for LLM Serving**, OSDI 2026 (FastPP): https://www.usenix.org/conference/osdi26/presentation/hwang
- **BlockServe**, July 2026: https://arxiv.org/abs/2607.08930
- **Denoising Surface**, September 2026: https://arxiv.org/abs/2610.00499
- **Epoch**, September 2026: https://arxiv.org/abs/2609.09748
- **FlashBlock / LoSA** and original dLLM prefix–canvas CP project prior overlap (links and distinctions there).

These works constrain novelty: generic profiling, adaptive PP scheduling, multi-request denoising cost prediction, block-grained batching, sparse-MoE block compilation and attention reuse are **not** automatically new. A contribution must exploit a remaining, physically demonstrated *dLLM × multi-GPU* execution gap beyond these baselines.

## 13. Execution hygiene and final handoff

- Physical GPU IDs **4,5,6,7 only**. Save complete device UUIDs and topology in evidence.
- Never reset/clean unrelated work; use new isolated branch/output directory. Do not terminate other users' work or change global server settings.
- Keep the prior validated runtime unchanged unless a narrow, reversible instrumentation patch is unavoidable. Save diffs and hashes.
- Avoid storing huge Nsight artifacts or untrusted datasets in Git; preserve report pointers and checksums to external local evidence.
- Keep failed or interrupted logs and identify non-headline invalid runs.
- Tests, raw data, report, figures and reproducing commands must be committed/pushed **only for this PoC**. Record branch, SHA and report path.
- No background/autonomous future follow-up: finish current bounded PoC or document explicit blocker.

**Research decision rule:** First produce credible, low-overhead real E2E/iteration timelines; only then choose what to optimize. Prefer closing another weak systems hypothesis to building an elaborate method around a non-critical metric.
