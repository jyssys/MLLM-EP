# FlashVEP PoC Specification
## Exact Vision-Tile Wavefront Execution for Qwen3-VL-30B MoE Expert Parallel Prefill

**Status:** PoC specification  
**Target model:** Qwen3-VL-30B-A3B or the exact Qwen3-VL-30B MoE checkpoint already used in the repository  
**Primary environment:** Existing multi-GPU EP codebase; expected 8 GPUs, but the implementation must discover and record the actual configuration  
**Primary target:** Batch-1 multimodal prefill latency, especially vision-induced EP tail latency  
**Correctness policy:** Exact model semantics; no token dropping, token merging, quantization, rerouting, surrogate outputs, or model retraining

---

## 1. Motivation

Existing profiling indicates that:

1. Visual tokens dominate the multimodal prefill sequence.
2. Visual routing creates severe per-rank expert imbalance.
3. The slowest EP rank determines the MoE-layer completion time.
4. Conventional execution waits for large layer-wide operations to finish before starting the next stage:
   - attention,
   - residual/norm/router,
   - dispatch All-to-All,
   - expert computation,
   - combine All-to-All.

FlashVEP investigates whether the same exact work can finish earlier by changing the execution unit from a full visual sequence to a stream of visual-token tiles.

The intended execution pattern is:

```text
Tile 0: Attention -> Norm/Router -> Dispatch -> Expert -> Combine
Tile 1:              Attention -> Norm/Router -> Dispatch -> Expert -> Combine
Tile 2:                           Attention -> Norm/Router -> Dispatch -> Expert
...
```

The method does **not** reduce the total expert FLOPs. Its goal is to:

- produce the first critical expert tile earlier;
- keep the critical EP rank's expert queue continuously supplied;
- overlap dispatch with later attention/producer work;
- overlap combine with later expert work;
- eventually eliminate unnecessary layer-wide barriers and intermediate materialization.

---

## 2. Relationship to ScMoE and Prior Pipelining

Reference paper:

- *Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts*
- arXiv: https://arxiv.org/pdf/2404.05019

Figure 7 contains a relevant standard-MoE pipelining baseline that chunks input tokens and overlaps communication and computation. ScMoE goes further by modifying the MoE architecture: it creates a shortcut-connected routed-expert stream and a shared-expert/backbone stream so that communication can be decoupled from the conventional sequential dependency.

### Important novelty boundary

The following alone is **not** sufficient novelty:

> Split MoE inputs into chunks and overlap All-to-All communication with expert computation.

That is established prior art and appears explicitly in the ScMoE Figure 7 comparison.

The FlashVEP research hypothesis must instead be evaluated around the following combined properties:

1. **Pretrained-model exactness:** keep the original Qwen3-VL computation graph and weights unchanged.
2. **Single-request multimodal prefill:** target one vision-heavy request, not only large training batches.
3. **Attention-to-MoE wavefront:** extend streaming upstream from MoE dispatch toward attention-output tiles.
4. **Vision-aware scheduling:** prioritize tiles expected to feed the critical vision-heavy rank.
5. **Global multi-rank scheduling:** optimize the maximum EP-rank completion time, not one local expert only.
6. **Barrier/materialization removal:** investigate direct tile handoff instead of only launching more chunked kernels.

The initial PoC is allowed to reproduce a prior-art-like chunked pipeline. Its purpose is to measure headroom and determine whether the more novel end-to-end design is technically justified.

---

## 3. Main Research Questions

### RQ1: Baseline headroom
How much layer time is currently exposed outside the critical-rank expert execution?

For each layer and EP rank, measure:

- QKV projection, if separable;
- attention core;
- attention output projection;
- residual and RMSNorm;
- router;
- dispatch All-to-All;
- local expert grouped GEMM;
- combine All-to-All;
- full layer wall time.

### RQ2: Earliest valid expert-tile arrival
How early can the first exact critical tile arrive at the straggler rank?

Define:

```text
T_first = time from layer start until the first valid critical tile
          is available for local expert execution.
```

This must include all real dependencies, including any full-K/V preparation barrier.

### RQ3: Producer-consumer balance
Can the upstream tile producer keep the critical expert consumer busy?

For tile `k` on rank `g`:

```text
arrival[k, g] = time the dispatched tile becomes available on rank g
service[k, g] = measured expert execution time for that tile on rank g
```

The critical queue avoids starvation when:

```text
arrival[k, g] <= finish[k-1, g]
```

The steady-state producer initiation interval should ideally satisfy:

```text
producer_interval <= critical_rank_tile_service_time
```

### RQ4: Fragmentation penalty
How much slower do expert kernels and All-to-All become when the full batch is split into tiles?

### RQ5: Multi-rank behavior
Does prioritizing one vision-heavy tile create network congestion or a new straggler on another rank?

### RQ6: End-to-end exactness
Can all tile outputs be restored to original token order with equivalent layer outputs and generated tokens?

---

## 4. Explicit Non-Goals

Do not add any of the following during the initial PoC:

- token pruning, merging, or dropping;
- reduced precision beyond the existing baseline;
- changed router decisions or local rerouting;
- expert placement or expert replication;
- speculative decoding or surrogate expert outputs;
- model training or fine-tuning;
- production CUDA-kernel rewrites before the profiling gates pass;
- changes to the existing baseline's default behavior.

Do not claim FlashVEP novelty or speedup from trace simulation alone.

---

## 5. Development Principles

1. Inspect the repository before assuming the framework, versions, launch commands, or source layout.
2. Reuse the existing straggler-profiling infrastructure.
3. Keep all PoC code isolated and disable it by default.
4. Preserve the existing baseline command and output.
5. Add feature flags rather than hard-coded behavior.
6. Work phase by phase and stop at each acceptance gate.
7. Prefer measurement over assumptions.
8. Record failed experiments and blockers.
9. Never hide tiling overhead from reported latency.
10. Report wall-clock latency and per-rank critical path, not only kernel sum.

---

## 6. Phase 0 — Repository and Environment Audit

### Tasks

1. Locate the repository root and read:
   - README files;
   - launch scripts;
   - environment files;
   - current profiling/straggler scripts;
   - local modifications.
2. Identify and record:
   - Git commit and dirty status;
   - model/checkpoint path or model identifier;
   - inference framework and exact version;
   - PyTorch, CUDA, NCCL, Triton, FlashAttention, and vLLM versions where applicable;
   - GPU model, count, memory, and topology;
   - EP, TP, DP, and pipeline configuration;
   - current launch command;
   - benchmark dataset/input format;
   - how vision-token ranges are identified.
3. Run a minimal baseline smoke test without modifying behavior.
4. Save a concise repository map and execution map.
5. Do not delete or rewrite existing result files.

### Deliverables

```text
poc_flashvep/
  STATUS.md
  repo_audit.md
  env_snapshot.txt
  baseline_command.sh
```

### Exit gate

- The existing baseline runs successfully.
- The exact model and parallel configuration are known.
- Existing straggler instrumentation is identified.
- No baseline output changed.

---

## 7. Phase 1 — Exact Baseline Profiling

### 7.1 Instrumentation

Use low-overhead NVTX ranges and CUDA events. Use Nsight Systems when available.

Add per-layer ranges for:

```text
layer/<L>/qkv_projection
layer/<L>/attention_core
layer/<L>/attention_output_projection
layer/<L>/residual_norm
layer/<L>/router
layer/<L>/dispatch_a2a
layer/<L>/expert
layer/<L>/combine_a2a
layer/<L>/total
```

If a stage cannot be separated due to fusion, record the fused stage honestly.

For expert execution, record per rank:

- received routed token assignments;
- visual and text assignment counts if modality labels are available;
- per-local-expert assignment counts;
- start and finish CUDA timestamps;
- maximum local expert batch;
- total expert-kernel duration.

### 7.2 Experiment protocol

Default, unless the repository already has a stricter protocol:

- batch size: 1;
- generation: prefill-focused, preferably `max_new_tokens=1`;
- warm-up: at least 5 iterations;
- measured iterations: at least 20;
- deterministic fixed inputs;
- synchronize only at measurement boundaries;
- report median, p90, mean, and standard deviation;
- run at least three visual-token buckets when possible:
  - small;
  - typical;
  - large/straggler-heavy.

Do not invent image resolutions. Derive valid inputs from the existing benchmark or processor.

### 7.3 Trace schema

Write machine-readable JSONL or CSV with fields such as:

```text
run_id
request_id
iteration
rank
layer
stage
start_us
end_us
duration_us
vision_token_count
text_token_count
routed_token_count
local_expert_id
local_expert_token_count
ep_size
tp_size
gpu_name
```

### 7.4 Required analysis

For every MoE layer compute:

```text
T_layer
T_attention
T_norm_router
T_dispatch
T_expert_max
T_combine
critical_rank
expert_fraction = T_expert_max / T_layer
exposed_nonexpert_fraction
```

Create two upper-bound estimates:

#### Optimistic overlap bound

```text
T_optimistic = max(T_expert_max, T_attention + T_norm_router + T_dispatch)
               + exposed_combine_tail
```

#### Conservative first-tile bound

Use measured or estimated first-tile production and fragmentation costs:

```text
T_conservative = T_first
                 + tiled_critical_expert_span
                 + exposed_combine_tail
                 + measured_tiling_overhead
```

### Deliverables

```text
poc_flashvep/
  instrumentation/
  results/baseline/*.jsonl
  results/baseline/summary.csv
  results/baseline/layer_breakdown.csv
  reports/phase1_profile.md
  scripts/run_baseline_profile.sh
```

### Go/no-go gate

Proceed to Phase 2 only if at least one representative vision-heavy case satisfies both:

- estimated oracle layer speedup is at least `1.15x`; and
- non-expert exposed time is at least `15%` of the MoE-layer wall time.

If these conditions fail, stop and document why FlashVEP is unlikely to provide meaningful speedup.

---

## 8. Phase 2 — Trace-Driven Tile Simulator and Offline Replay

This phase must not modify live attention execution.

### 8.1 Token classification

Identify exact token indices for:

- visual tokens;
- text prompt tokens;
- any special multimodal boundary tokens.

Assert that the classification covers the sequence exactly and does not rely on guessed token IDs.

### 8.2 Tile definitions

Start with contiguous visual sequence tiles:

```text
tile_sizes = [64, 128, 256, 512]
```

If valid 2-D patch metadata is available, also support rectangular spatial tiles. Preserve original token indices.

Text tokens should remain on the baseline path initially, or be placed in one explicit non-visual tile. Do not silently mix modality definitions.

### 8.3 Capture points

Capture the exact tensor entering the MoE sublayer after the attention/residual path, together with:

- router logits or top-k routes;
- routing weights;
- destination rank;
- expert ID;
- original token index;
- modality label.

Avoid persistent activation dumps for all layers if memory is excessive. Support selected-layer capture.

### 8.4 Simulator

Implement a discrete-event simulator that consumes measured stage timings and tile route distributions.

It must model:

- tile ready time;
- per-rank dispatch arrival;
- per-rank expert FIFO or priority queue;
- per-tile local expert service;
- combine readiness;
- layer makespan;
- critical rank changes;
- queue idle gaps;
- fill and drain overhead.

Scheduler modes:

1. `original_order`
2. `reverse_order`
3. `random_order` with fixed seed
4. `oracle_critical_first`
   - may use same-layer observed routing;
   - explicitly label as non-deployable upper bound.
5. `previous_layer_prediction`
   - optional in Phase 2;
   - uses only information available before current-layer routing.

### 8.5 Offline exact replay

Where feasible, replay the MoE path tile by tile using the captured MoE input.

Compare:

- full-batch MoE output;
- concatenated/reordered tile outputs.

Correctness:

```text
torch.testing.assert_close(...)
```

Use tolerances derived from the baseline dtype and operation ordering. Also report maximum absolute error and cosine similarity. Do not call the method exact if generated tokens differ under the same decoding mode.

### 8.6 Required metrics

For every layer, rank, scheduler, and tile size:

```text
T_first
producer_interval
critical_rank
critical_rank_queue_utilization
critical_rank_idle_gap
total_expert_compute_sum
tiled_expert_span
expert_fragmentation_penalty
dispatch_message_count
dispatch_overhead
combine_exposed_tail
simulated_layer_latency
predicted_speedup
```

### Deliverables

```text
poc_flashvep/
  flashvep/trace_schema.py
  flashvep/tiling.py
  flashvep/simulator.py
  flashvep/schedulers.py
  flashvep/replay.py
  tests/test_tiling_roundtrip.py
  tests/test_replay_correctness.py
  reports/phase2_tile_replay.md
```

### Go/no-go gate

Proceed to live overlap only when a representative configuration achieves:

- critical-rank queue utilization at least `90%`;
- expert fragmentation penalty below `10%`;
- first critical-tile arrival below `60%` of the baseline full-batch arrival;
- predicted layer speedup at least `1.15x`;
- exact output reconstruction within approved tolerance.

---

## 9. Phase 3 — Live Post-Attention Wavefront Prototype

This phase begins after the full attention output exists. It tests the downstream pipeline:

```text
Norm/Router -> Dispatch -> Expert -> Combine
```

It does not yet claim attention-to-MoE streaming.

### 9.1 Implementations to compare

1. `full_batch_baseline`
2. `tiled_serial`
   - tiles but no overlap;
   - isolates fragmentation overhead.
3. `tiled_overlap_fixed_order`
4. `tiled_overlap_oracle_order`
   - upper bound only.

### 9.2 Multi-GPU safety

- All ranks must use a consistent global tile order for collective operations.
- Do not launch collectives in rank-local orders.
- Use explicit CUDA-event dependencies.
- Avoid hidden default-stream synchronization.
- Use existing nonblocking communication APIs when supported.
- If the backend exposes only a monolithic blocking All-to-All, stop and document the limitation rather than replacing the communication backend in this phase.

### 9.3 Streams

Start with:

```text
producer_stream
expert_stream
communication_stream
```

Actual concurrent execution must be validated with Nsight Systems. Enqueue overlap alone is not evidence of hardware overlap.

### 9.4 Correctness

Validate:

- per-layer hidden state;
- final logits;
- greedy generated token sequence;
- routing identity relative to baseline.

### Exit gate

Continue only if actual post-attention wall-clock speedup is at least `1.10x` on a representative vision-heavy input and output correctness is preserved.

---

## 10. Phase 4 — Attention-to-MoE Feasibility

This phase determines whether a valid attention output tile can be released before the entire attention operation finishes.

### 10.1 Dependency audit

Determine whether the current attention implementation requires:

- full QKV projection before attention;
- full K/V availability;
- full sequence output materialization;
- a monolithic FlashAttention call.

Record the exact first valid handoff point.

### 10.2 Candidate prototype

Prefer this order:

1. Precompute exact full K/V.
2. Execute query-row attention in exact tiles against valid K/V ranges.
3. Preserve causal masking, multimodal position IDs, sequence metadata, and output order.
4. Pass completed query tiles to the Phase 3 downstream wavefront.

Do not implement a custom FlashAttention kernel until the exact query-tile prototype proves enough headroom.

### 10.3 Required comparison

```text
full_attention
query_tiled_attention_serial
query_tiled_attention_plus_downstream_wavefront
```

Measure:

- first query-tile output;
- full attention completion;
- attention fragmentation penalty;
- Tensor Core contention with expert work;
- first expert arrival;
- final layer wall time.

### Final main-method gate

A serious FlashVEP implementation is justified only when the exact end-to-end prototype demonstrates:

- at least `1.15x` prefill speedup over the strong current EP baseline;
- unchanged greedy outputs;
- no hidden reduction in work or precision;
- benefit across more than one visual-token length;
- no regression on short/text-heavy inputs when disabled or adaptively bypassed.

---

## 11. Phase 5 — Vision-Aware Tile Scheduling

Implement only after fixed-order wavefront execution is proven useful.

### 11.1 Scheduler inputs

Potential deployable signals:

- previous-layer tile-to-rank routing;
- previous-layer tile-to-expert counts;
- current rank queue depth;
- measured service-time model;
- tile age;
- predicted network bytes;
- 2-D spatial neighborhood routing consistency.

### 11.2 Required schedulers

1. natural spatial/sequence order;
2. previous-layer critical-rank prediction;
3. global backlog-aware scheduler;
4. same-layer oracle scheduler as upper bound.

### 11.3 Suggested score

A starting heuristic:

```text
score(tile) =
    predicted_work_on_current_critical_rank
    - lambda_network * predicted_destination_congestion
    - lambda_fragment * predicted_small_batch_penalty
    + lambda_age * waiting_age
```

The actual implementation may change after profiling.

### 11.4 Correctness restriction

Tile scheduling may change execution order only. It must not change:

- token position;
- attention dependencies;
- router top-k results;
- expert weights;
- combine weights;
- final output ordering.

---

## 12. Experimental Matrix

Minimum configurations where supported:

### Parallelism

```text
EP = current production value, expected 8
optional comparison: EP = 4
batch = 1
```

### Input buckets

Use the existing benchmark to choose at least:

```text
small visual prefix
typical visual prefix
large visual prefix
```

Record actual visual and text token counts.

### Tile sizes

```text
64, 128, 256, 512, full
```

### Methods

```text
baseline
tiled_serial
tiled_overlap_fixed
tiled_overlap_oracle
tiled_overlap_predicted
```

### Report

- end-to-end prefill latency;
- per-layer latency;
- median and p90;
- speedup;
- per-rank completion time;
- critical-rank identity;
- first tile arrival;
- queue utilization;
- exposed idle;
- tile fragmentation penalty;
- dispatch/combine communication;
- correctness.

---

## 13. Expected Repository Layout

Adapt to the existing repository rather than forcing this exact layout.

```text
poc_flashvep/
├── README.md
├── STATUS.md
├── repo_audit.md
├── env_snapshot.txt
├── configs/
│   ├── profile.yaml
│   └── tile_sweep.yaml
├── flashvep/
│   ├── instrumentation.py
│   ├── trace_schema.py
│   ├── token_metadata.py
│   ├── tiling.py
│   ├── simulator.py
│   ├── schedulers.py
│   └── replay.py
├── scripts/
│   ├── run_baseline_profile.sh
│   ├── run_tile_simulator.sh
│   └── run_replay.sh
├── tests/
│   ├── test_token_partition.py
│   ├── test_tiling_roundtrip.py
│   └── test_replay_correctness.py
├── results/
└── reports/
```

---

## 14. Coding and Safety Requirements

- Type annotate new Python code.
- Add concise docstrings.
- Fail loudly on missing modality metadata.
- Never guess visual-token boundaries.
- Use configuration files or CLI flags.
- Set deterministic seeds where relevant.
- Do not run destructive shell commands.
- Do not overwrite existing experiments.
- Do not commit model weights, activation dumps, or large traces.
- Add `.gitignore` entries for large PoC artifacts.
- Preserve baseline behavior when FlashVEP flags are disabled.
- Each phase must update `STATUS.md` with:
  - completed tasks;
  - commands run;
  - measurements;
  - blockers;
  - next recommendation.

---

## 15. Stop Conditions

Stop the PoC and report a negative result when any of the following is observed:

1. Oracle overlap headroom is below 10%.
2. Existing runtime already overlaps nearly all hideable work.
3. The full-K/V or full-attention barrier keeps `T_first` near the baseline full-batch arrival.
4. Tiled expert kernels slow down by more than 20%.
5. Small All-to-All messages dominate tile execution.
6. Attention/expert concurrency creates severe Tensor Core contention.
7. Exact outputs cannot be preserved.
8. Speedup appears only against an artificially weak baseline.

A negative PoC is a valid result and should be documented without forcing an implementation.

---

## 16. Final PoC Success Criteria

The PoC is considered promising when it demonstrates all of the following:

1. Exact baseline reproduction.
2. Trustworthy per-stage and per-rank profiling.
3. A measurable early critical-tile arrival.
4. Critical-rank queue utilization of at least 90%.
5. Less than 10% expert fragmentation penalty at the selected tile size.
6. Actual prefill speedup of at least 1.15x against the current strong baseline.
7. Identical greedy output tokens.
8. Benefits on multiple vision-token lengths.
9. A clear distinction from generic MoE chunked pipelining:
   - attention-to-MoE handoff,
   - vision-aware scheduling,
   - and/or barrier/materialization elimination.

