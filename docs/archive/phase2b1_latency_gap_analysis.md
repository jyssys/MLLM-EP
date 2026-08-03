# Phase 2-B1 Latency Gap Analysis

Date: 2026-06-25

## Question

The current layer-wise placement clearly reduces routed-load stragglers, but clean end-to-end wall-clock and TTFT do not improve. This is concerning because ReaLB and MACS report meaningful latency or throughput gains from mitigating MoE stragglers.

This note explains what we measured, how it differs from ReaLB/MACS, why E2E speedup is missing so far, and what I think we should do next.

## Current Result

Batch-32 profile setup:

- Model: Qwen3-VL-30B-A3B-Instruct
- Runtime: vLLM 8-way EP
- Batch size: 32
- `max_tokens=1`
- Data: ChartQA 64, MMMU 64, MMStar 64
- Total samples: 192
- Total prefill tokens: 313,440
- Number of measured batches: 6
- Prefix cache: disabled

Token scale:

| Batch | Dataset | Prefill tokens | Tokens/sample |
|---:|---|---:|---:|
| 0 | ChartQA | 37,106 | 1,159.6 |
| 1 | ChartQA | 28,767 | 899.0 |
| 2 | MMMU | 38,031 | 1,188.5 |
| 3 | MMMU | 96,065 | 3,002.0 |
| 4 | MMStar | 81,352 | 2,542.2 |
| 5 | MMStar | 32,119 | 1,003.7 |

Summary:

- Mean prefill tokens/batch: 52,240
- Median prefill tokens/batch: 37,568.5
- Mean tokens/sample: 1,632.5
- Total routed assignments: 120.36M = `tokens * 48 layers * top-8`

Main results:

| Metric | As-Is linear | To-Be layer-wise | Change |
|---|---:|---:|---:|
| batch-layer p95 load imbalance | 1.5033 | 1.0352 | -31.14% |
| batch-layer max load imbalance | 1.6309 | 1.0536 | -35.40% |
| layer-total p95 rank imbalance | 1.2917 | 1.0161 | -21.33% |
| MoE CUDA critical path total | 4588.74 ms | 4159.88 ms | -9.35% |
| clean batch wall time | 42.3955 s | 42.8201 s | +1.00% slower |
| clean mean TTFT | 3.1873 s | 3.3593 s | +5.40% slower |
| clean mean scheduled-to-first-token | 1.1335 s | 1.2558 s | +10.78% slower |

The load-balancing part works. The E2E latency part does not yet work.

## What ReaLB/MACS Actually Measure

ReaLB reports both fine-grained MoE-layer timing and vLLM end-to-end throughput. The paper says it uses CUDA events for fine-grained timing and vLLM benchmark throughput measured as processed input+output tokens/s. It reports three metrics: accuracy degradation, end-to-end throughput speedup, and MoE layer latency.

ReaLB also explicitly targets large-batch prefill where MoE execution is GEMM/compute-bound, and it activates load balancing only when the aggregated batch load exceeds a threshold. Their method is not just expert placement: it uses online per-rank precision adaptation, often FP4 on vision-heavy straggler ranks, and overlaps precision transformation with communication.

MACS is also not just placement. Its efficiency analysis focuses on MoE-layer speedup and decomposes MoE latency. It reduces the expert-computation stage by capacity-constraining overloaded experts and rerouting overflow tokens, with local rerouting designed to avoid extra communication overhead.

Sources:

- ReaLB arXiv: https://arxiv.org/html/2604.19503v3
- MACS arXiv: https://arxiv.org/html/2605.05225

## Key Difference

Our current Method 1 placement only changes expert-to-rank ownership.

It does not:

- reduce per-token expert FLOPs,
- reduce top-k,
- lower precision,
- cap expert capacity,
- reroute tokens to idle local experts,
- overlap new scheduling work with dispatch,
- optimize all-to-all or combine kernels.

So the expected effect is narrower:

```text
placement -> better rank load balance -> shorter MoE straggler critical path
```

ReaLB/MACS do something stronger:

```text
ReaLB -> straggler ranks execute cheaper kernels
MACS -> overloaded experts process fewer/redirected tokens
```

This distinction matters. Placement can reduce the waiting caused by rank imbalance, but if the full prefill path is dominated by non-MoE work, placement alone will not necessarily move E2E.

## Why E2E Did Not Improve

### 1. MoE critical path is only about 10.8% of measured wall time

As-Is clean wall time is 42.40 s. The measured MoE CUDA critical path is 4.59 s.

```text
MoE fraction ~= 4.59 / 42.40 = 10.8%
```

The MoE critical path improved by 9.35%. Even if that number transferred perfectly:

```text
ideal E2E gain ~= 10.8% * 9.35% = 1.0%
```

So the maximum expected E2E improvement under this measurement is around 1%, before noise and overhead. Our clean wall result is +1.00% slower, which is within the scale where scheduling, multimodal preprocessing, chunking, and run variance can easily dominate.

This is the biggest explanation.

### 2. TTFT includes many non-MoE components

TTFT and scheduled-to-first-token include:

- image preprocessing / processor work,
- vision encoder,
- multimodal embedding merge,
- attention,
- router,
- all-to-all dispatch/combine,
- scheduler and chunked-prefill behavior,
- one-token generation plumbing.

Placement only affects MoE expert ownership. It does not touch most of the above.

### 3. The clean slowdown is concentrated in a few batches

Batch-level wall time:

| Batch | Dataset | As-Is | To-Be | Change |
|---:|---|---:|---:|---:|
| 0 | ChartQA | 3.8895 s | 3.6342 s | -6.56% |
| 1 | ChartQA | 3.4124 s | 3.4284 s | +0.47% |
| 2 | MMMU | 13.1221 s | 13.1958 s | +0.56% |
| 3 | MMMU | 10.6075 s | 10.5449 s | -0.59% |
| 4 | MMStar | 7.0853 s | 7.4907 s | +5.72% |
| 5 | MMStar | 4.2787 s | 4.5261 s | +5.78% |

The overall clean wall regression is mostly from MMStar batches 4 and 5. This suggests run-to-run/runtime effects rather than uniform compute regression.

### 4. Layer-wise placement may change communication and locality

The To-Be map balances token counts, but it also changes which ranks own which experts for every layer. This may alter:

- all-to-all destination distribution,
- token packing order,
- local expert indexing,
- memory access pattern,
- vLLM fused-MoE kernel/cache behavior,
- combine-stage behavior.

Linear placement may accidentally be friendlier to the current vLLM implementation. Our objective optimizes token-load balance, not communication/locality.

### 5. vLLM chunked prefill complicates per-request latency

At batch 32, vLLM uses chunked prefill with `max_num_batched_tokens=131072`. This means request-level TTFT depends on scheduling and chunk order. A layer-level MoE improvement can be hidden if the request waits behind vision-heavy chunks or other non-MoE stages.

### 6. MoE CUDA timing still needs tighter bracketing

The current CUDA-event patch wraps `FusedMoE.forward` during the whole vLLM lifecycle. It records actual worker MoE time, but it also sees engine warmup/profile activity. We drop the first call per layer/rank, but the cleanest paper-grade version should explicitly enable timing only around the measured profile loop.

This does not invalidate the direction of the result, but the exact MoE-only number should be treated as a diagnostic until the timing window is tightened.

## My Opinion

I do not think the current result proves the idea is wrong. It proves something more specific:

```text
Layer-wise placement is effective at reducing routed-load stragglers.
Layer-wise placement also reduces measured MoE-only critical path.
But static placement alone is not currently enough to reduce full vLLM TTFT/E2E.
```

This is a meaningful but incomplete result.

The concern is valid because the research motivation is E2E prefill speedup, not only nicer load plots. If the final method only reports MoE-only CUDA speedup, reviewers will likely ask why the user-visible latency does not improve. ReaLB is stronger here because it reports both MoE-layer and vLLM throughput speedup. MACS is partly less directly comparable because much of its speedup is framed around MoE-layer inference speed and capacity/rerouting, but it still makes an efficiency claim that is closer to actual latency than load balance alone.

My current judgment:

- We can report MoE-only timing as a legitimate systems diagnostic.
- We should not claim E2E prefill speedup yet.
- The paper story should not stop at Method 1 placement unless we can make E2E move.
- Method 1 currently looks like a load-balancing substrate, not a complete latency solution.
- Method 2 merge/cap or a communication-aware placement objective may be needed for visible E2E gains.

## Recommended Next Experiments

1. Tighten MoE timing instrumentation

Add an explicit timing gate so CUDA events are recorded only during the measured profile batches, excluding vLLM engine warmup/KV profiling.

2. Repeat clean wall-clock runs

Run As-Is/To-Be clean wall timing 3 to 5 times with identical samples and report mean/std/CI. Current 6-batch result is too small to conclude a reliable +1% slowdown.

3. Add full prefill breakdown

Measure at least:

- vision preprocessing/processor time,
- vision encoder time,
- language model prefill time,
- attention time if possible,
- MoE time,
- all-to-all/dispatch/combine time,
- scheduler/chunking overhead.

The key missing number is: what fraction of prefill is actually MoE expert compute?

4. Make placement objective communication-aware

Current objective minimizes token-load imbalance only. Add penalties for:

- rank destination entropy,
- per-layer map churn,
- all-to-all skew,
- local expert index scatter,
- communication critical path.

5. Compare against common layer-wise map

Layer-wise maps reduce load well, but may harm locality. Test:

- linear,
- common global tail-optimized map,
- layer-wise tail-optimized map,
- layer-wise with locality regularization.

6. Move to Method 2 if E2E remains flat

If placement keeps improving MoE-only time but not E2E, then Method 2 merge/cap becomes important because it can reduce the actual number of expert computations on straggler paths, not just rearrange ownership.

## Bottom Line

The current result is not bad, but it is not yet the final claim we want.

The strong claim we can currently defend:

```text
Layer-wise placement reduces batch-layer routed-load imbalance by ~31%
and reduces MoE-only CUDA critical-path time by ~9%.
```

The claim we cannot yet defend:

```text
Layer-wise placement reduces end-to-end prefill/TTFT latency.
```

The next milestone should be to either make E2E move, or clearly show that E2E is dominated by non-MoE components and reposition Method 1 as a necessary but insufficient component before Method 2.
