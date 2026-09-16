# LLaDA2.0-mini virtual EP simulator: true-EP2 calibration

## Verdict

`TRUE_EP2: PASS`

`EP2_SIMULATOR_VALIDATION: CALIBRATED`

The calibrated scope is the routed-MoE stage only. The measured held-out
error is 3.87% median APE and 9.40% P90 APE. No production serving/BCT claim
is made.

## Cost-accounting contract

The following categories remain separate in code, raw artifacts, validation,
and reports:

1. EP dispatch
2. routed expert compute
3. EP combine
4. replicated-state bridge all-gather

Only 1--3 form the routed-MoE EP critical path. Category 4 reconstructs the
replicated TP1 hidden state in the minimal harness. It is never counted as EP
communication, EP-stage latency, or EP speedup.

The full harness wall time is named **EP-isolation harness latency**. It is not
production EP2 serving latency.

## Environment and model truth

| Item | Value |
|---|---|
| Model | `inclusionAI/LLaDA2.0-mini` |
| Revision | `dad945cac317da394b390f82c7b40691d8a881ed` |
| Dtype | BF16 |
| Layers | 20 total; routed MoE layers 1--19 |
| Hidden / MoE intermediate | 2048 / 512 |
| Routed experts / top-k / shared | 256 / 8 / 1 |
| Physical GPUs used | GPU 0 and GPU 1 only |
| Software | torch 2.8.0+cu128, transformers 4.57.0, SGLang 0.5.3.post1 |
| EP transport | DeepEP normal, BF16 dispatch |

Read-only topology capture reconfirmed all-pairs `NV18`, 18 active NVLinks per
GPU, and Fabric `Completed / Success` in clique 0. The simulator does not
interpret 18 displayed links as independent full-bandwidth links per peer.
It uses measured endpoint behavior.

## TP1/EP2 feasibility audit

The stock dInfer/SGLang full-model integration does not cleanly expose an
independent TP1/EP2 configuration:

- the benchmark initializes model parallel with the visible worker count;
- the LLaDA2 DeepEP block obtains EP size and process group from the TP group;
- local expert count/ownership is derived from that TP size;
- the existing row-partition bridge uses the same group.

This is an integration coupling, not a DeepEP restriction. DeepEP accepts an
arbitrary NCCL process group, so a bounded minimal TP1/EP2 implementation is
feasible without converting attention/dense execution to TP2.

The selected **EP-isolation harness** keeps official HF attention, dense,
router, and shared-expert paths replicated with TP1 semantics. Every routed
MoE block:

1. splits physical rows into two deterministic contiguous source shards;
2. dispatches real top-k routes over a true two-rank DeepEP process group;
3. executes only the 128 owner-local experts per rank;
4. performs DeepEP reverse combine;
5. reconstructs replicated hidden state through separately timed all-gather.

No TP2+EP2 fallback is used.

## True-EP2 proof

- Real NCCL world size is 2 on physical GPUs 0 and 1.
- Rank 0 owns experts 0--127; rank 1 owns 128--255.
- Cross-rank assignments and deduplicated remote activation rows occur in both
  directions.
- Only owner-local expert modules remain resident/executed on each rank.
- Reverse combine returns each source shard's routed contribution.
- Both ranks produced token-identical output for 8/8 sanity requests and
  32/32 GSM8K gate requests.

The explicit timing smoke contains 76 routed-MoE invocations per rank. Its
median critical-rank categories were:

| Category | Median (ms) | Interpretation |
|---|---:|---|
| EP dispatch | 0.206 | harness observation |
| routed expert compute | 12.731 | slow Python HF expert loop; not compute calibration |
| EP combine | 1.698 | reference-boundary observation; not communication calibration |
| replicated-state bridge all-gather | 0.104 | separate non-EP category |
| EP stage excluding bridge | 13.046 | categories 1--3 only |
| EP-isolation harness MoE wall | 13.147 | includes category 4; not production serving |

The headline timing model instead uses clean DeepEP communication and BF16
`torch._grouped_mm` replay.

## Single-GPU semantic bridge

The fixed 32-request GSM8K cohort used the same checkpoint, four-shot prompt,
tokenizer, threshold 0.95, block length 32, steps-per-block 32, and generation
cap 2048 as the reproduced reference.

| Metric | Single-GPU reference | True EP2 |
|---|---:|---:|
| Correct | 31/32 | 30/32 |
| Parsed-answer identity | — | 31/32 |
| Exact generation string | — | 2/32 |
| Rank-to-rank token identity | — | 32/32 |
| Mean NFE | 98.25 | 88.09 |

One reference-correct sample changed from answer 14 to 2. Per-layer random
hidden probes showed median BF16 relative-L2 drift of 0.278%, and the complete
forward preserved top-1 logits on the diagnostic input. The trajectory-level
difference is therefore bounded but not bit-exact: changing distributed BF16
reduction order can alter future diffusion decisions. Quality is not
catastrophically broken, so timing calibration proceeds, but exact semantic
parity is not claimed.

## Trace integrity

The heavy official-HF trace contains:

- 4 full GSM8K trajectories (request IDs 0, 1, 3, 5);
- 275 refinement forwards;
- all 19 routed-MoE layers;
- 8,135,040 physical token-layer rows;
- top-k expert IDs and router weights for every row;
- block/iteration/NFE, mask state, accepted count, and confidence summaries.

Hook-on/off output parity passed 4/4. The trace records
`physical_row_semantics=vanilla_full_rows`: decreasing MASK count never creates
fake FreshLane sparsity. For example, request 0's first block reduced masks
14→13→4→2→1→0 while every layer continued to execute 1,504 physical rows.

Assignment matrix `A[s,d]` and deduplicated activation matrix `U[s,d]` are
stored separately. Logical BF16 activation bytes are reported. DeepEP packed
protocol bytes and physical NVLink counter bytes were not directly observable
and are explicitly marked unavailable rather than equated with
`tokens × top_k × hidden_bytes`.

## EP2 communication calibration

DeepEP normal dispatch/combine was measured for 1--2,048 rows per source rank
under local, uniform, expert-hot, one-way rank0→rank1, one-way rank1→rank0, and
bidirectional-remote routes. Every point uses 12 warmups and 40 measured
repetitions.

The median endpoint alpha-beta fit is:

| Operation | Startup (ms) | Effective payload slope |
|---|---:|---:|
| Dispatch | 0.1133 | 85.51 GB/s |
| Combine | 0.0586 | 135.97 GB/s |

P10, median, and P90 fits define the named
`endpoint_optimistic`, `ep2_calibrated_base`, and `concurrency_stressed`
scenarios. These are sensitivity scenarios, not confidence intervals. EP2
cannot identify a separate multi-peer penalty, so that coefficient is zero
rather than invented. EP4/EP8 endpoint concurrency remains a true-hardware
validation risk.

## Rank-local compute replay

Actual virtual-rank expert histograms are replayed on one H100 with:

- the measured destination-local token/expert layout;
- receive-format branch packing and sorting;
- BF16 `torch._grouped_mm` gate/up;
- SwiGLU;
- BF16 grouped down projection;
- route-weighted destination-local reduction.

Separate real replays are measured for 128, 64, and 32 local experts (EP2,
EP4, and EP8). No `EP2/P` compute scaling is used. Training and held-out
samples are split by request ID, never random rows.

## Held-out validation

The validation substrate replays 96 real route states through actual true EP2
DeepEP plus the grouped BF16 expert path. Requests 0 and 5 are held out; 1 and
3 are calibration requests.

| Component | Median APE | P90 APE |
|---|---:|---:|
| EP dispatch | 4.37% | 5.67% |
| Routed expert compute | 2.06% | 4.12% |
| EP combine | 20.20% | 54.75% |
| Routed-MoE stage (1+2+3) | **3.87%** | **9.40%** |

Combine is a small absolute component (roughly 0.10 ms median in trace replay),
so its large percentage error has limited stage impact. The project gate of
median APE ≤10% and P90 APE ≤20% is passed.

## NFE-only versus EP-aware

Held-out routed-MoE stage prediction:

| Predictor | Median APE | P90 APE |
|---|---:|---:|
| A: NFE only | 3.86% | 7.67% |
| B: NFE + routed shape | 6.07% | 11.13% |
| C: NFE + shape + rank load + traffic/fanout | **2.93%** | **6.67%** |

EP-aware C improves median APE by 0.93 percentage points (24.1% relative) and
P90 by 1.00 point (13.1% relative) versus NFE-only. NFE already explains much
of this small vanilla cohort; the extra physical features are useful but not
dominant. This limitation is retained rather than hidden.

## Final EP2 status

`EP2-CALIBRATED`

This authorizes routed-MoE-stage EP4/EP8 screening. It does not authorize
production serving latency, BCT, queueing, or full-request E2E projections.
