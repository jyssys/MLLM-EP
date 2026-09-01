# DP/EP arrival-skew under two EP4 topologies

## Executive result

`FINAL STATUS: NO-GO`

`BOTTLENECK TYPE: NONE`

`STRAGGLER_FOUND: NO`

`METHOD DESIGN: NO`

Real vLLM serving produced a much larger *duration-based* DP arrival spread in
TP1/DP4/EP4, especially for long multimodal requests, but that spread did not
turn into a repeatable EP critical-path wait. Across 22,368 measured
request/layer observations, the median completion-spread wait proxy was 0.54%
for TP2/DP2 and 1.14% for TP1/DP4. Rank assignment imbalance was about 1.18x
in both cases and median expert-time imbalance was 1.05x/1.07x. Thus the
tested four-GPU serving range does not contain the requested robust
straggler regime.

## Question and scope

The test asks whether independent DP groups arrive at a common EP4 MoE
collective at different times, and whether increasing DP groups from two to
four amplifies a *real* synchronization stall. No scheduler, routing,
placement, chunking, replication, or kernel optimization was implemented.

The branch was created from base `c14a632` (the current
`ep4-serving-straggler-regime` tip). The live commands all used the same
entrypoint and only changed topology and token budget:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 WARMUPS=1 ITERATIONS=2 \
  poc_flashvep/dp_ep_arrival_skew_two_topologies/run_gpu.sh OUT A --scope primary
CUDA_VISIBLE_DEVICES=1,2,3,4 WARMUPS=1 ITERATIONS=2 \
  poc_flashvep/dp_ep_arrival_skew_two_topologies/run_gpu.sh OUT B --scope primary
CUDA_VISIBLE_DEVICES=1,2,3,4 WARMUPS=1 ITERATIONS=1 MAX_NUM_BATCHED_TOKENS=8192 \
  poc_flashvep/dp_ep_arrival_skew_two_topologies/run_gpu.sh OUT {A,B} --scope stress
CUDA_VISIBLE_DEVICES=1,2,3,4 WARMUPS=1 ITERATIONS=1 MAX_NUM_BATCHED_TOKENS={4096,2048} \
  poc_flashvep/dp_ep_arrival_skew_two_topologies/run_gpu.sh OUT {A,B} --scope stress
```

The analyzer received all eight source run directories and applies no
latency-based sample selection. The post-run analyzer additions only infer
stable labels from preregistered batch IDs and create aggregate figures; they
cannot change the raw measurements.

The primary serving runs were:

| run | topology | budget | scope | measured |
|---|---|---:|---|---:|
| `20260901_1530_A` | TP2/DP2/EP4 | 16,384 | vision/text c2/4/8/16, long c2/4/8; balanced + heterogeneous | 2 iterations |
| `20260901_1600_B` | TP1/DP4/EP4 | 16,384 | same | 2 iterations |

Because no strong straggler was found, the preregistered bounded stress scope
was also run at budgets 8,192, 4,096 and 2,048 (one measured iteration per
wave) for both topologies. These are diagnostics, not a post-hoc selection of
favorable conditions.

## Environment and topology proof

- Model: local Qwen3-VL-30B-A3B-Instruct snapshot, `Qwen3VLMoeForConditionalGeneration`.
- BF16, PP1, DeepEP high-throughput, TritonExperts, linear placement (32 of 128 experts/rank).
- vLLM `0.20.0`, V1 scheduler, eager mode, DBO off, prefix cache off.
- Runtime audit: PyTorch `2.11.0+cu129`, CUDA runtime `12.9`, Triton `3.6.0`, H100 driver `570.211.01`.
- `CUDA_VISIBLE_DEVICES=1,2,3,4`; physical GPUs 1, 2, 3, and 4 only. GPUs 5–7 were not part of these runs.
- A: TP2/DP2/EP4, world size 4. TP groups are `[0,1]` and `[2,3]`; vLLM's transposed DP group lists are `[0,2]` and `[1,3]`, corresponding to environment DP groups on physical pairs (1,2) and (3,4). EP group is `[0,1,2,3]`.
- B: TP1/DP4/EP4, each physical GPU is a TP world of one and a DP rank 0–3; DP and EP groups are `[0,1,2,3]`.
- Case B was feasible without OOM. A's `use_sequence_parallel_moe` is true under local vLLM's DeepEP condition (EP, TP>1, DP>1); B's is false because TP=1. This is an explicit topology confound, not hidden.

Runtime proof files are in the compact result directory under `topology_proof/`.
Each contains physical GPU, PID, global/local rank, TP/DP/EP/PP ranks and full
group rank lists. Backend proof records `DeepEPHTPrepareAndFinalize`,
`DeepEPHTAll2AllManager`, `TritonExperts`, and EP world size 4.

## Source/semantic audit

The local source and the current upstream references were inspected before
running:

- `vllm/config/parallel.py` and `distributed/parallel_state.py`: world-size and TP/DP/EP group construction.
- `model_executor/models/qwen3_moe.py`: decoder order is input RMSNorm → self-attention → post-attention RMSNorm → MoE.
- `model_executor/models/qwen3_vl_moe.py`: VL language decoder uses the Qwen3 MoE decoder path.
- `layers/fused_moe/prepare_finalize/deepep_ht.py`: `get_dispatch_layout`, `dispatch(previous_event=...)`, and `combine(previous_event=...)` with `EventOverlap` semantics.
- DeepEP dispatch/combine and `async_finish` semantics.

The detailed audit and URLs are in
[`source_audit.md`](../dp_ep_arrival_skew_two_topologies/source_audit.md).

## Instrumentation

The read-only hook records, per scheduler iteration and MoE layer:

- scheduler control (batch/request IDs, phase, concurrency, mode, token counts),
- layer-entry → MoE-entry CUDA event (`pre_moe_cuda_ms`),
- MoE entry → EP completion event (`ep_entry_to_done_ms`),
- layer-entry → EP completion,
- per-EP-rank assignments, expert histogram, expert/dispatch/combine CUDA-event durations, active/effective experts.

The analyzer uses DP-local duration events; it never subtracts CUDA event
timestamps from different GPUs as if they were a global clock. `ep_wait_proxy`
is the conservative spread of the DP-group `layer_entry_to_ep_done_ms`
durations, not a claim of direct NCCL wait time. Autotuner probe rows with
fewer than 64 assignments are excluded by a fixed filter. Artificial-delay
instrumentation validation was not run (`instrumentation_validation.json` is
`NOT_RUN`), so natural results are not contaminated by an injected delay.

## Serving workload and actual scheduler scale

The exact existing short image/text catalog and two long multi-image requests
were reused. Balanced and deliberately heterogeneous DP request distributions
were fixed in `run_topology.py` before timing. The measured positive scheduler
rows had the following token ranges (median is over request/layer records):

| topology | budgets | positive scheduled-token median range | p90 | observed max |
|---|---|---:|---:|---:|
| A | 16,384/8,192/4,096/2,048 | 2,048–2,651.5 | 5,234 | 15,702 |
| B | 16,384/8,192/4,096/2,048 | 1,252.5–1,369.75 | 4,218.5 | 7,089 |

The scheduler was configured with chunked prefill at each requested budget;
these are actual positive scheduler observations, not configuration values.

## Main topology results

Medians below aggregate all request families and layers at the indicated
budget. `arrival` is DP pre-MoE duration spread divided by the slowest DP
duration. `wait` is the completion-spread proxy divided by the slowest
DP-group completion duration.

| budget | topology | arrival skew (ms) | arrival skew (%) | EP wait proxy (ms) | sync-stall (%) | max/mean rank load | max/mean expert time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16,384 | A TP2/DP2 | 0.0640 | 7.36 | 0.0152 | 0.62 | 1.185 | 1.054 |
| 16,384 | B TP1/DP4 | 0.1619 | 26.57 | 0.0279 | 1.12 | 1.175 | 1.065 |
| 8,192 | A TP2/DP2 | 0.0568 | 6.47 | 0.0140 | 0.56 | 1.178 | 1.059 |
| 8,192 | B TP1/DP4 | 0.2498 | 35.56 | 0.0286 | 1.13 | 1.176 | 1.067 |
| 4,096 | A TP2/DP2 | 0.0410 | 4.96 | 0.0089 | 0.42 | 1.170 | 1.046 |
| 4,096 | B TP1/DP4 | 0.8777 | 67.98 | 0.0425 | 1.39 | 1.168 | 1.154 |
| 2,048 | A TP2/DP2 | 0.0331 | 3.46 | 0.0107 | 0.52 | 1.181 | 1.038 |
| 2,048 | B TP1/DP4 | 0.1101 | 14.25 | 0.0200 | 1.04 | 1.179 | 1.044 |

DP4 clearly amplifies arrival-duration skew (overall median 25.35% versus
5.29% for A; long multimodal median 44.28% versus 7.69%). It does not amplify
the EP completion wait to a meaningful level: overall wait is 1.14% versus
0.54%, and the 4,096 diagnostic's 1.39% is still far below the 5–10% HOLD
range.

There are isolated outliers (maximum sync-stall fraction 55.9% in A and
51.6% in B), but they are not robust: only 1.98% of A and 1.40% of B
observations exceed 10% stall, and no topology has a median assignment ratio
above 1.5. The p95 stall fractions are 7.06% (A) and 5.60% (B) over all
budgets.

## Workload and modality view

| topology | family | median arrival skew | median sync stall | p95 sync stall | max/mean rank load | max/mean expert time |
|---|---|---:|---:|---:|---:|---:|
| A | vision (short) | 4.74% | 0.65% | 7.76% | 1.174 | 1.041 |
| A | text control | 4.25% | 0.50% | 6.29% | 1.265 | 1.052 |
| A | long multimodal | 7.69% | 0.50% | 7.09% | 1.141 | 1.049 |
| B | vision (short) | 19.68% | 1.45% | 6.21% | 1.173 | 1.062 |
| B | text control | 17.54% | 1.11% | 5.63% | 1.251 | 1.075 |
| B | long multimodal | 44.28% | 0.92% | 4.91% | 1.136 | 1.078 |

The MLLM-versus-text stall difference is small (A +0.15 percentage points,
B +0.34 points for short families), nowhere near the requested +5 points.
Text often has equal or greater assignment imbalance, while long multimodal
requests have the largest arrival-duration spread but not the largest wait.

Duration-spread to completion-spread correlations are weak: `r=0.250` for A
and `r=0.036` for B. Rank-load ratio to wait is effectively zero (`r=-0.001`
and `r=-0.029`). This is direct evidence against interpreting pre-MoE skew
alone as a synchronization stall.

## Gate decision

`STRAGGLER_FOUND: NO`.

- Strong condition A was not met: median max/mean rank load was about 1.17–1.18, not ≥1.5.
- Strong condition B was not met: median max/mean expert time was 1.04–1.08, and median EP wait was 0.42–1.39%.
- Condition C was not met: high arrival spread did not produce a repeatable critical-path penalty across layers and repetitions.
- `MLLM_SPECIFIC_GO`: no. MLLM is not ≥5 percentage points above matched text.
- `GENERIC_DP_EP_GO`: no. The generic wait signal is below 5% in the tested range.
- Final classification: `NO-GO`, `BOTTLENECK TYPE: NONE`.

The expected causal chain is therefore cut after arrival skew:

```text
DP-local work variation → arrival-duration skew  ✓
arrival skew → robust EP collective wait        ✗ (not observed)
EP wait → serving critical-path slowdown         ✗
```

## Artifacts and reproducibility

The compact committed result contains the aggregate CSV, seven figures,
topology/backend proofs, and the trace inventory. Full raw rank and scheduler
JSONL streams (about 2.7 GiB across all eight runs) remain at the local paths
listed in [`trace_inventory.md`](../deepep_revalidation/results/dp_ep_arrival_skew_two_topologies_20260901_final/trace_inventory.md)
and were not duplicated into git. The exact commands, environment variables,
and run metadata are retained in each source run directory.

Figures:

- `plot1_dp_group_pre_moe_timeline.png`
- `plot2_dp_group_pre_moe_timeline.png`
- `plot3_dp_imbalance_vs_arrival_skew.png`
- `plot4_arrival_skew_vs_ep_wait.png`
- `plot5_dp2_vs_dp4_stall.png`
- `plot6_mllm_vs_text.png`
- `plot7_per_layer_stall_heatmap.png`

## Relation to prior work

The proposed barrier (independent DP attention progress followed by a shared
EP MoE collective) is the same broad systems issue studied by ASAP, and the
remaining-workload/scheduling framing is close to Gimbal. This experiment does
not provide an MLLM-specific amplification: DP4 increases measured arrival
spread, but the shared EP completion proxy stays small and is comparable for
vision and text. Accordingly, no MLLM-specific or generic DP+EP optimization
is justified by this trace.

## Next single action

Close this mechanism for the current 4-GPU Qwen3-VL serving setup and do not
start method design. If a future study must revisit it, the only useful next
measurement is a narrowly scoped direct DeepEP/NCCL wait trace on a workload
that is independently shown to have median rank-load ratio ≥1.5; the present
data do not justify broader stress sweeps.
