# EP4 Serving Straggler Regime Forensics

## Executive result

`FINAL STATUS: NO-GO`
`STRAGGLER_FOUND: NO`
`RUNTIME_CONTEXT: NO-GO`

The tested real vLLM V1 scheduler did not produce the preregistered strong
straggler regime.  The largest pure-prefill median rank imbalance was
1.290, and the largest pure-prefill median expert-time
imbalance was 1.059; the gate requires at least 1.5 and 1.10
respectively, across repeated layers/serving conditions.  No method or
scheduler change was made.

The most demanding clean condition was `long_multi_image`, submitted
concurrency 4, `max_num_batched_tokens=16384`, with a real scheduler iteration
of 13,671 prefill tokens and 3 co-batched requests.  Its median rank ratio was
1.157 and median expert-time ratio 1.078 (p95 1.149): latency grew with work,
but not as a robust rank straggler.

## Environment and serving proof

- Requested base commit: `f58334d4c51af915f73f2ca6d1606a64568852b7`.
  The checked-out branch point was its descendant `d25438863d44688a0abdd79689dc358e8118e2e2`.
- Model: Qwen3-VL-30B-A3B-Instruct, BF16, snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Runtime: vLLM 0.20.0 V1 engine, eager mode, chunked prefill, prefix cache off.
- Topology: TP2 / DP2 / EP4 / PP1, DBO off, linear expert placement
  (`expert_id // 32`), 32 local experts per EP rank.
- Backend proof: `DeepEPHTPrepareAndFinalize`, `DeepEPHTAll2AllManager`, and
  `TritonExperts`; all four EP ranks reported `ep_world_size=4` and
  `visible_devices=1,2,3,4`.
- GPU mapping: `CUDA_VISIBLE_DEVICES=1,2,3,4` (logical ranks 0–3 map to
  physical GPUs 1–4). No GPUs 0 or 5–7 were used by these runs.
- `max_num_batched_tokens`: 16,384, 8,192, 4,096, and 2,048 in the fixed
  token-budget sweep. The 16,384 run is the primary Stage A/B baseline.
- Workload: prior local real-image/text IDs plus two local long multi-image
  rows; no downloaded dataset or synthetic routes. Submissions were batched
  through the real V1 scheduler via `_add_completion_requests`.

## What the scheduler actually scheduled

The configured budget was not treated as the workload size. The scheduler
trace records positive iterations and their actual token/request counts in
`scheduler_positive_trace.csv`. At budget 16,384, the largest vision-heavy
co-batch contained 15 requests and 4,528 tokens; the long multi-image c4 wave
contained 3 requests and 13,671 tokens. Lower budgets fragmented waves and
reduced co-batching rather than increasing rank imbalance.

| budget run | largest vision scheduled tokens | largest scheduled requests | median rank ratio | median expert ratio | p95 expert ratio |
|---|---:|---:|---:|---:|---:|
| 1340 | 4528 | 15 | 1.186 | 1.041 | 1.385 |
| 8192 | 4528 | 15 | 1.186 | 1.043 | 1.160 |
| 4096 | 4096 | 15 | 1.179 | 1.033 | 1.176 |
| 2048 | 2048 | 7 | 1.182 | 1.033 | 1.127 |

## Stage A — Low-load baseline

At submitted concurrency 1, text-only had rank ratio 1.279 and expert-time
ratio 1.029; vision-heavy had 1.235 and 1.030; long multi-image had 1.124 and
1.038. These are ordinary small rank spreads, not a strong critical-path
straggler. The six required summary figures are in
`poc_flashvep/deepep_revalidation/results/ep4_serving_straggler_regime_20260901_final/figures/`.

## Stage B — Concurrent vision-prefill sweep

The fixed 16,384-budget sweep used submitted concurrency 1, 2, 4, 8, and 16.
The scheduler co-batched fewer requests than submitted when prompt/DP timing
prevented a single wave; this is why actual scheduled counts, rather than the
submitted number, are the primary covariate.

| condition | submitted c | median scheduled tokens | max scheduled tokens | median rank max/mean | median expert-time max/mean | p95 expert ratio | expert >=15% fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| long_multi_image | 1 | 3203 | 3203 | 1.124 | 1.038 | 1.086 | 2.1% |
| long_multi_image | 2 | 4218 | 5234 | 1.165 | 1.048 | 1.176 | 8.3% |
| long_multi_image | 4 | 8437 | 13671 | 1.147 | 1.055 | 1.149 | 5.2% |
| text_control | 1 | 128 | 128 | 1.279 | 1.032 | 1.157 | 6.2% |
| text_control | 2 | 188 | 248 | 1.283 | 1.037 | 1.214 | 15.6% |
| text_control | 4 | 582 | 1036 | 1.273 | 1.028 | 1.158 | 5.7% |
| text_control | 8 | 1164 | 2200 | 1.272 | 1.035 | 1.158 | 5.7% |
| text_control | 16 | 2328 | 4528 | 1.272 | 1.052 | 1.154 | 5.7% |
| text_only | 1 | 128 | 128 | 1.279 | 1.029 | 1.140 | 5.2% |
| vision_heavy | 1 | 128 | 128 | 1.235 | 1.030 | 1.165 | 6.2% |
| vision_heavy | 2 | 188 | 248 | 1.203 | 1.027 | 1.191 | 6.2% |
| vision_heavy | 4 | 582 | 1036 | 1.181 | 1.037 | 1.324 | 27.6% |
| vision_heavy | 8 | 1164 | 2200 | 1.184 | 1.029 | 1.250 | 9.9% |
| vision_heavy | 16 | 2328 | 4528 | 1.186 | 1.041 | 1.385 | 15.6% |
| vision_single | 1 | 128 | 128 | 1.235 | 1.030 | 1.179 | 7.3% |

At the largest matched vision/text co-batch (15 requests, 4,528 scheduled
tokens), the paired layer-level differences were:

- Vision minus Text rank ratio: median
  -0.086,
  bootstrap 95% CI
  [-0.14970201549852832, -0.08590395626379689].
- Vision minus Text expert-time ratio: median
  -0.017,
  bootstrap 95% CI
  [-0.032116110143535874, -0.013102360793302114].

Vision was not more imbalanced; its rank and expert ratios were slightly lower
than the matched text control. Long vision requests did show the largest
absolute expert/dispatch times, but the normalized rank spread remained below
the gate.

## Stage C — Token-budget sweep

Budgets 8,192, 4,096, and 2,048 were run only after no strong Stage B
condition was found. Decreasing the global budget changed effective chunking
and co-batching, but did not create a robust high-ratio condition. Across all
pure prefill groups, the maximum median rank ratio was 1.290
and maximum median expert ratio was 1.059.

## Stage D — Mixed prefill + decode

After Stage C remained below gate, two bounded real mixed runs were performed:
one 64-token text decode request plus 3, and then 7, image-prefill requests.
The scheduler trace shows a text prefill, decode iterations, image-prefill
iterations, and continuing decode iterations. Decode rows have one scheduled
token and therefore can show an uninformative rank ratio around 3.3 from tiny
assignment counts; they are not a large compute straggler. Prefill-phase
ratios are the relevant comparison.

| run | scheduler phase | observations | median rank ratio | median expert ratio | p95 expert ratio |
|---|---|---:|---:|---:|---:|
| c4b | decode | 5952 | 3.273 | 1.060 | 1.562 |
| c4b | prefill | 192 | 1.222 | 1.047 | 1.756 |
| c8b | decode | 5856 | 3.250 | 1.065 | 1.577 |
| c8b | prefill | 288 | 1.210 | 1.052 | 1.228 |

The high-scale 9,535-token c8 mixed-prefill sub-iteration reached an expert
ratio of about 1.095 (the c8 prefill aggregate median is 1.052) and had noisy
tails, but it still lacked the required median rank imbalance or repetition
robustness. No DeepEP collectives were launched concurrently by the
instrumentation.

### Context comparison (scope caveat)

The same-GPU isolated replay used the fixed `text_18_tui_main` expert shape
(N=2984, G=30) and 20 warmup + 100 measured expert-only iterations. Live
single-request and serving-like entries are not iso-N with that microbenchmark,
so the table is a context/scope diagnostic, not a causal latency ratio.

| context | measured scope | median expert ms | p95 | CV | >=15% tail |
|---|---|---:|---:|---:|---:|
| isolated (GPU 1–4) | one local expert, N=2984 | 0.2340 | 0.2624 | 8.98% | 4.00% |
| controlled vLLM | vision-heavy c1, live max-rank | 0.4146 | 0.5313 | 8.96% | 9.38% |
| serving-like vLLM | vision-heavy c16, 15 co-batched req | 0.6855 | 0.7938 | 12.44% | 6.25% |

## Stage E — Long multimodal stress

The bounded long multi-image family was included at c1/c2/c4 for every token
budget. The 13,671-token c4 iteration was the strongest clean load condition;
its median expert ratio 1.078 and rank ratio 1.157 remain below the strong
gate. Additional c8/c16 long stress was not run after this condition failed,
consistent with the early-stop/bounded-scope rule.

## Instrumentation and correctness

Each MoE invocation was timed with CUDA events around dispatch, TritonExperts,
and combine; event times were resolved after a bounded flush synchronization,
not by synchronizing every layer. The rank proof files report 69,120 captured
events per rank for each fixed-budget run and 13,824 per rank for each mixed
run. This confirms all four EP ranks and the intended backend path. There was
no CUDA or DeepEP runtime failure, no route/placement/model modification, and
all driver records returned the expected output count. A transient control-file
JSON read race was visible in the first mixed logs while the host atomically
replaced the control file; the experiment-local reader now catches that race,
and it did not change scheduling or returned outputs.

There is no clean instrumentation-OFF wall-time pair in this bounded run, so a
numeric instrumentation-overhead percentage is not claimed. The hook adds
nonblocking event records and one end-of-run synchronization; event overhead
should be measured separately before any production use.

## Bottleneck localization

The dominant observed effect is ordinary latency scaling with scheduled token
volume (especially long multi-image dispatch/expert/combine time), not an
imbalanced critical rank. The hottest rank changes with layer/condition rather
than forming one persistent device hotspot. Lower token budgets mostly reduce
co-batching and produce heterogeneous chunks; they do not amplify the median
rank or expert-time ratio.

Thus this data does not establish an MLLM-specific serving straggler, nor does
it establish a generic vLLM/DeepEP runtime-tail candidate. The previous
modality-specific claim is not rescued by high concurrency in this tested EP4
range.

## Gate and next action

`FINAL STATUS: NO-GO` and `STRAGGLER_FOUND: NO`.

No method design or further blind stress sweep is justified by this evidence.
The next single action is to stop the EP4 straggler direction and prioritize a
different mechanism with a reproducible paired effect; only if an external
requirement demands runtime-tail evidence should the next experiment be a
separate, longer real arrival-process trace with instrumentation-OFF/ON pairs.

## Artifacts

- Compact result: `poc_flashvep/deepep_revalidation/results/ep4_serving_straggler_regime_20260901_final/`
- Raw traces (preserved locally): the six run directories listed in
  `raw_manifest.json` under the compact result.
- Figures: `plot1_concurrency_rank_imbalance.png`,
  `plot2_concurrency_expert_imbalance.png`,
  `plot3_scheduled_tokens_vs_straggler.png`,
  `plot4_vision_vs_text_matched_serving.png`,
  `plot5_layer_hot_rank_heatmap.png`,
  `plot6_scheduler_iteration_timeline.png`, and
  `plot7_mixed_prefill_decode_context.png`.
- Machine-readable gate: `gate_summary.json`.
