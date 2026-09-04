# Critical-Rank Expert-Work Coalescing on Qwen3-VL EP8

## Executive decision

**FINAL STATUS: NO_GO**

The real EP8 trace shows a useful *routing-pressure* signal: Vision
assignments contribute a median 213 assignments above the mean on the
critical rank (mean 234), while Text contributes a median -4.  However, this
bounded multimodal workload does not establish a quality-preserving
critical-rank coalescing opportunity.  Measured routed-expert CUDA imbalance
is mild in the typical invocation (median max/mean 1.056, p90 1.113), sampled
hidden-vector matches cover only a small fraction of assignments, and no
EP8 expert outputs were captured to validate that hidden similarity means
output equivalence.  At the available-pair budget, critical-rank targeting
reduces the max-rank assignment proxy by only 0.47% median across the sampled
layers (1.00% in the strongest sampled layer), below the preregistered 3%
quality-preserving headroom floor.

This is a conservative trace-driven NO_GO, not a claim that coalescing is
mathematically impossible.  It means that implementing a real coalescing
runtime is not justified by this evidence.

## Configuration and exact run

| item | value |
|---|---|
| model | Qwen3-VL-30B-A3B-Instruct |
| checkpoint | `/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| architecture | `Qwen3VLMoeForConditionalGeneration` |
| text config | hidden 2048; 48 layers; 128 routed experts; top-8; MoE every layer |
| topology | TP2 / DP4 / EP8 / PP1 |
| experts per EP rank | 16; linear `expert_id // 16` |
| dtype/backend | BF16; DeepEP high-throughput; TritonExperts |
| controls | EPLB off; DBO off; prefix cache off; eager; max model length 8192 |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` (physical 0–7, one-to-one visible mapping) |
| measured schedule | 6 real-image requests × 4 repetitions; rep0 warmup, reps1–3 measured |
| measured route views | 18 waves × 48 layers = 864 canonical DP0 invocations; 6,912 route files (8 EP ranks/view) |
| timing | 9,216 raw per-rank rows; 8 EP-rank CUDA-event timing per measured invocation |

The exact command, model config hash, placement map, environment, and workload
manifest are stored in the trace result directory.  Run17 used the same
validated execution path as run16 but increased the measured repetitions from
one to three; run16 is retained as an independent reference.

## Workload and capture

The six real multimodal requests were:

| request | images | prompt / Vision tokens | category |
|---|---:|---:|---|
| `single_astronaut` | 1 | 217 / 196 | natural |
| `pair_natural` | 2 | 415 / 392 | natural |
| `pair_resolution` | 2 | 1199 / 1176 | fine-grained, 896-pixel input |
| `quad_diverse` | 4 | 741 / 714 | mixed |
| `quad_repeated` | 4 | 811 / 784 | repeated same image |
| `quad_fine` | 4 | 713 / 686 | fine-grained |

The child-worker route hook records, per TP shard and measured wave/layer,
global token position, token ID, Vision/Text label (`image_token_id=151655`),
the exact top-k logical expert IDs, router weights, and `expert_id // 16`
destination rank.  The DeepEP hook records per-rank dispatch, routed expert,
and combine CUDA-event durations plus the local expert histogram.  Hidden
vectors are sampled for up to 128 Vision rows per TP shard at layers 16, 24,
and 40.  No route, model tensor, placement, or scheduler decision is changed.

For this bounded measurement, the same real request is submitted to each of
the four DP engines so that the multimodal EP collective participates.  Route
statistics use one canonical DP0 copy; raw timing contains the four replicated
copies.  This is measurement-mode replication, not a serving policy, and is
explicitly recorded in `environment.json` and `trace_portability.md`.

## EP8 routing pressure and measured CUDA imbalance

### Overall

| metric | median | p90 | max |
|---|---:|---:|---:|
| rank assignment max/mean | 1.326 | 1.545 | 1.998 |
| routed-expert CUDA max/mean | 1.056 | 1.113 | 1.902 |
| expert CUDA max (ms) | 0.538 | 0.646 | 1.461 |
| dispatch max (ms) | 0.571 | 1.236 | 19.533 |
| combine max (ms) | 0.163 | 0.256 | 1.725 |

Only 38/864 (4.4%) invocations have expert CUDA max/mean ≥1.15, 7/864
(0.8%) ≥1.25, and 2/864 (0.2%) ≥1.50.  Thus route rank pressure is common,
but the typical image invocation is not an actual routed-expert straggler.
The rank assignment ratio and expert CUDA ratio have only a modest Pearson
association in this trace (r=0.255).  This reinforces that assignment counts
alone cannot justify coalescing.

### Request/layer robustness

The following values are median / p90 / max over 144 layer views per request;
`V excess` is the Vision contribution to the total critical rank above the
eight-rank mean.

| request | rank max/mean | expert CUDA max/mean | V excess (median) |
|---|---:|---:|---:|
| `single_astronaut` | 1.346 / 1.569 / 1.977 | 1.039 / 1.069 / 1.160 | 79 |
| `pair_natural` | 1.315 / 1.532 / 1.851 | 1.045 / 1.072 / 1.204 | 133 |
| `pair_resolution` | 1.285 / 1.461 / 1.626 | 1.088 / 1.156 / 1.669 | 329 |
| `quad_diverse` | 1.289 / 1.527 / 1.691 | 1.053 / 1.090 / 1.291 | 218 |
| `quad_repeated` | 1.353 / 1.549 / 1.998 | 1.063 / 1.135 / 1.902 | 284 |
| `quad_fine` | 1.340 / 1.635 / 1.826 | 1.061 / 1.116 / 1.327 | 248 |

Across all 864 views, Vision critical excess is median 213 and mean 234;
Text critical excess is median -4 and mean -1.1.  This is the strongest
positive evidence for Vision-induced critical-rank pressure.  It is not yet
evidence that removing Vision work is safe or that it reduces the GPU
critical path.

For context, the earlier Qwen3-30B-A3B **text-only** EP8 Stage-0 run had a
strong expert-CUDA ratio (median 1.287, p90 1.458).  That prior result is not
silently substituted for the present Qwen3-VL image trace: the current
multimodal run is reported separately and is mild on the actual expert stage.

## Hidden similarity and coalescing candidates

The bounded hidden capture contains 106,272 Vision assignment samples from
layers 16/24/40.  Same-layer, same-logical-expert pairs with cosine ≥0.90
produce 3,207 disjoint candidate pair rows (6.04% of the sampled assignment
pair pool), with median cosine 0.9306.  Only 18.8% of candidate pairs touch
the invocation's total critical EP rank.  Pair counts are concentrated in
the sampled layers (1,665 at layer 16, 396 at layer 24, and 1,146 at layer
40); the other 45 layers have no hidden-output evidence.

This is a candidate redundancy signal, not `EXPERT_OUTPUT_REDUNDANCY`:
actual EP8 expert outputs were not captured, so
`||E_e(h_i)-E_e(h_j)||` and logits/quality preservation are unknown.

### Requested assignment budgets

The table reports median removal and max-rank assignment reduction over all
864 invocation/layer views.  The reduction is an empirical count proxy, not
measured GPU speedup.

| requested budget | RANDOM removal / max-rank proxy | REDUNDANCY_ONLY removal / proxy | CRITICAL_RANK_AWARE removal / proxy |
|---:|---:|---:|---:|
| 5% | 5.01% / 4.97% | 0.00% / 0.00% | 0.00% / 0.00% |
| 10% | 10.01% / 9.95% | 0.00% / 0.00% | 0.00% / 0.00% |
| 20% | 20.01% / 19.93% | 0.00% / 0.00% | 0.00% / 0.00% |
| 30% | 30.01% / 29.83% | 0.00% / 0.00% | 0.00% / 0.00% |

The zeros for the two similarity policies are a meaningful scarcity result,
not an optimizer failure: the sampled pair pool is smaller than the requested
5–30% assignment budgets and is absent in most layers.  RANDOM is an
unconstrained diagnostic that removes arbitrary assignments; it is not a
quality-preserving coalescing baseline and must not be interpreted as gain.

### Available-pair matched diagnostic

To avoid comparing a full random budget with a capped similarity budget, an
additional diagnostic selects the same fraction of each invocation's
available candidate-pair pool.  Values below are median max-rank assignment
reduction:

| selected fraction of available pairs | RANDOM_MATCHED | REDUNDANCY_ONLY | CRITICAL_RANK_AWARE |
|---:|---:|---:|---:|
| 25% | 0.12% | 0.18% | 0.60% |
| 50% | 0.30% | 0.26% | 0.60% |
| 75% | 0.60% | 0.53% | 0.60% |
| 100% | 0.60% | 0.60% | 0.60% |

At the three sampled layers together, Critical-Rank-Aware is 0.47% median;
the best layer-specific median is 1.00% (layer 40).  The critical-aware
ordering therefore has a small directional advantage over redundancy-only at
the same available-pair budget, but its absolute headroom is far below 3%
and vanishes once all candidates are selected.

## Cost model and gate

`predicted_expert_latency_reduction` is intentionally conservative: it uses
the reduction in the maximum EP-rank assignment count as an empirical
critical-path proxy.  It does **not** assume that removing 20% assignments
produces 20% CUDA speedup.  No modified route was executed, no expert weights
were changed, and no coalesced output was compared on GPU.

| criterion | result |
|---|---|
| real EP8 token-level trace | PASS; 864 canonical measured views, exact top-k IDs and destination ranks |
| rank pressure | PRESENT; Vision excess median 213, rank max/mean median 1.326 |
| strong actual MLLM image-trace expert straggler | NOT ESTABLISHED; expert CUDA max/mean median 1.056, p90 1.113 |
| sampled hidden redundancy | CANDIDATE ONLY; cosine ≥0.90 pairs 3,207, output similarity not captured |
| critical-rank-aware advantage | SMALL; 0.60% at matched 25–50% pair selection, ≤1.00% in a sampled layer |
| quality-preserving critical-path headroom ≥3% | NOT DEMONSTRATED |
| actual coalescing GPU speedup | NOT MEASURED |

Therefore:

- `MLLM_EP8_STRAGGLER`: **YES for rank pressure**, **NO for a strong actual
  routed-expert CUDA straggler in this bounded image workload**.
- `COALESCIBLE_VISION_WORK`: **LIMITED / not quality-validated**.
- `EXPERT_OUTPUT_REDUNDANCY`: **UNVERIFIED**.
- `CRITICAL_AWARE_ADVANTAGE`: **DIRECTIONAL but sub-percent absolute proxy**.
- `GAIN_AT_5/10/20/30_PERCENT_BUDGET`: no quality-preserving gain is
  established; similarity policies are pair-pool capped.
- `RANDOM_GAIN`: only an unconstrained assignment-count diagnostic, not a
  valid quality-preserving gain.

The preregistered gate is **NO_GO** because the quality-preserving
critical-rank headroom is below 3% in the trace-driven proxy and expert-output
equivalence is absent.  This also satisfies the specified NO_GO condition
that critical-rank coalescible work is nearly absent at the available
evidence level.

## Interpretation and next action

The measured causal chain currently stops at:

`real visual input → Vision-heavy expert assignments → critical-rank pressure`.

The next links, `pressure → redundant expert computation → safe output
sharing → reduced actual EP critical path`, are not established.  Thus a
real coalescing implementation, token-pruning framing, or quality claim is
not justified.

The trace is reusable on four GPUs for offline analysis (`TRACE_REUSABLE_ON_4GPU:
YES`): token positions, top-k routes, modality labels, sampled hidden vectors,
timings, config, and placement are self-contained.  EP8 destination mapping
is topology-specific, so any 4-GPU study must explicitly remap it and cannot
be called an EP8 rerun.

If this direction is revisited, the single cheapest prerequisite is a bounded
EP8 expert-output capture for the existing same-expert candidate pairs at one
strong layer, followed by a quality/error check.  Do not implement a runtime
coalescer unless that measurement first clears the output-equivalence and
multi-layer critical-path gates.

## Artifacts

Trace (raw routes, timing, backend/runtime proof, manifest):

`poc_flashvep/deepep_revalidation/results/mllm_ep8_critical_rank_trace_20260904_run17/`

Trace-driven oracle analysis, figures, and gate summary:

`poc_flashvep/deepep_revalidation/results/mllm_ep8_critical_rank_coalescing_20260904_run17/`

Important files include `per_token_routes.csv.gz`,
`invocation_features.csv`, `timing_raw_flat.csv`,
`hidden_similarity_pairs.csv`, `coalescing_results.csv`,
`matched_pair_budget_results.csv`, `gate_summary.json`,
`model_config_audit.json`, `placement_map.json`, and the figures
`coalescing_gain.png`, `straggler_scatter.png`,
`critical_excess_by_request.png`, `expert_cuda_ratio_distribution.png`, and
`similarity_distribution.png`.  The reproducible analyzer is
`poc_flashvep/mllm_ep8_critical_rank_coalescing_poc/analyze_coalescing.py`;
`finalize_artifacts.py` creates the manifests and report figures.
