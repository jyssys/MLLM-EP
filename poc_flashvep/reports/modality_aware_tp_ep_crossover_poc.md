# Modality-aware TP/EP crossover PoC

## Executive result

**FINAL STATUS: `NO_GO` for the requested modality-dependent TP/EP crossover.**

The three matched real Qwen3-VL populations did not choose different static parallelization strategies: the EP-flag path was faster for text-heavy, mixed, and vision-heavy inputs alike. More importantly, the requested `TP4/DP1/EP4` configuration does not instantiate DeepEP all-to-all communication in the local vLLM 0.20.0 source. Its runtime backend is `MoEPrepareAndFinalizeNoDPEPModular`, the same no-DP/EP prepare-finalize class seen by the TP-only control. Therefore the roughly 29--30% mean (22--36% paired-median) T_MoE gap is an expert-sharding versus TP-only result, **not** evidence of a TP↔DeepEP crossover. A dynamic modality-aware topology selector is not justified by this PoC.

## Reproduction and provenance

| Item | Value |
|---|---|
| Model | Qwen3-VL-30B-A3B-Instruct |
| Model snapshot | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Hardware | 4× NVIDIA H100 80GB HBM3 |
| Visible/physical GPUs | `CUDA_VISIBLE_DEVICES=1,2,3,4` / physical 1–4 |
| Runtime | local vLLM 0.20.0, eager, BF16, TritonExperts |
| MoE config | 48 layers, 128 routed experts, top-8, hidden size 2048 |
| Placement | linear; diagnostic rank map `expert_id // 32` |
| Common controls | DBO off, prefix cache off, max model length 4096, max batched tokens 4096 |
| Repetitions | 2 warmups + 8 measured waves per workload and topology |
| CAI reference | `/tmp/Capacity-Aware-MoE`, commit `9c73c8eee6ca64836eb873e77aa096fb4955e658` |

The prior 8-GPU EP8 trace `mllm_ep8_critical_rank_trace_20260904_run17` was
used only to motivate visual-routing pressure and workload selection. No EP8
latency was projected onto this 4-GPU result.

Exact commands used:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 \
PYTHONPATH="$PWD/poc_flashvep/modality_aware_tp_ep_crossover/hooks:$PWD" \
/home/esjung/anaconda3/envs/flashvep-poc/bin/python \
  poc_flashvep/modality_aware_tp_ep_crossover/run_crossover.py \
  --topology {ep4|tp_only} --warmups 2 --iterations 8 --decode-tokens 1 \
  --output <run-directory>
```

```bash
python poc_flashvep/modality_aware_tp_ep_crossover/analyze_crossover.py \
  --ep4 poc_flashvep/deepep_revalidation/results/modality_aware_tp_ep_crossover_poc_20260904_final_ep4 \
  --tp-only poc_flashvep/deepep_revalidation/results/modality_aware_tp_ep_crossover_poc_20260904_final_tp_only \
  --output poc_flashvep/deepep_revalidation/results/modality_aware_tp_ep_crossover_poc_20260904_final_analysis
```

The hook wraps existing `FusedMoE.forward`, `_prepare`, `_fused_experts`, and `_finalize` calls and resolves CUDA events once at worker exit. It does not modify routing, weights, placement, scheduling, or model math. Prompt tokenization and one-token greedy outputs were checked per workload.

## Workload matching

Four real local images (astronaut, camera, coffee, and Chelsea) were used for the visual populations at 896×896. Vision fractions are exact processor input counts of `image_token_id=151655`.

| Workload | Images | Prompt tokens | Vision tokens | Vision fraction | Description |
|---|---:|---:|---:|---:|---|
| text-heavy | 0 | 2,980 | 0 | 0.000 | long text control |
| mixed | 2 | 3,022 | 1,568 | 0.519 | two images + long text |
| vision-heavy | 4 | 3,163 | 3,136 | 0.991 | four images + short compare text |

Token volume is within 6.1%. With top-8 routing, assignment volumes are
23,840, 24,176, and 25,304. The final runs contain 48 layers × 3 workloads ×
8 measured waves × 4 worker files.

## Topology audit (decisive)

| Label | Requested | Runtime proof |
|---|---|---|
| TP-only | TP4/DP1, expert parallel disabled | `use_ep=false`, `ep_size=1`, `MoEPrepareAndFinalizeNoDPEPModular` |
| EP flag | TP4/DP1, EP4, DeepEP HT requested | `use_ep=true`, `ep_size=4`, but `use_all2all_kernels=false`, `MoEPrepareAndFinalizeNoDPEPModular` |

The installed vLLM source defines:

```python
use_ep = (dp_size * pcp_size * tp_size > 1
          and parallel_config.enable_expert_parallel)
use_all2all_kernels = self.dp_size > 1 and self.use_ep
use_deepep_ht_kernels = (self.use_all2all_kernels
                          and all2all_backend == "deepep_high_throughput")
```

Thus TP4/DP1 with the EP flag shards experts across four ranks but cannot
activate distributed all-to-all/DeepEP. Both primary runs use the no-DP/EP
prepare-finalize class. A true DeepEP comparison requires DP>1 (for example
TP2/DP2/EP4); that optional secondary was not substituted into the primary
comparison, and historical runs with other requests/protocols were not pooled.

## T_MoE definition and primary results

For each layer, T_MoE is the rank-critical CUDA-event span of the existing
`FusedMoE.forward`. Dispatch, expert, and combine are separately timed around
the existing modular calls; request T_MoE is the sum of 48 layer-critical
spans. It excludes vision encoder and non-MoE decoder work.

| Workload | TP-only T_MoE (ms) | EP-flag T_MoE (ms) | EP-flag reduction | TP-only wall median (ms) | EP-flag wall median (ms) | Winner |
|---|---:|---:|---:|---:|---:|---|
| Text-heavy | 393.55 | 274.99 | 22.34% paired median | 460.69 | 336.94 | EP flag |
| Mixed | 395.93 | 279.16 | 35.78% paired median | 746.94 | 540.97 | EP flag |
| Vision-heavy | 414.85 | 290.66 | 29.53% paired median | 978.07 | 825.70 | EP flag |

Values are mean request-level sums across eight waves; reductions are medians
of paired waves. The large reduction is nearly flat as vision fraction changes,
and the winner never changes.

### Phase breakdown (mean request sums, ms)

| Workload | Topology | Dispatch | Expert | Combine | Expert share |
|---|---|---:|---:|---:|---:|
| Text-heavy | TP-only | 1.219 | 31.015 | 0.921 | 93.55% |
| Text-heavy | EP flag | 0.921 | 30.468 | 0.580 | 95.30% |
| Mixed | TP-only | 1.255 | 32.315 | 0.954 | 93.60% |
| Mixed | EP flag | 0.921 | 30.648 | 0.589 | 95.31% |
| Vision-heavy | TP-only | 1.212 | 32.269 | 0.858 | 93.97% |
| Vision-heavy | EP flag | 0.896 | 31.335 | 0.594 | 95.46% |

Expert execution dominates the measured phase sums. Since DP1 suppresses
all-to-all, this is not a communication crossover.

## Routing statistics

| Workload / topology | Active experts | Expert-load CV | Rank-load CV* | Rank max/mean* |
|---|---:|---:|---:|---:|
| Text-heavy / TP-only | 103.4 | 1.356 | 0.239 | 1.325 |
| Mixed / TP-only | 126.9 | 0.810 | 0.118 | 1.158 |
| Vision-heavy / TP-only | 127.1 | 0.726 | 0.101 | 1.131 |
| Text-heavy / EP flag | 103.0 | 1.352 | 0.239 | 1.324 |
| Mixed / EP flag | 127.0 | 0.812 | 0.119 | 1.159 |
| Vision-heavy / EP flag | 127.1 | 0.726 | 0.100 | 1.131 |

\*Rank columns are the diagnostic `expert_id // 32` histogram, not a DeepEP
communication load. Vision activates almost all experts, while text has a
narrower working set, but no topology winner changes.

## Gate and interpretation

- **VISION_DOMINATES_MOE_WORK: `NO` (at most `PARTIAL` as a composition fact).**
  Vision supplies most assignments in the four-image population, but its T_MoE
  is only about 5.4–5.6% above the text control with 6.1% more tokens.
- **TOKEN_COUNT_ONLY_EXPLAINS_EFFECT: `PARTIAL`.** Token volume explains much
  of the small visual-vs-text increase. Active-expert and CV changes confirm
  routing structure differences, but phase proportions and the winner are
  stable.
- **OPTIMAL_TOPOLOGY_DEPENDS_ON_MODALITY: `NO`.** All three populations select
  the same EP-flag path; the measured modality-dependent crossover is 0%.
- **MAX_CROSSOVER_GAIN: 0%.** The mean T_MoE gap is about 29–30% (paired
  medians 22.34–35.78%) and is a flat
  expert-sharding/no-DP-EP effect and is not a TP↔DeepEP crossover.

The requested gate is `NO_GO`: no modality-dependent static crossover was
observed, and the requested EP-oriented primary path was not a DeepEP topology.
This closes the dynamic modality-aware TP↔EP selector direction for the tested
setup without claiming that a properly matched true TP2/DP2/EP4 study would be
identical.

## Required answers

1. Vision makes most assignments only in the vision-heavy request; the measured
   cost increase is modest and largely token-volume aligned.
2. Composition changes active working set and expert-load CV, but not phase
   proportions or topology winner.
3. No TP/EP optimum crossover was found.
4. Expert execution drives the measured difference (about 94–95% of phase sums);
   dispatch/combine changes are smaller and not DeepEP communication here.
5. Dynamic TP↔EP or hybrid runtime is not justified by this result.

## Artifacts

- Code: [`poc_flashvep/modality_aware_tp_ep_crossover/`](../modality_aware_tp_ep_crossover/)
- Analysis: [`modality_aware_tp_ep_crossover_poc_20260904_final_analysis/`](../deepep_revalidation/results/modality_aware_tp_ep_crossover_poc_20260904_final_analysis/)
- EP-flag raw run: [`..._final_ep4/`](../deepep_revalidation/results/modality_aware_tp_ep_crossover_poc_20260904_final_ep4/)
- TP-only raw run: [`..._final_tp_only/`](../deepep_revalidation/results/modality_aware_tp_ep_crossover_poc_20260904_final_tp_only/)

The analysis directory contains `layer_metrics.csv`, `route_statistics.csv`,
`request_metrics.csv`, `paired_comparisons.csv`, `crossover_curves.csv`,
`configuration_audit.json`, `cai_reference.json`, `gate_summary.json`, and
latency/ratio figures. Raw worker JSONL preserves per-layer route histograms
and CUDA-event spans; historical artifacts were not overwritten.

## Next single action

Do not implement a dynamic modality-aware TP↔EP runtime. If the question remains
important, run one separately bounded and properly matched TP2/DP2/EP4 (true
DeepEP) versus TP4/DP1 TP-only experiment on these exact populations, explicitly
reporting backend semantics. Otherwise close this selector direction and keep
modality analysis as workload/routing characterization.
