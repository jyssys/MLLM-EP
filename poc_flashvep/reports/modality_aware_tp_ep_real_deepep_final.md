# Modality-aware TP↔EP final bounded validation

Date: 2026-09-04  
Branch: `flashvep/modality-aware-tp-ep-crossover-poc`  
Model: Qwen3-VL-30B-A3B-Instruct (local snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`)  
GPU: physical 4, 5, 6, 7 only (`CUDA_VISIBLE_DEVICES=4,5,6,7`)  
Precision: BF16; vLLM 0.20.0 V1; eager; TritonExperts; DBO/prefix cache off; linear placement.

## Final gate

**FINAL STATUS: HOLD**

Real DeepEP was active and gave a clear **load-dependent** T_MoE crossover:
it was slower at the small regime and faster at medium/large regimes. However,
within each matched volume, Text-heavy, Mixed, and Vision-heavy workloads all
selected the same topology direction. Thus this run does not establish a
modality-dependent TP↔EP selector.

| Field | Result |
|---|---|
| `TP_ONLY_RUNTIME_VERIFIED` | YES |
| `DEEPEP_RUNTIME_VERIFIED` | YES |
| `OPTIMAL_TOPOLOGY_DEPENDS_ON_MODALITY` | NO |
| `OPTIMAL_TOPOLOGY_DEPENDS_ON_LOAD` | YES |
| `TOKEN_COUNT_ONLY_EXPLAINS_EFFECT` | PARTIAL (volume dominates; modality adds no robust crossover) |
| `LARGE_BATCH_STRAGGLER` | weak / not a primary explanation |

The gate is HOLD rather than GO because the observed topology switch is from
small to medium/large volume, not a repeated Text↔Vision crossover at matched
volume. No dynamic topology method is justified by this evidence.

## Runtime and backend proof

The TP-only proof reports TP4/DP1 with `MoEPrepareAndFinalizeNoDPEPModular`.
The REAL_DEEPEP proofs report TP2/DP2/EP4, `use_sequence_parallel_moe=true`,
and `all2all_backend=deepep_high_throughput`. The startup log contains both
`Using DeepEPHTAll2AllManager all2all manager` and
`Using DeepEPHTPrepareAndFinalize`; every retained REAL_DEEPEP row has
`prepare_finalize_backend=DeepEPHTPrepareAndFinalize` and
`expert_backend=TritonExperts`.

This follows the local vLLM source condition
`use_all2all_kernels = dp_size > 1 and use_ep`: TP4/DP1 cannot activate the
all-to-all path, while TP2/DP2/EP4 does. The complete proof is in
`backend_proof.md` and the two `runtime_proof*.json` files in the result
directory.

CAI was used only as a clean instrumentation/reference scaffold at commit
`9c73c8eee6ca64836eb873e77aa096fb4955e658`; no capacity/drop method was used.

## Workloads and token matching

The same deterministic requests were run at both topologies. Processor counts
were measured before the GPU run with the same Qwen3-VL processor. The table
shows actual prompt tokens and visual-token fraction (the routed assignment
volume is top-8 per token; DeepEP source TP rows are summed in the route
diagnostic).

| Volume | Workload | Images | Prompt tokens | Vision tokens | Vision fraction |
|---|---|---:|---:|---:|---:|
| small | Text-heavy | 0 | 257 | 0 | 0.000 |
| small | Mixed | 1 | 299 | 196 | 0.656 |
| small | Vision-heavy | 1 | 234 | 196 | 0.838 |
| medium | Text-heavy | 0 | 1,024 | 0 | 0.000 |
| medium | Mixed | 2 | 2,089 | 1,568 | 0.751 |
| medium | Vision-heavy | 2 | 1,647 | 1,568 | 0.952 |
| large | Text-heavy | 0 | 4,105 | 0 | 0.000 |
| large | Mixed | 4 | 5,932 | 4,900 | 0.826 |
| large | Vision-heavy | 6 | 7,463 | 7,350 | 0.985 |

There were two measured repetitions per request (18 paired observations); the
TP-only run also had one additional diagnostic repetition, which was not used
for pairing. Greedy output token IDs and prompt token counts matched for all
18 pairs (`correctness_check.json`: PASS).

## Primary metric and paired result

`T_MoE` is the sum over the 48 layer-level CUDA-event critical spans from
dispatch/prepare through combine/finalize. Dispatch, expert, and combine are
reported as the sum of their layer spans. This is the primary topology metric;
the request-wall column is not treated as TTFT because the DP2 driver includes
an idle DP participant and explicit per-wave synchronization. Both topology
runs used the same read-only per-MoE CUDA synchronization capture mode, so the
paired comparison is valid, while absolute phase values should not be read as
production TTFT.

| Volume | Workload | TP-only T_MoE (ms) | Real DeepEP T_MoE (ms) | DeepEP reduction |
|---|---|---:|---:|---:|
| small | Text-heavy | 65.5 | 74.0 | −13.2% |
| small | Mixed | 69.3 | 76.0 | −9.9% |
| small | Vision-heavy | 61.6 | 73.3 | −19.1% |
| medium | Text-heavy | 121.1 | 102.3 | +15.5% |
| medium | Mixed | 200.5 | 146.6 | +26.9% |
| medium | Vision-heavy | 166.8 | 128.6 | +22.8% |
| large | Text-heavy | 350.5 | 229.1 | +34.6% |
| large | Mixed | 492.9 | 394.1 | +20.1% |
| large | Vision-heavy | 637.9 | 369.1 | +42.1% |

Positive means REAL_DEEPEP is faster. Small requests pay DeepEP dispatch /
combine overhead. At medium and large volumes, expert work is large enough
that EP's sharding/all-to-all path wins for every modality.

The complete per-repetition table is `paired_comparisons.csv`. The median
across all 18 pairs is **19.24% T_MoE reduction**, but this aggregate is not a
modality claim: it is dominated by volume.

## Breakdown

Median phase times per layer/request group are below (ms, TP-only / DeepEP).

| Volume | Dispatch | Expert | Combine |
|---|---:|---:|---:|
| small | 1.0 / 16.3 | 18.9 / 23.4 | 1.7 / 5.8 |
| medium | 1.0 / 19.5 | 22.9–24.4 / 27.6–28.9 | 0.5–1.1 / 6.4–7.2 |
| large | 0.9 / 19.4–25.7 | 30.5–43.0 / 36.5–49.8 | 0.7–1.1 / 13.7–19.1 |

DeepEP has a persistent communication cost (dispatch/combine), but TP-only
expert execution grows more steeply with global token volume. Therefore the
load crossover is a phase trade-off, not a Vision-specific communication
shape. One large Mixed repetition had a 199.9 ms dispatch span under the
diagnostic synchronized capture; the paired median remained positive but this
outlier is retained and visible in raw CSVs.

## Routing and large-batch diagnostic

The routed expert histograms and EP4 destination proxy are in
`layer_metrics.csv`/`route_statistics.csv`. Representative median layer
statistics were:

| Topology | Volume | Rank max/mean load | Expert CUDA max/mean |
|---|---|---:|---:|
| TP-only | small | 1.188 | 1.022 |
| TP-only | medium | 1.147 | 1.016 |
| TP-only | large | 1.147 | 1.016 |
| REAL_DEEPEP | small | 1.190 | 1.010 |
| REAL_DEEPEP | medium | 1.141 | 1.013 |
| REAL_DEEPEP | large | 1.147 | 1.025 |

These are not ReaLB/MACS-like large stragglers. Rank-load skew remains near
1.15–1.19 and measured expert-time skew near 1.01–1.03. The dominant large-
volume advantage is therefore not explained by a new critical-rank imbalance.

## Volume versus modality effect

Simple diagnostics over paired data give:

- correlation of T_MoE with prompt tokens: **0.999**;
- correlation of relative DeepEP gain with prompt tokens: **0.743**;
- correlation of relative gain with visual fraction: **0.136**.

Within every volume the direction is consistent:

- small: TP-only wins for Text, Mixed, and Vision;
- medium: REAL_DEEPEP wins for all three;
- large: REAL_DEEPEP wins for all three.

The visual fractions change substantially (0 → 0.985), but no matched-volume
topology winner changes. Modality median reductions (Text/Mixed/Vision) are
15.5%/14.2%/22.8% across all scales; this modest aggregate spread is not a
repeated crossover and is sensitive to the large Mixed outlier. Thus the
evidence supports a **load-conditioned static choice**, not modality-aware
TP↔EP switching.

## Answers to the requested questions

1. Real DeepEP all-to-all is active only in TP2/DP2/EP4 and is proven by both
   runtime class and startup log.
2. No topology is universally best: TP-only is better for this small regime;
   DeepEP is better at medium/large volume.
3. Modality composition does not change the winner at matched volume. The
   observed relative gain is primarily volume/load-conditioned.
4. Dispatch/combine communication is the DeepEP fixed cost; expert compute is
   the term that makes EP favorable at larger volumes.
5. Large-batch rank imbalance is weak in this run and does not provide a
   separate straggler motivation.
6. A simple token-volume threshold may be useful for a static deployment
   choice, but a modality-aware dynamic TP↔EP policy is not justified.

## Artifacts and limitations

Raw traces are preserved in:

- `../modality_aware_tp_ep_real_deepep_20260904_tp_only_sync/`
- `../modality_aware_tp_ep_real_deepep_20260904_real_deepep_v6/`

The DeepEP raw directory contains all four worker files. DP1's 8-assignment
padding forwards are retained but excluded by the analyzer; request-owned DP0
source rows are used for the paired T_MoE. The synchronized diagnostic hook
adds overhead and can amplify an individual dispatch span; both topologies
use it, and all raw values are available for audit. Request wall is reported
but deliberately not used as the topology gate.

Analysis files: `layer_metrics.csv`, `request_metrics.csv`,
`paired_comparisons.csv`, `volume_summary.csv`, `modality_summary.csv`,
`modality_by_volume.csv`, `correctness_check.json`, `gate_summary.json`, and
the three PNG figures in the result directory.

**Next single action:** close the modality-aware TP↔EP direction for now and,
if deployment work is needed, validate one conservative volume-based static
topology threshold on a real serving trace. Do not implement dynamic
modality-aware switching from this PoC.
