# Intra-Expert Visual Token Redundancy PoC

## Final decision

`INTRA_EXPERT_VISUAL_REDUNDANCY: NO-GO`

The tested Qwen3-VL workload does not support representative expert
computation for visual tokens. After matching Visual and Text/Question group
sizes inside the same `(layer, expert)`, visual expert outputs are consistently
*less* redundant. At 50% representatives, even an output-space oracle proxy
reconstructs non-representative visual outputs with median cosine 0.474 and
relative L2 1.008. Hidden-state selection is slightly worse (0.456 / 1.025).

The conditional system-headroom stage was therefore not run. No routing,
DeepEP, expert kernel, model weight, or forward behavior was modified.

## Environment and preregistered policy

- Model: Qwen3-VL-30B-A3B-Instruct, BF16
- Runtime: vLLM 0.20, `TritonExperts`, DeepEP high-throughput
- Topology: TP2 / DP2 / EP4 / PP1, DBO off
- Physical GPUs: 1, 2, 3, 4 only
- Fixed layers: 4, 12, 24, 36, 44, 47
- Workload: 24 existing local images, with 8 each from natural,
  fine-grained/scientific, and chart/document categories
- Image input: 448x448; fixed prompt `Describe the image briefly.`
- Token sampling: deterministic SHA-1 25% sample
- Matched analysis: same `(layer, global expert)`, same number of visual and
  text assignments, minimum 8 and maximum 32 per modality
- Representative ratios: 25%, 50%, 75%, 100%
- Quality target: non-representative cosine >=0.99 and relative L2 <=0.10
- Oracle proxy: deterministic greedy medoids selected using raw output-space
  cosine distance
- Practical selector: the same procedure using only pre-expert hidden-state
  cosine distance

The matching policy, layers, ratios, and gate were written to the manifest
before GPU execution. There are 115 matched layer-expert groups and 1,786
matched assignments per modality. A typical matched group covers 18 visual
source images and 12 text source images (minimum 4 and 6, respectively).

## Capture validity

For each assignment the capture stores the pre-expert hidden state `x`, raw
unweighted output `E_e(x)`, original Top-8 expert ID, and original routing
weight. Diagnostic expert calls preserve the actual stock forward result.

Recombining captured expert outputs with the original router weights and
comparing with the stock result gives:

| Check | Result |
|---|---:|
| Minimum cosine | 0.9999948 |
| Median cosine | 0.9999963 |
| Maximum relative L2 | 0.3378% |
| Median relative L2 | 0.2720% |
| Expert-slot coverage | exactly once |

The residual is consistent with saving diagnostic tensors as FP16 before CPU
analysis.

## POC1 — Within-expert output redundancy

All values use equal-sized Visual/Text samples from the same layer and expert.

| Region | Modality | Pairwise cosine | Dispersion | Centered participation rank | NN relative L2 |
|---|---|---:|---:|---:|---:|
| Early | Visual | 0.244 | 0.844 | 9.080 | 1.020 |
| Early | Text | 0.510 | 0.643 | 3.202 | 0.337 |
| Middle | Visual | 0.125 | 0.919 | 9.584 | 1.129 |
| Middle | Text | 0.453 | 0.695 | 3.903 | 0.450 |
| Late | Visual | 0.368 | 0.791 | 6.477 | 0.872 |
| Late | Text | 0.532 | 0.658 | 3.108 | 0.351 |

Across all 115 paired groups, Visual minus Text is:

| Metric | Difference | 95% group-bootstrap CI | Visual redundancy direction? |
|---|---:|---:|---|
| Pairwise cosine | -0.218 | [-0.262, -0.173] | No |
| Normalized dispersion | +0.166 | [+0.129, +0.207] | No |
| Centered participation rank | +4.478 | [+3.883, +5.123] | No |
| NN relative L2 | +0.594 | [+0.550, +0.639] | No |

The opposite direction repeats in all three image categories. Category-matched
groups number 29 chart/document, 22 natural, and 17 fine-grained. Visual
pairwise cosine is lower by 0.107, 0.116, and 0.086, respectively; the other
three diversity metrics also consistently show greater visual diversity.

Layer 47 is the only narrow positive hint: visual pairwise cosine is 0.747
versus text 0.706. It is contradicted by centered participation rank (5.649
visual versus 2.228 text) and nearest-neighbor relative L2 (0.476 versus
0.225), so it does not constitute functional collapse.

## POC2 — Representative-compute oracle

Quality below is evaluated only on non-representative tokens. Representative
tokens' trivial exact matches are excluded from the gate.

### Output-space oracle proxy

| Representatives | Visual cosine | Visual rel. L2 | Visual pass | Text cosine | Text rel. L2 | Text pass |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.404 | 1.083 | 1.14% | 0.953 | 0.318 | 5.63% |
| 50% | 0.474 | 1.008 | 1.71% | 0.971 | 0.241 | 9.79% |
| 75% | 0.600 | 0.897 | 4.39% | 0.983 | 0.185 | 22.20% |

### Pre-expert hidden-state practical selector

| Representatives | Visual cosine | Visual rel. L2 | Visual pass | Text cosine | Text rel. L2 | Text pass |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.382 | 1.083 | 1.09% | 0.952 | 0.321 | 6.11% |
| 50% | 0.456 | 1.025 | 1.79% | 0.970 | 0.247 | 9.91% |
| 75% | 0.578 | 0.927 | 4.50% | 0.982 | 0.188 | 22.75% |

The oracle-practical gap is small compared with the much larger failure to
approximate visual outputs. This is not a case where a strong output-space
oracle is merely inaccessible from hidden states; the oracle itself fails.

Both modalities have a median required representative ratio of 100%. At the
group level, Text minus Visual required ratio is -4.13 percentage points for
the oracle (95% CI [-6.52, -1.96] pp) and -3.91 points for the practical
selector (CI [-6.30, -1.96] pp). Thus Visual tends to require slightly *more*
representatives, opposite the required >=20-point advantage.

## POC3 — Visual-specificity

`INTRA_EXPERT_VISUAL_REDUNDANCY: NO-GO`

1. Visual is not more redundant than Text within the same expert. All four
   primary diversity metrics show the opposite with CIs excluding zero.
2. The result repeats across early/middle/late layers and all three image
   categories, rather than arising from one image or expert.
3. Pre-expert hidden-state selection closely tracks the output-space oracle,
   but neither provides useful visual reconstruction.

This is not a generic depth-collapse result being mislabeled as MLLM-specific.
Visual outputs remain more diverse than Text at nearly every tested layer; a
narrow layer-47 cosine reversal does not survive the rank and nearest-neighbor
checks.

## POC4 — Conditional system headroom

`POC4: NOT-RUN`

POC3 is NO-GO, so theoretical GEMM-row/EP-assignment reduction and the
spatial-neighbor comparison were not produced. Reporting a reduction from the
representative fraction alone would be misleading because non-representative
visual outputs fail the required fidelity by a wide margin.

## Figures and result files

- `plot1_within_expert_output_redundancy.png`: layer-wise matched diversity.
- `plot2_reconstruction_vs_rep_ratio.png`: non-representative reconstruction
  for output-space oracle and hidden-state practical selection.
- `plot3_visual_vs_text_rep_ratio.png`: representative fraction required by
  the fixed fidelity rule.
- `within_expert_diversity.csv`: one row per matched modality/layer/expert.
- `representative_reconstruction.csv`: method/ratio reconstruction metrics,
  including representative-inclusive and non-representative-only fields.
- `required_representative_ratio.csv`: fixed-target ratio per group.
- `category_matched_diversity.csv`: category-controlled replication.
- `matched_group_manifest.csv`: group sizes and source-image coverage.
- `raw_output_correctness.csv`: reconstruction integrity checks.

Result directory:
`poc_flashvep/deepep_revalidation/results/intra_expert_visual_redundancy_20260825_164500/`

The 355 MiB raw hidden/output tensors remain local and are not committed. The
manifest, group-level data, correctness checks, summaries, and figures are
committed for auditability.

## Limitations

- The suite contains 24 images and one fixed question, not a broad accuracy
  benchmark; generated answer tokens were not included.
- Only a deterministic 25% assignment sample and six fixed layers were
  captured.
- The output-space selector is a deterministic greedy-medoid upper-bound proxy,
  not a globally optimal combinatorial medoid solution. Its large failure
  margin makes that optimization gap unlikely to reverse this gate, but this is
  not formally proven.
- Diagnostic expert calls are suitable for functional profiling, not latency
  measurement.
- Group bootstrap treats matched layer-expert groups as resampling units. The
  category replication and image-coverage counts provide additional robustness,
  but no accuracy benchmark is claimed.

## Strongest evidence and counter-evidence

The strongest positive observation is only layer 47's +0.041 visual pairwise
cosine difference. It conflicts with the other output-space metrics at that
same layer.

The strongest counter-evidence is the 50% output-space oracle result: visual
non-representative outputs achieve just 0.474 cosine and 1.008 relative L2,
while every aggregate diversity metric and all image categories independently
show greater visual diversity.

## Next single recommended action

Do not pursue representative visual expert computation for this model. Focus
future work on exact per-token computation or communication/runtime mechanisms
that preserve every routed visual assignment.
