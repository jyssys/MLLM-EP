# Visual Expert Functional Redundancy PoC

## Decision

`VISUAL_FUNCTIONAL_COLLAPSE: NO-GO`

The preregistered claim is not supported. Under the fixed approximation rule
(`cosine >= 0.99` and relative L2 `<= 0.05`), the mean functional effective-K
is 8.000 for both visual and question/text tokens in the late layers. The
source-request-clustered text-minus-visual gap is exactly 0.000 (95% CI
[0.000, 0.000]), rather than the required >=30% visual reduction. Raw expert
outputs also point in the opposite direction: late visual outputs have lower
pairwise cosine and higher dispersion/participation rank than text outputs.

Because POC3 is NO-GO, the conditional Top-8 to Top-4/2 forward intervention
(POC4) was not run. A modality-asymmetric dynamic-K method is not justified by
this workload.

## Environment and workload

- Model: Qwen3-VL-30B-A3B-Instruct, BF16
- Runtime: vLLM 0.20, `TritonExperts`, DeepEP high-throughput
- Topology: TP2 / DP2 / EP4 / PP1, DBO off
- Physical GPUs: 1, 2, 3, 4 only
- Layers fixed before measurement: 4, 8, 12; 20, 24, 28; 36, 40, 44, 47
- Workload: 24 existing local images (8 natural, 8 fine-grained/scientific,
  8 chart/document), all processed at 448x448
- Prompt: fixed `Describe the image briefly.`
- Sampling: deterministic SHA-1 25% token sample, fixed before measurement
- Analyzed tokens: 11,642 visual and 595 post-image question/text tokens

The request source alternated between DP ranks while all four EP ranks
participated. The manifest records each source image, visual span, prompt token
count, category, schedule, fixed layer set, thresholds, and physical GPU set.

## Measurement validity

For each sampled token, the stock Top-8 IDs and weights were retained. The
stock expert implementation was invoked once per selected slot with that slot's
weight set to one and all other weights set to zero. These diagnostic results
were not fed into subsequent model layers. Outputs were copied immediately
because TritonExperts reuses an internal output buffer.

As an end-to-end check, the captured unweighted outputs were recombined as
`sum_e p_e E_e(x)` and compared with the stock local expert output summed over
the four EP ranks:

| Check | Result |
|---|---:|
| Minimum cosine | 0.9999948 |
| Median cosine | 0.9999963 |
| Maximum relative L2 | 0.3378% |
| Median relative L2 | 0.2732% |
| Top-8 slot coverage | exactly once per token |

The small residual is consistent with storing captured tensors as FP16 and
recombining on CPU; it is far below the functional-K threshold.

## POC1: expert-output diversity

Values below are token-weighted means over each fixed layer region.

| Region | Modality | Pairwise cosine | Normalized dispersion | Participation rank | Router effective-K |
|---|---|---:|---:|---:|---:|
| Early | Visual | 0.0116 | 0.9459 | 6.980 | 7.727 |
| Early | Text | 0.0296 | 0.9478 | 6.401 | 7.136 |
| Middle | Visual | 0.0130 | 0.9645 | 6.197 | 7.535 |
| Middle | Text | 0.0225 | 0.9471 | 6.584 | 7.150 |
| Late | Visual | 0.0972 | 0.9344 | 5.146 | 7.777 |
| Late | Text | 0.1434 | 0.9094 | 4.863 | 7.247 |

For late layers, text-minus-visual pairwise cosine is +0.0481 (clustered 95%
CI [0.0202, 0.0770]), normalized dispersion is -0.0253 (CI [-0.0408,
-0.0099]), and participation rank is -0.318 (CI [-0.542, -0.097]). All three
directions mean the selected visual expert outputs are *more*, not less,
functionally diverse.

Layer 47 shows depth-related collapse for both modalities, but it remains
stronger for text: pairwise cosine is 0.355 for visual versus 0.527 for text,
and participation rank is 2.553 versus 2.138. This is counter-evidence to a
visual-specific collapse interpretation.

## POC2: functional effective-K

| Region | Visual K | Text K | Visual router K | Text router K |
|---|---:|---:|---:|---:|
| Early | 8.000 | 8.000 | 7.727 | 7.136 |
| Middle | 7.950 | 8.000 | 7.535 | 7.150 |
| Late | 8.000 | 8.000 | 7.777 | 7.247 |

The strongest preregistered per-layer gap occurs at layer 24: text-minus-visual
K is 0.148 (95% CI [0.074, 0.237]), only 1.85% of text K. This is well below
both the 30% GO requirement and the 10% partial-effect HOLD rule.

The approximation errors reinforce the hard functional-K result:

| Late approximation | Visual cosine | Visual rel. L2 | Text cosine | Text rel. L2 |
|---|---:|---:|---:|---:|
| Top-2 | 0.726 | 1.309 | 0.820 | 1.017 |
| Top-4 | 0.867 | 0.685 | 0.942 | 0.475 |
| Top-6 | 0.946 | 0.358 | 0.982 | 0.213 |

Even Top-6 fails the 0.99/0.05 rule by a wide margin. Visual approximation is
consistently worse than text at the same m.

## POC3: modality-specificity gate

`VISUAL_FUNCTIONAL_COLLAPSE: NO-GO`

- Late visual K is not lower than late text K: 8.000 versus 8.000.
- The clustered CI excludes the requested positive gap by collapsing at zero.
- Every raw-output diversity direction contradicts visual collapse.
- Router-only metrics also show broader visual routing: late entropy
  effective-K is 7.777 visual versus 7.247 text; 95%-mass K is 7.993 versus
  7.863.
- Router adjustment cannot rescue a raw functional gap that is zero.

The observed depth effect is therefore not evidence of MLLM-specific visual
functional redundancy. At layer 47, both modalities become more aligned, and
text collapses more strongly.

## POC4: conditional headroom

`POC4: NOT-RUN`

The POC3 condition was not met. No router, expert assignment, weight, runtime
kernel, or production forward path was modified, and no accuracy claim is made.

## Figures and raw results

- `figures/plot1_expert_output_diversity_by_layer.png`: raw expert-output
  diversity by fixed layer and modality.
- `figures/plot2_functional_k_by_modality.png`: functional effective-K by
  layer; the curves remain at or extremely near eight.
- `figures/plot3_router_k_vs_functional_k.png`: router entropy effective-K
  against raw-output functional-K.
- `token_metrics.csv`: per sampled token/layer metrics and Top-m errors.
- `raw_output_correctness.csv`: stock-output reconstruction checks.
- `layer_summary.csv`, `region_summary.csv`, and `summary.json`: aggregates and
  fixed-gate decision.

Result directory:
`poc_flashvep/deepep_revalidation/results/visual_expert_functional_redundancy_20260825_154000/`

The 455 MiB per-rank raw expert tensors remain local and are intentionally not
committed; the manifest, aggregate/token metrics, correctness checks, and
figures are sufficient to audit the reported calculations.

## Limitations

- The bounded suite has 24 images and one fixed question, not a broad accuracy
  benchmark.
- Question/text token count is much smaller than visual token count, although
  uncertainty is clustered by source request rather than treating tokens as
  independent samples.
- Generated answer tokens were not added; this was optional and would require a
  separate decode attribution path.
- Diagnostic expert calls add substantial profiling overhead. They preserve the
  returned stock forward tensor but are not suitable for latency measurement.
- The conclusion applies to the fixed Top-8 Qwen3-VL BF16 configuration and
  tested layers/thresholds.

## Strongest evidence and counter-evidence

The strongest evidence that any functional reduction can occur is the small
layer-24 visual mean K of 7.853 versus text 8.000. Its 1.85% relative gap is not
system-relevant under the preregistered gate.

The strongest counter-evidence is that late visual and text K are both exactly
8.000 while all three independent output-diversity metrics show greater visual
diversity. Top-4 visual reconstruction has only 0.867 cosine and 68.5% relative
L2 error.

## Next single recommended action

Do not pursue modality-asymmetric dynamic-K for this model. Pivot to a
routing-preserving optimization whose benefit does not assume redundant visual
expert outputs.
