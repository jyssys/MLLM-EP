# Official-calibration quality confirmation (fresh GPU)

Frozen held-out pool:128 ChartQA and128 GQA questions,128 and89 image clusters.
GQA calibration and held-out images are disjoint. Natural greedy generation,
max32 tokens, identical short-answer prompt, official full-prediction scoring.
HF replicas provide quality evidence, **not EP speed measurements**.

| Policy | ChartQA delta vs vanilla, pp (95% image-cluster CI) | GQA delta, pp (95% CI) |
|---|---:|---:|
| SERE S2/rho .5 | -11.71875 [-18.75,-5.46875] | -8.59375 [-14.0764,-3.4953] |
| SERE S4/rho .5 | -.78125 [-2.34375,0] | 0 [-3.0303,3.1502] |
| SERE S2/rho .7 | -.78125 [-4.6875,3.125] | +.78125 [-3.1752,4.6154] |
| MoDES target70 | -2.34375 [-7.03125,2.34375] | -.78125 [-5.46875,3.9683] |
| MoDES target85 | -4.6875 [-8.59375,-.78125] | -.78125 [-5.46875,3.8469] |

Vanilla scores:89.0625% ChartQA,56.25% GQA. These are bounded subsets, not the
full published benchmarks. A point estimate close to zero is not proof of
noninferiority. MoDES thresholds are from the **completed** official1024-question,
100-grid frontier; actual calibration skipping71.328% and85.267%, KL .0110761
and .0170227. They are not the earlier coarse pilot.

## SERE obvious-fix / membership control

With HF fixed B16 cohorts the aggressive SERE quality delta is0pp on both tasks.
Among samples vanilla answers correctly at both B1 and B16,17/17 ChartQA and
11/12 GQA newly wrong B1 SERE answers recover at B16. S4/rho and rho=.7 also
recover most B1 loss. No held-out labels were used to choose these paper-like
trivial controls. **Actual online batch membership must be measured separately**:
HF fixed cohorts keep finished/padded rows, unlike continuous-batching EOS
removal. Neither an MLLM-specific cause nor a scheduler solution follows from
this comparison alone.

Official B1 termination diagnostic across both tasks:29 vanilla-correct/SERE-wrong
answers; only4 lack EOS at the32-token cap. First-line-only scoring would recover
4; labeled-answer-prefix checking finds10. These are optimistic diagnostics,
not alternate task scores. Content/numeric errors remain. MoDES wrong answers
do not recover under the first-line diagnostic and all terminate in this pool.

## Claim boundaries

The fresh vLLM results are separate measurements and can differ from HF because
backend arithmetic, dynamic batch membership and generated trajectories differ.
Compare each treatment with its own matched vanilla. The older exploratory
FP32-Gram SERE table is not used for confirmatory conclusions. SERE's original
single-device batch-union is adapted explicitly to each DP-local batch before
TP sequence sharding; no claim about an untested EP-global primary-set design.

Data: result root `analysis/official_confirmatory_b1_20260908`,
`analysis/sere_batch_control_20260908`, `analysis/official_termination_20260908`,
and `quality/modes_frontier1024_grid100`. Serving,port-cost and native-Libra
controls are now completed within the boundaries in the final main report.
The final all-three verdict is FOUND_INCREMENTAL_ONLY; these HF quality numbers
are not relabeled as distributed speed measurements.
