# Decode-first Dynamic-Depth-Aware DP Scheduling: v3

## Verdict

The final verdict is **`NO_GO_SCHEDULING`**.
The source-selection decision is **`INCUMBENT_CONTINUE`**.
The quality status remains **`QUALITY_WARNING`**, with **`REFERENCE_MATCH_PARTIAL`**.
The v2 `QUALITY_FAIL` result remains unchanged.
It did not stop the v3 scheduling discovery.

The campaign found physical decode-depth heterogeneity and residual cross-GPU work.
It also found conditional clairvoyant scheduling headroom.
However, the tested prompt-only predictor did not recover that headroom.
The largest modeled gain did not reproduce as a statistically clear live gain.
The evidence does not justify RL training or a new multi-GPU framework.
This is a NO-GO for proceeding from this pilot, not proof that all depth-aware schedulers must fail.

## Model and selected runtime

The model is `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The CDB authors' source SHA is `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
The incumbent already uses that implementation and the checkpoint's original gate.
No verified independent better-quality replacement was available in the bounded source audit.
The selection therefore preserves q=0.5, min R2, max R4, and `last_exited` KV.

Four independent BF16 replicas use physical GPUs 4/5/6/7.
Each process sees logical CUDA device 0. There is no inter-replica TP or EP collective.
Decode uses CUDA graphs. The quality recheck uses the retained eager evaluation path.
We do not report eager-to-graph gains as a research contribution.

## Quality and reference recovery

| Domain | Old Fixed R4 | Old Adaptive | New 8192-cap Adaptive | Fixed minus Adaptive 95% CI |
|---|---:|---:|---:|---:|
| GSM8K | 22/32, 68.75% | 21/32, 65.625% | 21/32, 65.625% | [-6.25, 12.5] pp |
| MATH-500 | 24/32, 75.00% | 19/32, 59.375% | 19/32, 59.375% | [3.125, 28.125] pp |

The old primary cap was 4096, followed by a paired 8192 truncation-union sensitivity check.
The new adaptive recheck uses cap 8192 for all 64 frozen IDs.
All shared token prefixes match the old adaptive outputs exactly.
Two MATH outputs still reach the new cap. Each domain has two parse failures.
The old physical decode-work savings were 22.86% and 27.97%.
The new MATH work denominator includes longer capped outputs and is not the old primary denominator.
We do not interpret that denominator change as an optimization gain.

The literature evaluation protocols differ from this greedy answer-extraction pilot.
No paper-equivalent absolute quality claim or fixed-R4 quality equivalence is established.
Changed CDB batch shapes also produce some changed greedy outputs.
Full benchmark quality across placement policies is `NOT_RUN`, not `PASS`.

## Stage 1: physical heterogeneity

The new pilot has 128 requests, including 32 calibration and 96 held-out requests.
Its four domains are GSM8K, MATH-500, Alpaca, and ShareGPT.
It records 31,946 online decode exits and 95,768 actual loop-token operations.
Mean request depth spans about 2.475–3.867.
The retrospective B1 cost model improves from R²=0.8712 to 0.9926 when actual depth work is included.
Its held-out MAE changes from 148.214 ms to 20.814 ms.
This is cost relevance, not causal predictability or a scheduling speedup.

The generation cap is 256 with natural EOS. Only seven requests reach EOS within that cap.
The traces characterize bounded progress, not complete-answer serving quality.
The campaign does not label this pilot as a 512-request or final large-scale study.

## Stage 2: live baselines and model validity

The live burst sweep covers C=1/2/4/8/16/32/64 total requests.
It compares round-robin, queue, calibrated length, and inexpensive calibrated cost placement.
Closed-loop and Poisson discovery use the C16 setting.
No request changes GPU after prefill.

Native CDB requires slot-homogeneous recurrent kernels under the selected KV layout.
This creates local batch fragmentation and interacts with cross-GPU load balance.
The simulator reproduces all 28 burst/rank operator-count comparisons exactly.
Median RR/queue prediction errors are about 2.00% for makespan and 1.86% for P95 E2E.
Median P95 ITL error is 1.31%. Some P95 TTFT strata remain `PARTIAL` above 10% error.

Idle/eligible witnesses confirm that one rank can finish while another retains runnable work.
Those witnesses do not establish that depth-aware admission can remove all remaining work.
The cost baseline is not proven strongest among all CDB-aware generic schedulers.
The optional async CDB path is not a validated comparator.

## Stage 3: oracle versus generic and length controls

| C | Best observed non-depth control | Control P95, ms | Oracle P95, ms | Model gain | 95% paired CI |
| --- | --- | --- | --- | --- | --- |
| 8 | round_robin | 7206.7 | 7136.6 | 0.973% | [-6.341, 2.220]% |
| 16 | round_robin | 12968.7 | 11803.0 | 8.989% | [-0.662, 12.894]% |
| 32 | round_robin | 20720.5 | 18953.4 | 8.528% | [0.000, 17.618]% |
| 64 | strong_cost | 33202.1 | 28962.4 | 12.770% | [3.696, 16.809]% |

These are modeled estimates from a best-known clairvoyant search.
C1/C2/C4 have zero additional solved-optimal headroom under the symmetric model.
Larger cases are not solved to global optimality.
The true-length heuristic sometimes loses to a generic control.
We therefore do not present its larger isolated gap as depth-specific gain.

The C64 refined estimate motivates a bounded physical hindsight-placement check.
The same pilot also supplies C16/C32 checks.

| C | Repeats | Generic mean P95, ms | Hindsight mean P95, ms | Measured gain | 95% paired CI |
| --- | --- | --- | --- | --- | --- |
| 16 | 3 | 12948.7 | 11573.6 | 10.619% | [-0.466, 14.793]% |
| 32 | 3 | 21407.6 | 19165.3 | 10.475% | [-1.745, 19.211]% |
| 64 | 3 | 33051.6 | 31593.9 | 4.410% | [-8.298, 13.341]% |

These are measured live placements selected with future information.
They are not measured causal-policy speedups.
They reuse selection requests and allow batch-dependent output/depth differences.
The three-repeat uncertainty and natural K-token controls remain visible in the raw artifacts.
They do not prove untouched-cohort quality equivalence.

At C64, the 12.77% modeled P95 gain becomes a 4.41% live point estimate.
Its paired 95% CI is [-8.298, 13.341]%.
All three live comparison intervals include zero.
The changed placements also change outputs for many requests.
Therefore, this pilot does not establish statistically clear, quality-matched additional scheduling headroom.
On the observed C64 routes, the cost model predicts the live checks within about 1.5%.
Thus accurate conditional cost prediction does not establish portable depth traces across placements.
This is not proof of a specific numerical error or gate bug.

## Stage 4: admissible prediction

| Target | Calibration N | Held-out N | Ridge MAE | Constant MAE |
| --- | --- | --- | --- | --- |
| mean_depth | 32 | 96 | 0.157745 | 0.147477 |
| output_length | 32 | 96 | 7.539452 | 7.516927 |

At C64, the tested depth-only matched-length policy worsens modeled P95 latency by about 5.56%.
The joint depth/length predictor worsens it by about 7.96%.
The depth MAE is worse than the constant control.
Thus predictor accuracy and useful scheduling gain do not pass together.
No RL, complex predictor, or new engine is used to rescue the result.

## Gates and limitations

| Gate | Result |
|---|---|
| Stage 0R | `INCUMBENT_CONTINUE`. Online gate, physical skipping, and usable CDB verified. |
| Quality | `QUALITY_WARNING`. The old MATH drop and partial reference fidelity remain. |
| Heterogeneity | `RUN`: physical variation remains beyond output-length controls. |
| Cost and DP4 | `RUN`: measured DP4, exact burst transition counts, validated primary latency model. |
| Scheduling oracle | `NOT_CONFIRMED`: conditional modeled headroom does not establish a statistically clear quality-matched live gain. |
| Causal predictor | `PREDICTION_BLOCKED` for the tested prompt-only models. |
| Prefill | `PREFILL_FULL_DEPTH`. No new prefill runtime. |
| RL | `NOT_RUN`. |

The pilot is small and heavily capped. It does not establish broad natural-EOS scheduling gains.
The oracle CI excludes cost-model and search-selection uncertainty.
Batch-dependent numerical/output changes limit quality-portability claims.
The full non-burst concurrency/rate matrix is not measured.
The generic cost controls can still be strengthened.

## Next action

Do not start RL or a new multi-GPU architecture now.
First measure logits and exit decisions with teacher-aligned prefixes at B1/B4/B16.
This small control can test the portability of depth traces across batch shapes.
Keep the same gate and KV policy. Do not train a new gate.
Any later scheduling study also needs a stronger CDB-aware generic cost baseline and fresh natural-EOS requests.

## Publication and reproduction

Branch: `flashvep/decode-first-depth-aware-dp-v3`.
Parent evidence commit: `e6a350e0a862d94251c5cefa0ef38038b017b8c4`.
The published Git HEAD identifies this code/evidence snapshot.
The final user response records its commit SHA.
Only the new v3 directory is part of this commit.
Old dirty/staged work, reports, and raw evidence remain unchanged.
Large per-rank JSON traces are published as lossless gzip copies.
Their original uncompressed files remain on the server.

See `reproduce.md`, `timing_contract.md`, and the stage reports for exact commands and definitions.

## Figures

- [Online depth and cost](../figures/depth_heterogeneity.png)
- [Live DP4 scaling](../figures/live_dp4_scaling.png)
- [Cost-model validation](../figures/cost_model_validation.png)
- [Modeled oracle headroom](../figures/oracle_headroom.png)
- [Measured hindsight checks](../figures/live_hindsight_confirmation.png)
- [Prompt-only predictor](../figures/causal_predictor_scheduling.png)
