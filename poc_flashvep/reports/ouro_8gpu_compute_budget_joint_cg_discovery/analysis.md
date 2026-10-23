# Compute-Budget-Aware Joint Chunk–Loop Optimization

Final verdict: **NO_GO_INCREMENTAL** for the implemented Joint optimizer on the tested RLT/Ouro Static-R4 workloads.
Conditional C/G cost interaction exists, but Joint adds no executed operating point beyond the tuned G-only control.
This is not a proof that every possible C/G policy or backend must fail.

## Research contract

The model is pretrained Base `ByteDance/Ouro-1.4B`, BF16, Static R4.
The model snapshot is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
The engine is pinned RLT `4a350344027960a48384626aa646f1765fda6d4f` with the recorded removable G2/G3 patch.
Eight physical H100 GPUs run independent single-GPU experiments.
This is not a TP/PP engine or an eight-GPU serving-capacity measurement.
The paused native-vLLM Loop PP project remains paused.

The primary question is incremental Joint value beyond equally informed C-only, G-only, and native Mixed controls.
A larger action space or a binary formulation is not sufficient evidence.
A budget is not a next-token deadline guarantee.
The policy ID `G1` means the adaptive G-only baseline. It does not mean fixed group size G=1.
Every policy still performs the model's complete R4 work.

| Contract field | Actual configuration |
| --- | --- |
| Model | Base Ouro-1.4B, pretrained weights, BF16 |
| Recurrent work | 24 physical layers × Static R4, 96 depth/layer KV namespaces |
| Hardware | Eight H100 80-GB devices, physical IDs 0–7, independent single-GPU workers |
| Backend | Torch 2.11.0+cu129, FlashAttention4 4.0.0b30, pinned RLT |
| GPU execution | Native attention/GEMM, CUDA Graphs enabled, static buffers, native event dependencies |
| Cache | last_exited layout with four depth planes, 18,432 blocks × 16 positions |
| Request width | N4 and N8 for the primary serving comparison |
| Token budget | 2,048 batched tokens |
| Candidate menu | C64/128/256/512/768/1024/1536/2048 and legal G1/2/3/4 |
| Cost inputs | Current legal state and frozen historical service lookup, no future-token completion predictor |

The environment and actual UUIDs remain in the lock files.
The 1,099 registered prompt variants are not 1,099 independent measured examples.
The per-experiment source families and repetition counts determine the evidence denominator.

## Completed evidence

| Evidence | Completed scope | Interpretation |
| --- | --- | --- |
| Resource calibration | 64 device/start jobs, 384 measured cohorts | Diagnostic parallelism is not perfectly neutral |
| Initial pretrained numerical probes | 64 executions, 48 reference comparisons | Exact hidden/logit/KV/token agreement on these isolated probes |
| Additional frontier-cycle probes | 32 executions, 16 reference comparisons | Exact agreement for G1→G3 and G1→G2→G1 continuations |
| Fairness ablation | 160 observer cohorts, 480 request pairs | Same output length and executed decode-loop work in every pair |
| New cost collection | 1,056 cohorts, 298,842 quantum records | Readiness-separated stream service, not kernel-only busy time |
| Fixed-budget TRAIN pilot | 32 jobs, 448 natural observer cohorts | Exploratory policy comparison over all eight budgets |
| Quiet FINAL_ONLINE confirmation | 12 completed starts, 168 cohorts, 1,008 request executions | Three process blocks per shape/budget, separate from TRAIN |
| Full-answer quality pilot | 798 completed generations, 64 GSM8K + 50 MATH per arm | Seven policies, preserved prompts and evaluator |

The software suite has 17 passing tests.
Software fixtures are not pretrained quality evidence.
All measured prompt work passed the Full-R4 assertion.
No future token, future arrival, or future actual service duration enters the controller.

## Independent findings

**Physical cost interaction:** confirmed for the controlled P2048 endpoint.
An additive categorical model had 4.937-ms leave-one-process-out MAE.
The interaction model had 1.594-ms MAE.
The result is conditional, not a universal cost law or serving gain.
P8192 estimates have much larger variation.

**Cost prediction:** usable only as an experimental occupancy estimate.
Blind Atomic MAE was 9.613 ms and upper exceedance was 5.786%.
Blind Staged Core MAE was 2.769 ms and upper exceedance was 3.862%.
The process-block intervals do not establish a universal 5% bound.
The controller does not claim deadline safety.

**Fairness:** an independent mechanism with its own trade-offs.
It reduced some P99 stalls without a consistent improvement in TTFT or P95.
All 480 paired requests matched output length and executed decode-loop work.
Token sequences matched in 426/480 pairs.
Task accuracy cannot be inferred from that parity count.

**Joint incremental value:** not established by TRAIN or quiet confirmation.
Joint improves some strongly constrained C-only operating points.
Those comparisons include C-only infeasibility and smallest-progress fallbacks.
Tuned G-only often remains close, and native Mixed has much lower P99 in many P8177/N8 conditions.
The quiet confirmation found the same actual G1/J1 per-request Chunk/Loop plans in 144/144 pairs.
All 144 pairs also matched output tokens and output length exactly.
Their total output counts were identical: 16,623 tokens per arm.
Thus output-length or quality differences do not explain their matched throughput.

| Prompt/N | Paired B, ms | C1A output tokens/s | G1 output tokens/s | J1 output tokens/s | M1 output tokens/s | J1/G1 throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2033/4 | 40 | 44.61 | 78.72 | 78.49 | 75.02 | 0.997× |
| 2033/4 | 120 | 75.37 | 75.64 | 75.88 | 74.09 | 1.003× |
| 8177/8 | 40 | 36.23 | 81.85 | 81.54 | 80.02 | 0.996× |
| 8177/8 | 120 | 80.42 | 82.17 | 82.25 | 79.57 | 1.001× |

These are measured clean finite-cohort output rates, not sustainable serving capacity.
M1 is an unconstrained native control. Its B label identifies the paired stratum only.
Each point has three independent process blocks and two source cohorts per block.
`main_paired_pooled_CI.csv` retains paired block intervals for the pooled metrics.

At P8177/N8/B120, J1 pooled ITL P99 was 172.19 ms, versus M1's 59.32 ms.
J1 TTFT P95 was lower: 1,062.16 ms versus 1,372.74 ms.
That is a trade-off, not a strict improvement over Mixed.
G-only already realizes the same per-request execution plan as Joint.
The extra C search supplies no new executed operating point in these confirmation cells.
Across all 16 TRAIN shape/budget comparisons, the largest Joint/G-only mean-cohort throughput ratio was 1.0289×.
No TRAIN point reached the registered 5% throughput screening target against G-only.

**Capacity and quality:** not inferred from structural accounting or finite-cohort throughput.
The sustainable-rate contract requires 1,000 offered requests per arm/rate and three independent starts.
The Poisson/burst capacity sweep is NOT_RUN.
J1 scored 47/64 on GSM8K and 32/50 on extracted MATH answers.
Its completed-correct MATH result was 30/50, with six capped outputs.
J1 and G1 had identical correctness labels on all 114 questions.
They matched token sequences on 113/114 questions.
C1A/C1S lost 3.125/6.25 GSM8K percentage points versus stock Atomic.
The C1S paired interval excluded zero in this pilot.
Thus no blanket quality-equivalence PASS applies to every configuration.
The serving evidence does not justify an H2/H3 solver, live feedback extension, or a full capacity sweep.

## Direct answers to the final questions

1. **Are C and G independent?** No, not in the controlled P2048 cost surface. The measured interaction contradicts additive separability. P8192 variability limits generalization.
2. **Does a smaller B improve ITL/SLO?** Not monotonically. Feasibility, fallback, repeated yields, complete-token fairness, and full-prompt completion all matter. B is not an ITL guarantee.
3. **Does Joint beat equal-budget C-only?** Some constrained points improve greatly. Those points include C-only infeasibility and fallback. Tuned G-only selects the same 144 request plans and reproduces all 144 outputs in quiet confirmation.
4. **Does binary optimization add value?** One-action enumeration already solves the finite 0–1 choice. H2/H3 has no measured result and did not satisfy its progression gate.
5. **Does Joint improve practical goodput beyond strong controls?** No verified incremental region emerged beyond G-only. Mixed has a different TTFT/tail trade-off. Sustainable-rate capacity is NOT_RUN, not a modeled speedup.
6. **Are quality and correctness maintained?** Isolated hidden/logit/KV probes and Full-R4 accounting passed. Quality is configuration-sensitive. J1/G1 matched answer correctness in the pilot, while C-only GSM8K regressions require caution. Universal batched parity and full-benchmark equivalence remain unproved.

## Research decision

Do not build a more complex Joint solver or a new serving engine from this evidence.
Retain tuned G-only and stock/fair Mixed as strong controls.
Further Joint work needs a verified state where changing C adds value beyond those controls.
The C-only quality sensitivity also needs a separate fidelity diagnosis before a quality-matched performance claim.
This study does not resume the paused native-vLLM Loop PP project.

## How to read the results

`predictor.md` and `cost_interaction.md` describe measured component evidence.
`fairness.md` separates fairness from optimizer effects.
`pilot_paired_effects.csv` keeps TRAIN_OBSERVER and QUIET_FINAL blocks separate.
`main_table.csv` contains quiet pooled request/token metrics.
`quality.md` reports actual answer counts, caps, and paired uncertainty.
`serving_capacity.md` records the missing qualification explicitly.

Large relative gains against a constrained fallback are not the main research claim.
The final comparison must retain staged-G4 C-only, tuned G-only, stock Mixed, and fair-Mixed.
No physical stream-service sum is converted directly into an E2E speedup.
