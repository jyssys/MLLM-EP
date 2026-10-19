# Completed baseline robustness extension

## Result and scope

All four frozen arms complete their 64 requests.
The integrity validator admits all 256 raw records.
Each model and mode has 32 GSM8K requests and 32 MATH-500 requests.
These are additional baseline observations, not a candidate's quality confirmation.

The [summary](../analysis/baseline_extension_complete_v1/summary.json) retains every score and source hash.
The [integrity record](../artifacts/baseline_extension_integrity_complete_v1.json) checks the prompt, output text, stopping rule, and launch provenance.
The validator evaluates no answer correctness.
The scoring analyzer separately checks the recorded depth events and native counters.

## Frozen execution

The models are Base Ouro-1.4B and Base Ouro-2.6B.
Their pinned revisions remain unchanged.
Both arms use BF16, greedy natural generation, an 8192-token cap, and the frozen prompt and stop rules.
EOS can terminate generation.

The selected CDB reference uses graph-enabled B1 execution.
Fixed execution uses R4.
Adaptive execution uses the original checkpoint gate, q=0.5, minimum R2, maximum R4, and `last_exited` KV.
Both modes execute full-R4 prefill.
No trained lookahead or preloop gate participates.

The four arms use physical GPUs 4, 5, 6, and 7 separately.
They are independent single-GPU baseline jobs, not physical adaptive TP4.
Their successful outer guards retain cleanup and burner-restoration records.
The [launch contract](../artifacts/baseline_extension_launch_contract_v1.json) records the model, protocol, and runner hashes.

The new request IDs exclude the original calibration, original held-out, previous pilot, and protected-factorial metadata IDs.
The original candidate held-out cohort remains untouched.
No result is pooled with the earlier eight-request calibration cohort.
The [post-launch integrity supplement](baseline_extension_postlaunch_integrity_contract_v1.md) remains explicitly post-launch.
It changes no grading or generation rule.

## Natural-generation quality

All confidence intervals below use independent requests and 4000 bootstrap resamples.
Accuracy excludes incomplete length-capped responses under the frozen rule.
Parsing failures and truncation remain visible.
They need not identify disjoint requests.

| Base model | Mode | Domain | Correct/N | Accuracy | 95% CI | Parse failures | Length-capped/N |
|---|---|---|---:|---:|---|---:|---:|
| 1.4B | Fixed R4 | GSM8K | 24/32 | 75.000% | [59.375, 90.625]% | 1 | 0/32 |
| 1.4B | Adaptive | GSM8K | 24/32 | 75.000% | [59.375, 90.625]% | 2 | 0/32 |
| 1.4B | Fixed R4 | MATH-500 | 17/32 | 53.125% | [37.500, 68.750]% | 1 | 4/32 |
| 1.4B | Adaptive | MATH-500 | 21/32 | 65.625% | [50.000, 81.250]% | 3 | 4/32 |
| 2.6B | Fixed R4 | GSM8K | 28/32 | 87.500% | [75.000, 96.875]% | 0 | 0/32 |
| 2.6B | Adaptive | GSM8K | 28/32 | 87.500% | [75.000, 96.875]% | 0 | 0/32 |
| 2.6B | Fixed R4 | MATH-500 | 25/32 | 78.125% | [62.500, 90.625]% | 5 | 5/32 |
| 2.6B | Adaptive | MATH-500 | 24/32 | 75.000% | [59.375, 87.500]% | 4 | 6/32 |

These sample accuracies do not reproduce a full-benchmark score.
They do not establish the 1 pp GSM8K or 2 pp MATH noninferiority margins.
They do not establish quality preservation for an unimplemented architecture.

## Paired original operating points

The denominator is 32 shared request IDs per model and domain.
The difference is adaptive accuracy minus fixed accuracy.
It describes existing operating points, not a quality change caused by a new mechanism.

| Base model | Domain | Difference | Paired bootstrap 95% CI |
|---|---|---:|---|
| 1.4B | GSM8K | 0.000 pp | [-9.375, 9.375] pp |
| 1.4B | MATH-500 | +12.500 pp | [3.125, 25.000] pp |
| 2.6B | GSM8K | 0.000 pp | [-9.375, 9.375] pp |
| 2.6B | MATH-500 | -3.125 pp | [-9.375, 0.000] pp |

The [output comparison](../analysis/baseline_output_comparison_complete_v1/summary.json) retains the first divergent token and finish reasons.
Fixed and adaptive output IDs match for 10/32 and 15/32 GSM8K requests for Base 1.4B and Base 2.6B, respectively.
No MATH request has identical complete output IDs between modes.
No new logit or KV dump accompanies this extension.
Different free-running trajectories prevent causal token-by-token work comparisons after divergence.

## Actual adaptive execution

The recorded depth events match the native counters in every admitted request.
Recurrent graph hits plus captures equal the recorded executed core passes.
This confirms bounded original-gate execution and actual skipped passes in the selected reference.
It does not validate distributed adaptive inference or another runtime's gate arithmetic.

| Base model | Domain | Token-weighted mean decode depth | Request-weighted mean decode depth |
|---|---|---:|---:|
| 1.4B | GSM8K | 3.059609 | 3.056707 |
| 1.4B | MATH-500 | 2.809121 | 2.955099 |
| 2.6B | GSM8K | 3.024811 | 3.019958 |
| 2.6B | MATH-500 | 2.929966 | 2.932353 |

Depth accounting excludes the first output token produced by prefill.
Recorded core-pass work excludes prefill, gates, the LM head, and other non-core costs.
It is not a FLOP count or a latency estimate.
Fewer passes do not imply fewer allocated depth-indexed KV slots.

## Consequence for discovery

The extension strengthens the both-size, both-mode baseline evidence.
It does not create an independent ready frontier or demonstrate a new distributed architecture.
Candidate E2E latency and candidate free-running quality remain NOT_RUN.
The [current Tables C/D](../analysis/result_tables_cd_complete_v1/tables_cd.md) keep quality and controlled-work hardware panels separate.
No natural-generation wall ratio becomes a scheduling speedup when generated work differs.

## Reproduction

Use the unchanged frozen runner and guarded commands from the launch contract.
Use fresh output paths for any new execution.
Run the integrity validator before the scoring analyzer.
Then run `baseline_output_comparison` and `build_result_tables_cd` on the admitted summary.
The stored source hashes identify the exact versions used here.
