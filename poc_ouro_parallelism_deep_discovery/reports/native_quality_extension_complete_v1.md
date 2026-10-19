# Completed native TP4 quality extension

## Result and evidence class

Both fresh-cache v2 arms complete 64 requests.
The integrity validator admits all 128 records.
Each model has 32 GSM8K and 32 MATH-500 outputs.
Both sizes use the same 64 distinct problem IDs.
These are existing-baseline observations, not candidate quality or fresh held-out confirmation.

The [complete integrity receipt](../artifacts/native_quality_integrity_complete_v1.json) checks launch and raw-record provenance.
The [scored summary](../analysis/native_quality_extension_complete_v1/summary.json) retains every result and comparison.
The original candidate held-out remains untouched.
The evidence class is QUALITY_ONLY.
No native quality-run wall time enters a matched-work architecture speedup.

## Frozen execution

Both models use their unchanged pinned Base checkpoints, BF16, and fixed R4.
The physical TP4 group uses GPUs4/5/6/7.
Each rank owns four Q heads and four KV heads.
The loaded model has 24 or 48 physical layers and 96 or 192 logical depth/layer KV slots.
Real pretrained weights and TP collectives execute.
The models are dense, so there is no MoE EP Dispatch/Combine All-to-All.

The native path uses async scheduling, graph capture sizes 1/2/4/8, and sequential B1 natural generation.
It disables prefix caching, chunked prefill, and EP.
Each rank reserves 20 GiB KV.
The maximum runtime length is 10240.
Greedy generation permits EOS and retains the frozen 8192-token cap and domain stop strings.
No sampling, prompt, cap, grading rule, model, or policy changes follow from an outcome.

The [v2 source manifest](../artifacts/native_quality_extension_contract_v2.json) freezes 14 named sources before launch.
Each runner checks their hashes before native imports.
Each launch uses fresh job-local vLLM, Inductor, and Triton caches.
The runner rejects inherited model-control variables.
It uses the read-only ownership extension, not the earlier module with import-time model controls.
These checks do not supply a complete native binary-build or internal cached-pointer proof.

## Accuracy

The scorer uses the same frozen parser and completed-answer rule as the staged extension.
A length-capped response remains incomplete and contributes zero to correct/N.
Parsing failure and truncation can concern the same request.
Intervals use 4000 request-level bootstrap resamples with the frozen seed.
They do not treat tokens or layers as independent samples.

| Base | Domain | Correct/N | Accuracy | Bootstrap 95% CI | Parse failures | Length-capped/N |
|---|---|---:|---:|---|---:|---:|
| 1.4B | GSM8K | 25/32 | 78.125% | [62.500, 90.625]% | 1 | 0/32 |
| 1.4B | MATH-500 | 16/32 | 50.000% | [34.375, 65.625]% | 1 | 6/32 |
| 2.6B | GSM8K | 28/32 | 87.500% | [75.000, 96.875]% | 0 | 0/32 |
| 2.6B | MATH-500 | 25/32 | 78.125% | [62.500, 90.625]% | 2 | 6/32 |

The domains are small baseline subsets.
They do not reproduce a full-benchmark score or the Thinking checkpoint's official protocol.
They do not establish 1 pp GSM8K or 2 pp MATH noninferiority.
No unimplemented candidate receives an accuracy PASS.

## Native versus staged fixed R4

Each comparison uses 32 identical problem and prompt-token IDs within a model/domain cell.
The difference is native TP4 accuracy minus staged fixed-R4 accuracy.
It is a backend comparison, not a new mechanism's effect.

| Base | Domain | Staged correct/N | Native correct/N | Difference | Paired bootstrap 95% CI | Complete raw output-ID matches |
|---|---|---:|---:|---:|---|---:|
| 1.4B | GSM8K | 24/32 | 25/32 | +3.125 pp | [0.000, 9.375] pp | 20/32 |
| 1.4B | MATH-500 | 17/32 | 16/32 | -3.125 pp | [-15.625, 6.250] pp | 4/32 |
| 2.6B | GSM8K | 28/32 | 28/32 | 0.000 pp | [0.000, 0.000] pp | 25/32 |
| 2.6B | MATH-500 | 25/32 | 25/32 | 0.000 pp | [-9.375, 9.375] pp | 0/32 |

The degenerate GSM8K interval for 2.6B reflects identical correctness bits in this sample.
It does not prove population equivalence or token parity.
Equal final correct counts also do not require the same correct request IDs.
The structured summary retains each paired correctness outcome and first divergent token.

Native string-substring stops and staged token-suffix stops are different backend mechanisms.
The native record preserves canonical decoded output IDs and API text separately.
Matching intended stop strings does not establish identical stopping tokens.
Four matched numerical positions in earlier probes also cannot establish complete natural-output parity.

## Bounded raw-output inspection

Four 1.4B pairs have discordant completed-answer correctness.
They illustrate why timing cannot ignore changed work.
This inspection changes no score or parser rule.

| Request | First raw divergent token, zero-based | Staged / native output tokens | Observed difference |
|---|---:|---:|---|
| `gsm8k/test/901` | 67 | 112 / 140 | Both use the frozen text stop. Their reasoning and final numeric answers differ. |
| `math500/test/141` | 89 | 477 / 649 | Both reach EOS. Their final boxed remainders differ. |
| `math500/test/58` | 186 | 1726 / 187 | Native ends after the requested arithmetic answer. Staged continues into another problem and ends with a different box. |
| `math500/test/90` | 53 | 282 / 8192 | Staged ends with the requested fraction. Native continues into repeated unrelated content and reaches the length cap. |

These observations concern stored text and token IDs.
They do not identify the first unequal numerical operator.
They do not attribute a causal defect to one kernel, graph option, or stopping implementation.
No new logit or KV dump accompanies this extension.

## Shutdown and preserved failures

Both v2 outer guards report success and exit code 0.
Both preserve complete outputs before shutdown.
The native logs still contain forced worker cleanup and leaked-shared-memory warnings.
The 2.6B shutdown also records TCPStore/NCCL monitor warnings.
Thus, cleanup is qualified rather than perfectly graceful inside the engine.
These logs are not clean performance evidence.

The first unisolated native v1 attempt remains separate and NOT_ADMITTED.
It stopped after two partial records for a cache-provenance precaution.
No wrong-output or shared-cache defect was established by that stop.
Its source, logs, partial outputs, and decision record remain preserved.

The guard restores verified owned utilization workers after each released job.
After the 2.6B arm, the four restored workers use physical GPUs4/5/6/7.
No foreign process or protected experiment stops.

## Consequence and reproduction

The extension strengthens both-size native fixed-R4 baseline evidence.
It does not establish adaptive native TP4, numerical backend equivalence, a new architecture, or candidate quality preservation.
The current shortlist remains empty for independent mechanism, dependency, and cost reasons.

Table C2b in the [updated result panels](../analysis/result_tables_cd_complete_v2/tables_cd.md) contains the four native cells.
It remains separate from staged adaptive quality and constant-token hardware latency.
The [reproduction index](reproduce_final_v1.md) supplies guarded launch and archived-analysis commands.
A fresh reproduction needs fresh directories and reviewed selectors.
The frozen original outputs and failed attempts must not be overwritten.
