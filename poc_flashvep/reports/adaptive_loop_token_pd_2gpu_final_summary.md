# Two-GPU Dynamic Loop-versus-Token PD: Final Summary

Campaign start: 2026-10-30T17:37:00+00:00.
Report generation elapsed time: 10.267 hours.
This timestamp precedes the final Git push.
Primary cells accepted as complete: 56/56.
Primary cells incomplete or rejected: 0/56.
Branch: `flashvep/adaptive-loop-token-pd-2gpu-w1-w4`.

## Decisions

| Model | Verdict |
|---|---|
| Ouro R4 | CONDITIONAL |
| Huginn R32 | CONDITIONAL |

CONDITIONAL means the evidence does not establish a stable-capacity advantage.
NO_GO_PERFORMANCE applies only to the implemented policy and completed conditions.
It does not prove that every dynamic partitioning policy must fail.
BLOCKED_CORRECTNESS prevents a valid complete seven-arm comparison for the affected model.
Unfinished configurations remain separate from measured negative results.

## What was implemented

The runtime uses actual two-GPU P/D ownership changes.
Token migration transfers full-depth prefix KV before D executes the suffix.
Loop migration transfers valid depth-specific KV and complete prompt carrier rows.
Huginn Token migration also transfers unused canonical initialization states.
The online methods select provisional boundaries from observed runtime state.
They recheck D admission at the export frontier and can finish Prefill on P.
All subsequent Decode tokens execute full fixed depth on D.
DP and DP_Chunked use two independent colocated replicas.
Shared-GEMM Prefill/Decode Mixed co-batching is off in every main arm.

## Correctness and quality limits

Both models passed the initial seven-arm 32-request migration qualification.
There were 192/192 non-reference output and Prompt-state comparisons per model.
The fixture contains repeated prompt content, not 192 distinct benchmark questions.
Zero/full endpoints and complete long-prompt Loop/Token migration also passed finite checks.
Huginn's additional normalized Token grid has separate qualification evidence.
The main serving harness rejects unexplained controlled-output mismatches immediately.
Rejected cells do not become valid performance results.
Original-HF equivalence and GSM8K/MATH accuracy are NOT_ESTABLISHED in this campaign.

## Evidence and primary-metric limits

Main runs generate real greedy outputs with declared controlled targets.
They use ignore_eos and are not Natural-EOS results or frozen-token replay.
W1 targets 384/448/512 tokens in 25/50/25 percent proportions.
Huginn's prompt plus output target never exceeds 4,096 tokens.
The primary SLO combines TTFT and request-average TPOT.
It is not a request-level maximum-ITL SLO.
The SLO thresholds were frozen from disjoint PD_G1 calibration before candidate serving.
All failures and missing requests remain in offered-request denominators.

Each primary point has one 32-request or 24-request start unless the frozen reduced matrix applies.
Secondary W1 trials use the predeclared 1.5-rate point and independent server starts.
All secondary cells use the separately declared 16-request budget uniformly across target methods and models.
The mandatory primary matrix keeps its original request counts.
Base-rate and 1.5-rate trials are not pooled for confidence intervals.
Their cohort sizes and prompt subsets differ. Rate-point figures therefore show markers without an interpolated capacity curve.
The exact completed start counts appear in the paired comparison table.
Finite-cohort goodput includes final drain time.
The displayed arrival rate is the normal Poisson intensity. W3 uses five times that intensity during its long-prompt burst.
The raw metric table also records the realized scheduled rate over the finite arrival span.
Finite-cohort goodput can exceed the normal intensity without implying sustainable capacity.
Maximum indefinitely sustainable serving capacity is NOT_ESTABLISHED.
A passing finite rate grid point is not proof of stationary queue stability.
For reference, 32/32 successes give a Wilson 95% lower bound of 89.28%.
Even perfect attainment in that cohort does not confirm 90% population attainment.
Wilson intervals are descriptive binomial intervals. Queue coupling can violate their independence assumption.

## Transport and novelty controls

PD_G1 uses native completed-token-chunk streaming.
Loop and Token methods use acknowledged snapshot handoff.
Separate full-cut W1 snapshot controls isolate that protocol change where completed.
Protocol-only gains are not Loop-partitioning gains.
The two axes use matched normalized candidate fractions before token-chunk alignment.
The online rule is conservative and is not an exhaustive optimizer.
It does not dynamically select the D-local Cut0 endpoint.
Generic logical-layer equivalence remains an inherited novelty limitation.
Layer Split and 4/8GPU pooled serving are outside this campaign.
Loop-versus-Token performance alone cannot prove Loop-exclusive novelty.

## Timing and resource limits

Clean runs use one physical GPU pair at a time.
All methods in one workload/model use the same pair.
Other managed GPU experiments and burners remain absent from that clean lane.
This is not a claim of complete CPU-host exclusivity.
The resource manager restores approved burners after measurement.
The report preserves all five failed development namespaces and rejected prototypes.

`e2e_component_shares_aggregate.csv` contains exclusive host-observed request-time percentages.
Its denominator is summed request E2E, not the cohort critical path.
Those percentages are not exclusive GPU attention, GEMM, or DMA shares.
NIXL cumulative completion observations are not pure DMA time.
Overlapping device durations are never added to claim an E2E speedup.
Optional profiler results remain separate from clean latency.

## Artifact index

- [Main matrix](adaptive_loop_token_pd_2gpu_w1_w4_matrix.md)
- [Correctness and protocol](adaptive_loop_token_pd_2gpu_correctness_and_protocol.md)
- [Online policy and SLO](adaptive_loop_token_pd_2gpu_online_policy_and_slo.md)
- [All measured run statistics](adaptive_loop_token_pd_2gpu/per_run_metrics.csv)
- [Mandatory executed and missing cells](adaptive_loop_token_pd_2gpu/experiment_matrix.csv)
- [Paired start uncertainty](adaptive_loop_token_pd_2gpu/paired_start_comparisons.csv)
- [Host E2E percentages](adaptive_loop_token_pd_2gpu/e2e_component_shares_aggregate.csv)
- [Communication payloads](adaptive_loop_token_pd_2gpu/communication_payloads.csv)
- [Matched full-cut transport controls](adaptive_loop_token_pd_2gpu/protocol_comparisons.csv)
- [Actual online cuts](adaptive_loop_token_pd_2gpu/online_cut_outcomes.csv)
- [Frozen SLO calibration](adaptive_loop_token_pd_2gpu/slo_calibration.json)
- [Complete aggregated requests, lossless gzip](adaptive_loop_token_pd_2gpu/per_request.jsonl.gz)

Raw requests preserve input IDs, output IDs, exit depths, stop reasons, and token timestamps.
Every job preserves configuration, source archives, GPU mapping, and complete logs.
The uncompressed aggregate remains local. Its complete gzip archive is published instead of an oversized Git object.
Each individual run also retains its plain request JSONL file.
`commands.jsonl` contains the actual GPU launch commands.
`controller_commands.jsonl` contains the phase commands.
Model and source pins are in `configs/initial_contract.json` and `configs/frozen_execution_hashes.json`.
The final Git SHA is the verified branch tip, not an assumed clean-worktree source identity.

## Reproduction

Use `/home/esjung/.venvs/looped-pd-abc/bin/python`.
The engine path is `/home/esjung/external/looped-pd-abc-rlt`.
The repository path is `/home/esjung/MLLM-EP-github`.
Each recorded GPU launch includes its actual configuration and environment.
The corresponding source archive is in that run's raw directory.

Run the CPU contract and report tests before a new authorized GPU reproduction.
Use the following exact environment for those tests.

```bash
PYTHONPATH=/home/esjung/external/looped-pd-abc-rlt:/home/esjung/MLLM-EP-github /home/esjung/.venvs/looped-pd-abc/bin/python -m unittest poc_flashvep.adaptive_loop_token_pd_2gpu.test_contract poc_flashvep.adaptive_loop_token_pd_2gpu.test_reporting
```

Run `python -m poc_flashvep.adaptive_loop_token_pd_2gpu.analyze` to rebuild derived data from preserved raw files.
Run `python -m poc_flashvep.adaptive_loop_token_pd_2gpu.resources status` before any GPU lease.
Use `configure` to clone a completed configuration under a new run name.
Never reuse a completed or failed raw namespace.
Replay its recorded worker command and environment only after ownership checks and isolated GPU assignment.
The original dispatcher deliberately refuses new jobs after this campaign deadline.
Do not reset that deadline to extend this campaign.
A future authorized reproduction needs a separate time budget.

## Exact workload token lengths

The following values count frozen tokenizer IDs, not predicted generation lengths.
Controlled output targets are not Natural-EOS measurements. Completion counts remain separate in the main matrix.

| Model | Workload | Offered N | Input min | Input mean | Input p50 | Input p95 | Input max | Output target mean | Maximum prompt+target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | W1 | 32 | 961 | 1902.00 | 1670.00 | 3329.00 | 3470 | 448.00 | 3982 |
| ouro | W2 | 24 | 2071 | 3043.96 | 3168.00 | 3462.05 | 3472 | 192.00 | 3728 |
| ouro | W3 | 32 | 868 | 1858.91 | 1662.00 | 3329.00 | 3470 | 384.00 | 3801 |
| ouro | W4 | 24 | 815 | 1084.04 | 1030.00 | 1418.20 | 1446 | 453.33 | 1958 |
| huginn | W1 | 32 | 968 | 1872.38 | 1623.50 | 3294.45 | 3374 | 448.00 | 3806 |
| huginn | W2 | 24 | 2014 | 3004.92 | 3096.50 | 3371.90 | 3420 | 192.00 | 3676 |
| huginn | W3 | 32 | 863 | 1832.94 | 1621.00 | 3294.45 | 3374 | 384.00 | 3758 |
| huginn | W4 | 24 | 863 | 1094.12 | 1066.50 | 1403.85 | 1440 | 453.33 | 1952 |

## Calibration-selected partitions

| Model | Workload | Loop cut | Token P-prefix fraction |
| --- | --- | --- | --- |
| ouro | W1 | 4 | 1.0 |
| ouro | W2 | 0 | 1.0 |
| ouro | W3 | 4 | 0.25 |
| ouro | W4 | 4 | 1.0 |
| huginn | W1 | 32 | 0.75 |
| huginn | W2 | 32 | 1.0 |
| huginn | W3 | 24 | 1.0 |
| huginn | W4 | 8 | 0.75 |

These are limited calibration choices, not proven global optima for the main workloads.
Changing workload also changes prompt and output lengths. It does not isolate queue state alone.

## Answers to the serving questions

Q1: The table above records workload-dependent calibration choices. The full main cut map is not measured.

Q2 and Q3: The following ratios compare actual online Loop decisions with selected Static and Token baselines.
| Model | Workload | Baseline | p95 E2E speedup | SLO-goodput ratio |
| --- | --- | --- | --- | --- |
| ouro | W1 | Loop_Static | 0.8848 | 1.0030 |
| ouro | W1 | Token_Dynamic | 0.9177 | 0.9996 |
| ouro | W2 | Loop_Static | 2.5009 | 1.9211 |
| ouro | W2 | Token_Dynamic | 0.9271 | 0.9900 |
| ouro | W3 | Loop_Static | 0.7701 | 0.7836 |
| ouro | W3 | Token_Dynamic | 0.7043 | 0.6138 |
| ouro | W4 | Loop_Static | 0.8969 | 0.9465 |
| ouro | W4 | Token_Dynamic | 0.9943 | 1.0241 |
| huginn | W1 | Loop_Static | 0.9613 | 1.0000 |
| huginn | W1 | Token_Dynamic | 0.9815 | 1.0005 |
| huginn | W2 | Loop_Static | 0.8807 | 0.9905 |
| huginn | W2 | Token_Dynamic | 0.8746 | 0.9878 |
| huginn | W3 | Loop_Static | 1.0794 | 0.9341 |
| huginn | W3 | Token_Dynamic | 1.0997 | 0.9422 |
| huginn | W4 | Loop_Static | 1.2606 | 1.0577 |
| huginn | W4 | Token_Dynamic | 1.1689 | 1.0476 |

These primary ratios have one independent start. The equal-rate repeated table provides separate paired uncertainty.
A ratio above one does not by itself establish maximum stable capacity.

The next table checks the actual partitions in the repeated W1 trials.

| Model | Arm | Recorded N | Full-Prefill on P | D continuations |
| --- | --- | --- | --- | --- |
| ouro | Loop_Dynamic | 48 | 48 | 0 |
| ouro | Token_Dynamic | 48 | 48 | 0 |
| huginn | Loop_Dynamic | 48 | 48 | 0 |
| huginn | Token_Dynamic | 48 | 48 | 0 |

A timing difference without a different partition plan is not evidence of adaptive cut selection.
If the selected Static cut is also full depth, small timing gains do not demonstrate recurrent work redistribution.

Q4: Loop and Token arms use the same configured token chunk size. Token Split is not forced to use smaller chunks.
Exclusive clean GEMM efficiency attribution is NOT_MEASURED. Optional profiles are separate diagnostic evidence.

Q5: communication_payloads.csv records actual state payloads and NIXL completion counts.
It includes the additional Huginn suffix initialization state. Payload reduction is not assumed to be an E2E speedup.

Q6: online_cut_outcomes.csv separates initial Full-PD choices, accepted continuations, and source-side fallbacks.
If every request uses Full-PD, any native-streaming difference is not evidence of adaptive partitioning.

Q7: The table below checks observed Decode occupancy and full-request KV reservation credit.
The credit calculation uses the runtime block formula and the declared controlled output budget.
It is structural accounting from observed reservations, not GPU kernel-time attribution.

| Model | Workload | Completed decision N | Mean observed active Decode | Insufficient D credit N | Insufficient D credit % |
| --- | --- | --- | --- | --- | --- |
| ouro | W1 | 32 | 2.56 | 0 | 0.00 |
| ouro | W2 | 24 | 1.96 | 0 | 0.00 |
| ouro | W3 | 32 | 3.31 | 14 | 43.75 |
| ouro | W4 | 24 | 3.38 | 0 | 0.00 |
| huginn | W1 | 32 | 2.28 | 12 | 37.50 |
| huginn | W2 | 24 | 1.54 | 18 | 75.00 |
| huginn | W3 | 32 | 3.06 | 22 | 68.75 |
| huginn | W4 | 24 | 3.46 | 7 | 29.17 |

Every cut still reserves D KV for full-depth Decode. Moving Prefill work does not remove this reservation requirement.
Ouro and Huginn also differ in width, recurrent structure, boundary state, and supported async execution.
This experiment does not isolate one of these differences as the sole measured cause.
