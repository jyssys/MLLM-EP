# Huginn recovery and FluidToken-style versus recurrent split

## Contract and budget

This campaign starts on 2026-11-03 at 23:18:43 KST. The hard deadline is 2026-11-04 at 10:18:43 KST.
Stop new long GPU jobs after 08:48:43. Stop GPU experiments by 09:18:43. Reserve the final hour for analysis, publication, and resource restoration.
The maximum eleven hours include implementation, experiments, figures, and ordinary Git push.
Prior evidence ends at commit `ef3efe173ae48c00925ec616bdb5cc4afddf86a2`.
The publication branch is `flashvep/huginn-recovery-fluidtoken-loop-split`.
The active worktree, original staged index, external runtime, and prior artifacts remain unchanged.
New code resides in `poc_flashvep/fluidtoken_loop_split/`.
New evidence resides in `reports/looped_pd_fluidtoken_vs_dynamic_loop_split/`.

## Models and resources

Use ByteDance/Ouro-1.4B, BF16, fixed R4, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Use tomg-group-umd/huginn-0125, BF16, fixed R32, revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
Retain the qualified numerical fixes, original weights, attention, and all recurrent work.
Huginn input plus output cannot exceed 4096 tokens.
Use isolated 1P1D pairs among physical GPUs 0 through 7. Record UUIDs, topology, and source hashes.
Stop only verified owned burners. Never terminate unrelated GPU processes.
Clean timing uses the existing exclusive quiet lane. Restore approved burners after measurements.

## Prior evidence and first gate

Read the autonomous, KV-recovery, and atomic-PD final reports completely before execution.
Ouro has finite same-window operational evidence. This is not a measured Dynamic A speedup or a physical FLOP ceiling.
Huginn has clean D-local work tolerance, but no qualified sustained same-window combined result.
Huginn recovery has priority over method comparisons.
Do not expand Ouro-only experiments into a large method campaign while the Huginn gate remains unresolved.
Small Ouro correctness controls are permitted.

## Phase A: reproduce and recover Huginn

Start with the recorded L2048/O16 profile failure. Preserve its actual arrival sequence and geometry for reproduction.
The bounded runner must detect response progress, retain stacks, and terminate only its registered children after the existing watchdog limit.
A terminal process alone does not prove completion. Require every offered request, exact outputs, and full-depth accounting.
Minimize capture duration and cohort only after the failure reproduces.
Test one variable per diagnostic. Candidate axes include collector activity, graph replay, transport submission, buffer pressure, and capture boundaries.
Record each hypothesis, prediction, intervention, result, and unresolved alternative.
Do not treat a Python stack inside NIXL as proof of its native root cause.
Do not increase timeout as the proposed fix.
If CUPTI remains perturbing, evaluate lower-overhead instrumentation against actual kernel traces. Event spans alone are not kernel union.

Use L2048/O16 first. Permitted follow-ups include 2560/16, 3072/16, 2048/32, and 2560/32.
Begin rates in 0.5–2.5 requests/s, guided by finite independent P/D measurements.
Adjust only supported slots, pools, token budgets, chunk budgets, and graph geometry.
Do not deliberately reduce P efficiency to create backlog.

The joint gate requires all of these conditions in one common interval:

1. D-clear P-ready pressure increases, with at least two ready requests.
2. P kernel union is at least 80%.
3. P service reaches at least 85% of the strongest matched finite reference.
4. Matched profiling response-wall perturbation is at most 5%.
5. D admission has legal credit and no binding long-request reservation block in the claimed interval.
6. D TPOT p95 remains below the frozen 90% margin, and ITL p99 meets its frozen limit.
7. Real, exact D-local Prefill completes inside the interval.
8. Output, KV, hidden state, and complete R32 work checks pass.

Retain event-only and host-span queue sensitivity results. Do not combine metrics from different starts.
For L2048, retain TPOT limit 186.571506 ms, 90% margin 167.914355 ms, and ITL p99 limit 192.821254 ms.
For L2560, retain TPOT limit 191.515103 ms, 90% margin 172.363593 ms, and ITL p99 limit 198.378139 ms.
New lengths require disjoint calibration before held-out evaluation. Never tune thresholds after candidate results.
Target three independent starts with at least 128 requests each. Report missing repetitions explicitly.
Maximum ITL remains separate from the p99 contract.

## Phase B: prior work and reuse audit

Read [FluidPD](https://arxiv.org/abs/2610.06917), including pressure equations, suffix budget, and decision frequency.
Check official code availability, revision, license, and APIs before reuse.
Without the original implementation, label the baseline `FluidToken-style reproduction`.
Also inspect Load-Aware Prefill Deflection, DuetServe, and chunked Prefill.
Record differences in engine, scheduling, transfer, predictor calibration, and admission policy.
Treat `ref_loopserve` as read-only. Audit its graph, allocator, scheduling, and profiling facilities before considering reuse.
Reuse existing Token and Loop connectors after verifying their actual execution and state contracts.

## Phase C: split correctness and serving

Compare Static PD, Static Token, Dynamic Token, Static Loop, and Dynamic Loop.
Use Token prefix ratios near 25%, 50%, 75%, and 90%, with equal tuning budgets.
Do not constrain Token Split to a small chunk while allowing Loop Split a large chunk.
Ouro loop cuts include 1, 2, 3, and full 4. Huginn begins with 8, 16, 24, 28, and full 32.
Also test boundaries zero and full as correctness controls where supported.
The destination performs all remaining Prefill and then full-depth Decode.
No probe request can substitute for a migrated serving request.

First run 8-request state tests, then at least 32 independent requests with changed order or arrival seed.
Check exact sequence output against the same native reference, transferred bytes, hidden state, valid KV planes, and recurrent rows.
Verify Huginn Prelude, Coda, reinjection state, first-token sampling, and buffer lifetime.
Block performance expansion after any unexplained mismatch.
Keep original-HF fidelity and downstream accuracy separate and unmeasured unless actually evaluated.

Use identical source, graph warmup, dataset, arrival trace, GPU budget, precision, and SLOs across each comparison.
Separate native streaming from snapshot handoff with an explicit same-cut transport control.
Hold co-batching policy constant across methods. Do not claim Native Mixed as a new contribution.
Dynamic decisions use only observed queue, active batch, KV credit, latency slack, and calibrated service/transfer costs.
Recheck admission at handoff. Continue safely on P when D cannot accept work.
Record every decision and fallback reason. No future arrival, output length, or correctness label enters an online policy.

Ouro starts at L4096/O32, nominal 6.356 QPS, P/D slots 2/8, with the prior fixed SLOs.
Huginn uses the independently validated Phase A condition.
Prefer 128-request paired starts and three repetitions for final candidates. A 512-request persistence check is optional.
Use existing tokenizer-specific ShareGPT/LongBench controlled prefixes and fixed output caps.
Natural-EOS validation is optional and must have a separate label.

## Phase D: analysis and claim limits

Primary outcomes are measured joint SLO-goodput, TTFT, E2E, and Decode protection.
Report p50/p95/p99 TTFT, TPOT, ITL, E2E, maximum ITL, throughput, and all offered/completed/failed counts.
Record actual offloaded requests, tokens, loops, recurrent rows, kernel counts, and measured GPU time.
Label FLOP accounting as structural unless directly measured. Do not convert work reduction into speedup.
Report KV/hidden bytes, fragments, source retention, exposed handoff, synchronization, D admission, and HBM.
Host E2E percentages and overlapping device durations remain separate.
Use start-level paired uncertainty. Tokens are not independent experimental repetitions.
Finite overloaded cohorts do not establish sustainable capacity.
Equal logical boundaries can make Generic Layer Split and Loop Split equivalent. Record `EQUIVALENT_MECHANISM` when applicable.
Loop faster than Token does not by itself establish Loop-specific novelty.

Required figures cover motivation, same-run P/D evidence, four-way methods, split granularity, and supported mechanism differences.
Never draw unmeasured idle intervals or fill missing metrics with estimates.
If the method gate fails, publish explicit blocked panels or omission records instead of invented comparisons.

## Gates and final artifacts

Use `PASS`, `CONDITIONAL`, `BLOCKED_CORRECTNESS`, `ROOT_CAUSE_INCONCLUSIVE`, `BLOCKED_BY_HUGINN_GATE`, and `NOT_RUN_TIME_BUDGET` as appropriate.
Use `NO_CONFIRMED_LOOP_ADVANTAGE` if the evidence does not establish a benefit.
Preserve all failed namespaces and raw logs. Every run records its configuration and source hash.
Write `looped_pd_fluidtoken_vs_dynamic_loop_split_final_summary.md` with model-specific verdicts and all requested comparison fields.
Publish only related files using an isolated index or worktree. Use ordinary pushes and verify the remote SHA.
Record final elapsed time, incomplete work, burner ownership, and resource restoration.
