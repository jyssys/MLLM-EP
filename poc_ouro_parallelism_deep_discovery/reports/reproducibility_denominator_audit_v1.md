# Reproducibility and performance-denominator audit v1

## Result

The closed hardware counts, output denominators, medians, and paired context ratios pass this CPU audit.
One draft device mapping is wrong.
Three reproduction instructions need more precise scope.
One descriptive comparison also needs its explicit comparator and denominator.
The current draft also retains the modeled-evidence wording issues from the preceding review.
No measured candidate result follows from these baseline checks.

Here, S1 means the single-GPU native TP1 control.
P2/P4 mean native TP2/TP4 unless a fixture or external replica layout is explicit.
All reviewed distributed hardware controls use fixed R4.
No result here establishes original-gate adaptive distributed performance.

I read the complete reproduction index, four frozen hardware contract reports, and five hardware protocol JSON files.
Those include the superseded 32 GiB proposal and active 52 GiB cohort protocol.
I inspected runner, worker, timing, selector, and analyzer source.
The draft is byte-identical to the preceding complete-read version.
The [JSON record](../artifacts/reproducibility_denominator_audit_v1.json) pins the source extent and exact hashes.

## 1. Required correction: TP2 uses physical GPUs4/6

The draft's [line 84](final_report_draft_v1.md:84) assigns native TP2 to GPUs4/5.
Both constant-token native TP2 receipts instead record `mask="4,6"`.
Both long-context TP2 receipts also record `mask="4,6"`.

Direct evidence:

- [Base1.4B cohort receipt](../outputs/native_cohort_base14_P2_v1/run_contract.json:14).
- [Base2.6B cohort receipt](../outputs/native_cohort_base26_P2_v1/run_contract.json:14).
- [Base1.4B context receipt](../outputs/native_longctx_base14_P2_v3/run_contract.json:12).
- [Base2.6B context receipt](../outputs/native_longctx_base26_P2_v3/run_contract.json:12).

This also supersedes R1's proposed cohort mapping in the [preceding review](final_report_independent_review_v1.md:23).
That proposal incorrectly retained GPUs4/5 for the constant-token cohort.
The prior report remains unchanged.

| Control | Actual physical ownership | Reserved KV |
|---|---|---|
| Cohort TP1 | GPU4 | 52 GiB total |
| Cohort TP2 | GPUs4/6 | 52 GiB/rank, 104 GiB total |
| Cohort TP4 | GPUs4/5/6/7 | 52 GiB/rank, 208 GiB total |
| Cohort DP4 | One TP1 replica on each of GPUs4/5/6/7 | 52 GiB/replica, 208 GiB total |
| Cohort TP2×DP2 | TP2 groups4/5 and6/7 | 52 GiB/rank, 208 GiB total |
| Context TP1/2/4 | GPU4, GPUs4/6, or GPUs4/5/6/7 | 52 GiB/group, or 52/26/13 GiB/rank |

The hybrid groups differ from the standalone TP2 pair.
At C1, idle DP or hybrid replicas still reserve their devices, weights, and KV pools.
They cannot count as useful C1 compute or free resources.
The retained UUID records match their declared physical mappings.
No live device inventory occurred.

## 2. Closed counts and work denominators

| Protocol | Completed jobs | Summary cells | Clean trials | Request instances | Controlled output tokens |
|---|---:|---:|---:|---:|---:|
| Constant-token TP/DP/hybrid | 10 | 80 | 400 | 1500 | 192000 |
| Separate C1 context scaling | 6 | 24 | 144 | 144 | 18432 |

These are request instances, not independent natural requests.
The cohort uses C1/2/4/8 and N1024/4096 across both Base sizes and five execution layouts.
Each cell has five clean repetitions.
Each model/layout job has eight cells and 40 clean trials.
Every request returns exactly 128 controlled tokens with `finish_reason="length"`.

Each cohort job has three 16-output warm-ups per case.
Eight separate trace trials follow its clean lane.
Neither warm-ups nor trace timing enters the headline denominator.
The constant token is 378, with singleton allowed-token sampling and EOS ignored.
This still executes the model and LM head.
Returned constant-token identity does not establish unrestricted numerical parity or task quality.

The context protocol uses N1024/8192/16384/32768 and C1.
Every model/TP job contains two patterns, four contexts, and three clean repetitions per pattern/context.
Thus, each context cell has six trials, not six independent prompts.
Its warm-ups use four outputs, not the cohort protocol's 16.
Post-timing numerical capture also uses four outputs and does not enter the 128-output timing denominator.

For either protocol, output TPS divides returned output tokens by observed wall seconds.
The cohort numerator is `C*128`.
The context numerator is 128.
Neither numerator includes prompt tokens or sums duplicate TP-rank outputs.
Neither quantity is steady-state serving throughput.

## 3. Timer endpoints

| Metric | Exact inspected boundary | Excluded or separate work |
|---|---|---|
| Cohort TP wall | Host `perf_counter` around one `LLM.generate` call | Sampling-parameter preparation precedes timing. Output checks and result-file writes follow timing. |
| DP4/hybrid parent wall | First sequential RELEASE send through all matching DONE receipts, on one parent monotonic clock | Prompt preparation and READY waits precede timing. Final result-file writes follow timing. |
| Replica local wall | One active frontend's native generate call | Not a replacement for parent wall. Idle workers execute no generate call. |
| Context wall | Host `perf_counter_ns` around its generate expression | This expression also constructs sampling parameters and indexes the returned list. Output assertions follow timing. |
| Context decode TPOT | `(last_token_ts-first_token_ts)*1000/127` | This excludes the first output interval. It is not isolated attention or collective time. |

The DP/hybrid parent interval includes release skew, completion transport, controller work, output assertions, and worker serialization.
Its residual against the longest local generate interval is not pure IPC.
It does not identify every scheduling cost.
The draft correctly keeps TP API and parent endpoints separate.
The [table generator](../scripts/build_result_tables_cd.py:25) suppresses cross-endpoint speedup normalization.

The native first-token metric is a complete frontend first-output metric.
It is not isolated GPU prefill compute.
The first/last token timestamps share their own clock.
Subtracting the frontend arrival clock from a core monotonic timestamp is invalid.

## 4. Numerical and uncertainty checks

All 80 cohort medians and 24 context medians match the closed summaries.
The CPU checks also reproduce all 16 paired context mean ratios.
The C1/N1024 draft ratios round to 1.340× and 1.267×.
They divide TP1 wall medians by TP4 wall medians within the cohort protocol.

The 14/16 lowest-TP4 count matches the descriptive four-GPU comparison table.
The two other C8/N4096 TP4-versus-best-alternate differences remain below 0.4%, relative to TP4 wall.
The [draft's line 150](final_report_draft_v1.md:150) must name that comparator and denominator.
It must not imply that every pair among the three arms differs by less than 0.4%.
For Base2.6B, the largest-to-smallest median spread is approximately 1.203%, relative to the smallest median.
Different timer endpoints and sequential jobs prevent interpreting those differences as a reliable architecture ranking.
The runner randomizes case order within each job's repetitions.
Method jobs themselves are not randomized paired architecture trials.

The N32768 context ratios are means of two matched pattern ratios after within-pattern timing medians.
They are not a broad workload estimate or an attention-only speedup.
The retained request-bootstrap 95% intervals are:

- Base1.4B: 2.188318954×, interval [2.186261364, 2.190376545].
- Base2.6B: 2.209844422×, interval [2.208711805, 2.210977038].

The bootstrap uses 4000 resamples of only two frozen patterns.
Repeated trials, tokens, layers, and depth events do not become independent requests.
The draft's uncertainty limits must remain beside these ratios.

The memory reservations in Section 1 match the frozen configurations.
The former cohort v1 proposal used 32 GiB and never ran.
Base2.6B C8/N4096 plus 128 reserved output positions needs 49.5 GiB of structural TP1 KV.
The 52 GiB v2 pool exceeds that quantity.
It does not prove zero preemption or sufficient space for every graph, workspace, and weight allocation.

KV reservation is not actual HBM traffic.
Job peak allocated memory is not total device residency or a separate per-context peak.
The 32896-position runtime cap covers 32768 prompt tokens plus 128 output tokens.
It does not establish execution at the configured 65536-position model limit.

## 5. Required reproduction-index fixes

### Locate and rebuild the closed context analysis explicitly

The [index](reproduce_final_v1.md:50) names `long_context_*` runs.
The actual closed directories are `outputs/native_longctx_base{14,26}_P{1,2,4}_v3`.
Use those exact receipts.
For a CPU-only rebuild from the documented project directory, the source supports:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.analyze_native_long_context \
  --attempt 3 --output analysis/long_context_reproduction_01
```

This command is not executed in this audit.
Its output argument is relative to the campaign root inside the analyzer.
Do not add a second campaign-directory prefix.
The hybrid analyzer instead uses its output `Path` directly from the working directory.

### Distinguish archived rebuilds from fresh-run analysis

The hardware analyzers name original completed directories.
Changing a guard output name does not redirect those selectors.
A new reproduction therefore needs a reviewed selector or source-root change before its new trials enter analysis.
Rebuilding the archived analysis is still valid.
It is not analysis of a newly generated output merely because the analysis directory is new.

The index already gives this warning for native-quality validation.
Apply the same distinction to the hardware selectors.
Preserve the original selectors and source manifests.

### Reconstruct the complete guarded launch

The [index](reproduce_final_v1.md:46) refers to commands in `run_contract.json`.
Those receipts store the inner runner command and a separate device mask.
The inner command alone is not a self-contained guarded launch.
Document the `gpu_session` wrapper, recorded mask, fresh output name, timeout, and inner command together.
The preserved queue receipts and frozen contracts supply those command forms.
Do not execute a runner directly without its guard environment.

## 6. Source freeze and modeled-evidence limits

All 667 hashes named by the two closed hardware summaries match current files.
They include 544 clean trial JSON files and the summaries' supporting metadata and captures.
The audit also pins all 16 launch receipts separately.

A selected source check compares 142 receipt entries with current files.
Five entries differ, all for unrelated DP helper files captured in two TP4 launch snapshots.
The TP cohort runner does not import those DP helpers.
These broad-snapshot differences do not establish executed TP-source drift.
They also do not justify claiming that every file in every launch snapshot remains unchanged.
The source check is not a complete binary-build or dynamic-call-closure receipt.

The draft still needs the preceding review's C14 denominator, C08 negative-scope, and depth-affinity hindsight qualifiers.
Its P8 projections remain incomplete and unmeasured.
No baseline count, capacity equation, or cost-model coefficient becomes measured candidate headroom.
No missing candidate experiment becomes universal failure.

## Limits and actual work

This audit does not execute any reproduction command, unit-test suite, model, GPU operation, or new architecture.
It does not read pending native2.6 quality outcomes, candidate held-out data, or protected factorial results.
Quality-command selectors receive static source inspection only.
Task-accuracy counts and pending-arm completion status remain outside this hardware audit.
No old file changes.

Work-segment initial timestamp: 2026-10-19T19:06:43Z.
The JSON records the final accounting cutoff and validation results.
No pause or deliberate wait occurred.
Concurrent worker time does not add campaign wall time.
