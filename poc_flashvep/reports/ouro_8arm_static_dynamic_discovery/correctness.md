# Correctness and fidelity: final review

## Separate conclusions

Actual task accuracy, tensor fidelity, placement correctness, and observer validity are different checks.
The historical strict P512 numerical gate remains FAIL.
The study does not increase its thresholds after seeing task scores.
Custom T4, TD, and the native TP4 comparison retain an exploratory numerical warning.

## Model and inputs

The model is base ByteDance/Ouro-1.4B at revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Static uses 24 physical layers, R4, and 96 independent logical KV slots.
All primary weights remain BF16.
Every primary arm uses the same explicit canonical prompt IDs.

The first older Single pilot contained one tokenizer-version input mismatch.
That raw record remains preserved and excluded.
Whole-cohort reuse depends on canonical input identity, never answer score or runtime.
All eight final views contain the same 400 valid question IDs.
The artifact validator checks 122 protected earlier files and the checkpoint hash.

## Historical TP audit

The earlier operator audit passed 192/192 bounded checks.
That count is not a full-model numerical or quality certification.

| Historical P512 quantity | Observed value | Existing limit |
|---|---|---|
| Custom TP max hidden relative L2 | 0.469359 | 0.03 |
| Custom TP max logit absolute error | 5.210938 | 0.75 |
| Custom TP max per-step logit mean error | 0.802111 | 0.08 |
| Custom TP rank-0 owner-KV relative L2 | 0.024168 | 0.03 |
| Native TP4 max hidden relative L2 | 0.498974 | Historical diagnostic comparison |

The old Custom TP contract remains failed.
A passing rank-0 KV quantity does not cancel the other failed quantities.

The new study compares HF Single, custom unsharded, custom TP4, native TP1, and native TP4.
It covers two synthetic inputs at each P128/P256/P512/P2048 length.
P512 has 32 aligned positions. The other lengths have eight positions each.
There are 112 aligned positions per engine, not 112 independent serving samples.

Custom unsharded matches the captured HF seams bitwise.
Every BF16 variant matches the bounded FP32 reference's argmax at 112/112 positions.
Native TP1 already differs internally from HF.
The sampled first-crossing and local FP32 controls support numerical sensitivity.
They do not identify a unique full-model root cause.
The observation is last-token-only, not full-prompt state capture.
See [numerical_tp_diagnosis.md](numerical_tp_diagnosis.md).

## Physical PP probes

| Comparison | Live probe requests | Output IDs | Exit traces | KV scope |
|---|---|---|---|---|
| S4 vs L4 | 4 | 4/4 exact | 4/4 exact | All 96 slots bitwise |
| S4 vs O4 | 4 | 4/4 exact | 4/4 exact | All 96 slots bitwise |
| SD vs LD | 4 | 4/4 exact | 4/4 exact | All 96 slots bitwise |
| SD vs OD | 4 | 4/4 exact | 4/4 exact | All 96 slots bitwise |

Separate forced mixed-R2/R3/R4 diagnostics also pass the LD/OD cache checks.
They test current-row publication, deeper-loop reads, and unchanged historical rows.
Forced exits do not count as live Dynamic quality.

The first output token comes from full R4 prefill.
Only subsequent decode tokens can terminate early.
Independent body counters exclude graph preparation.
The work audit compares those counters with completed depth traces.
It validates 833 completed cohorts across the audited lanes.

## Full natural outputs

Static L4 and O4 match S4 output IDs on all 400 questions.
Dynamic LD and OD match SD output IDs and complete exit traces on all 400 questions.
The same-backend PP placement did not change observed quality on this cohort.

Custom T4 matches S4 complete output sequences on 208/300 GSM8K and 23/100 MATH questions.
TD matches SD sequences on 197/300 GSM8K and 20/100 MATH questions.
Their actual answer accuracy is measured separately.
Long output drift does not by itself prove an implementation defect or an accuracy loss.

Native TP4 matches native TP1 sequences on 215/300 GSM8K and 24/100 MATH questions.
Its answer accuracy differs by 0.000 pp and -1.000 pp, respectively.
The paired intervals remain wide.
No population-equivalence claim follows.

The primary grader counts a correct extracted answer from a truncated output.
The completed-correct column excludes such answers.
Parsing failures remain in the denominator.
All raw text, token IDs, stop reasons, and grades remain available.
[quality_results.md](quality_results.md) contains correct/N, truncation, and all required accuracy deltas.

## Official CDB reference

All 269 original checkpoint tensors in the reference match the pinned weights bitwise.
Optional unused adapter gates do not replace the original trained gate.
The CPU policy check matches the pinned official cumulative-hazard implementation in FP32 and BF16.
Hazard conversion uses FP32. Host CDF and survival use Python floating-point scalars.

Four short synthetic requests match official CDB and custom SD output/exit traces.
Cross-backend full-KV equality was not measured.
The official reference separately verifies current-row last_exited copies.

Long behavioral parity does not hold on the 114-question reference pilot.
Exact output matches are 41/64 GSM8K and 8/50 MATH.
Exact depth-trace matches are 17/64 and 0/50.
The custom family therefore does not claim task-wide behavioral reproduction of the native CDB backend.

The first-depth-drift audit finds 97 requests with different traces.
For 67 requests, the first depth difference precedes any different input-token history.
There are 36 same-history pairs with CDF observed at the same decision loop.
All 36 straddle q=0.5.
Missing earlier reference CDF values remain unavailable.
This identifies threshold sensitivity, not one unique defective kernel.

## TD KV warning

The short TD token and exit probes match their Single references.
Their assembled four-shard KV tensors remain numerically unresolved.
Aggregate KV relative L2 is 0.0533463462 or 0.0761231817 across the four probes.
Worst-slot values are 0.2152278870 and 0.2906669676.
These scopes differ from the historical rank-0-only KV comparison.
They do not pass a 0.03 diagnostic.
Each rank's local current-row copy identity passes independently.
A correct copy does not certify the preceding attention calculation.

## Instrumentation and timing

All 18 bounded observer comparisons preserve output IDs.
All custom comparisons also preserve recorded depth and computed-argmax sequences.
Each all-eight serving-layout control matches its primary task's first 32 output/depth positions on four requests.
Those controls use the original O4096 KV allocation.
They are not whole-generation quality evaluations.

Heavy wall overhead is 24.9% to 405.0%.
Custom heavy host clocks can include profiler-start latency.
Nsight GPU activity uses the later NVTX serving window.
Heavy wall time is UNVALIDATED for latency headlines.
Semantic observer parity does not remove this timing warning.

Clean replay uses identical frozen tokens, separate warm-up, and five measured repeats.
Dynamic replay also locks the observed future exit schedule.
Replay is not a live policy, benchmark accuracy, or a task-wide numerical PASS.
Native reference fields without exit traces remain NOT_AVAILABLE, not inferred R4 trace arrays.

## Tests, limits, and preservation

The CPU contract suite passes all 18 tests.
The profiler PATH regression failed before its repair and passes afterward.
The resume test preserves failed attempts and skips completed captures.
See [validation_tests.md](validation_tests.md).

The original Graph lifetime bug and skipped-owner KV-length bug are preserved in the failure ledger.
Both fixes precede their valid follow-up GPU results.
Neither fix retroactively explains the historical eager TP failure.

Full public-suite evaluation, optional DP4, optimized PP, and a formal noninferiority study are NOT_RUN.
No raw failure, earlier report, original model tensor, or frozen protocol was rewritten.
