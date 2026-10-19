# Native TP2 natural calibration

## Scope

Both Base checkpoints completed the frozen native TP2 calibration cohort.
Each model uses eight GSM8K requests and eight MATH-500 requests.
These are baseline quality observations, not candidate results.
Held-out requests remain untouched.

The native runtime uses fixed R4, BF16, greedy generation, and an 8192-token cap.
EOS and the frozen domain stop rules remain active.
The two jobs use physical GPUs 4 and 6.
The [queue record](../artifacts/native_tp2_quality_queue_v1.json) retains exact commands and completion times.

## Observed results

| Base model | Domain | TP1 correct/N | TP2 correct/N | TP2 minus TP1 | Paired 95% CI, pp | Exact output-ID parity | TP1/TP2 truncations |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1.4B | GSM8K | 3/8 | 4/8 | +12.5 pp | [0.0, 37.5] | 6/8 | 0/0 |
| 1.4B | MATH-500 | 4/8 | 6/8 | +25.0 pp | [0.0, 62.5] | 0/8 | 0/0 |
| 2.6B | GSM8K | 7/8 | 7/8 | 0.0 pp | [0.0, 0.0] | 8/8 | 0/0 |
| 2.6B | MATH-500 | 4/8 | 5/8 | +12.5 pp | [0.0, 37.5] | 1/8 | 3/2 |

The TP1 reference is the previously recorded native asynchronous baseline.
It is not the separate same-GPU scheduling recheck.
Model revision, prompt IDs, cap, temperature, and source protocol hash match within each pair.
The comparison uses 4000 request-level bootstrap resamples.
The [pair artifact](../artifacts/native_quality_tp1_tp2_calibration_v1.json) retains first divergence and raw-output paths.
The [scored checkpoint](../artifacts/quality_checkpoint9.json) retains parsing failures, stop reasons, and all baseline arms.

Eight requests per domain cannot establish population quality equivalence or a reliable accuracy improvement.
The observed accuracy changes do not establish that TP improves reasoning quality.
Different BF16 execution paths can produce different natural continuations.
The measurements do not identify the numerical cause.
Unpaired GPU identity and execution differences also remain possible confounders.

## Consequences for the campaign

Native natural calibration now covers TP1, TP2, and TP4 for both Base sizes.
The TP4 comparison remains in its [separate report](native_quality_tp_calibration_v1.md).
Equal correct counts do not imply exact token parity.
Natural wall-time ratios can include different output lengths and termination behavior.
The controlled-length hardware comparisons remain separate.

No new architecture is active in these jobs.
No candidate quality gate or architecture speedup passes from this table.
This small calibration cohort does not reproduce an official reference score.

## Reproduction

Run the frozen commands in the queue record through the scoped GPU guard.
Use new output directories to preserve existing evidence.
The following commands reproduce the completed-output analysis:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.score_quality --output poc_ouro_parallelism_deep_discovery/artifacts/quality_checkpoint9.json
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.compare_native_quality_tp --scored artifacts/quality_checkpoint9.json --output artifacts/native_quality_tp1_tp2_calibration_v1.json --comparison-tp 2
```

Both analysis commands refuse to replace an existing destination.
