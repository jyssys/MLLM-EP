# Existing native scheduler control

This checkpoint strengthens the comparison baseline. It does not introduce an architecture.
The earlier native runs explicitly disabled asynchronous scheduling.
The new control enables the existing native option.
All six model/TP configurations completed.

## Matched controlled work

Both arms use pinned Base checkpoints, BF16, fixed R4, FlashAttention, and CUDA Graphs.
The native custom All-Reduce remains enabled where applicable.
Each context contains two frozen request shapes and five trials per shape.
Each request produces the same 128 teacher-forced output tokens.
The prompt lengths are 1024, 4096, and 8192 tokens.
These are hardware workloads, not natural reasoning evaluations.

| Model | TP | Async E2E at 1K / 4K / 8K, ms | Default / async at 1K / 4K / 8K |
|---|---:|---|---|
| Base1.4B | 1 | 1046.385 / 1211.351 / 1457.154 | 1.286× / 1.328× / 1.158× |
| Base1.4B | 2 | 925.956 / 1007.258 / 1151.865 | 1.403× / 1.369× / 1.326× |
| Base1.4B | 4 | 791.739 / 841.135 / 937.198 | 1.412× / 1.388× / 1.374× |
| Base2.6B | 1 | 2068.839 / 2287.745 / 2779.484 | 1.182× / 1.163× / 1.077× |
| Base2.6B | 2 | 1825.577 / 1987.189 / 2277.982 | 1.156× / 1.138× / 1.109× |
| Base2.6B | 4 | 1564.218 / 1650.577 / 1837.073 | 1.234× / 1.221× / 1.194× |

The values are medians of ten trials per cell.
Method order was sequential, not randomized.
Only two distinct request shapes support each cell.
The results therefore do not establish a broad request-level confidence interval.
They do establish a stronger tested native controlled-work reference.

## Numerical scope

The separate capture compares four teacher-forced output positions.
Both TP1 and TP2 controls match the default logits, hidden states, and sampled KV bits for both models.
Both TP4 controls retain the four top-1 choices, but some captured values differ.
This checkpoint does not assign a cause to those TP4 differences.
It does not establish full-generation or task-quality parity from four positions.

Two natural TP1 calibration controls are active.
They use the same frozen sixteen requests and common 8192-token cap as the earlier native pilots.
They permit EOS and the frozen stop strings.
The held-out cohort remains untouched.
Their output and accuracy comparison is pending.

## Effect on candidate assessment

The scheduling benefit is generic runtime capability, not a recurrence-specific contribution.
A candidate must compare with the stronger eligible native reference.
It cannot count a cost that this existing option already removes.
Natural quality and numerical obligations remain separate from controlled-work performance.

The analysis keeps default, async, and fused variants in separate summaries.
It never pools their trials into one median.
The earlier analysis snapshots remain unchanged.

The refreshed C05 cost screen uses async-enabled TP1 and same-P TPOT values.
It still uses TPOT divided by four as a coarse recurrent-pass estimate.
This estimate is not a directly measured pass timer.
No modeled number becomes an achieved architecture speedup.

## Evidence and reproduction

- `artifacts/native_controls_checkpoint4.json` contains comparisons and captured numerical differences.
- `analysis/async_checkpoint_01/native_raw.csv` contains the controlled-work trials.
- `analysis/async_checkpoint_01/native_summary.csv` contains the separate async medians.
- `artifacts/cost_screens_async_checkpoint1.json` contains the refreshed modeled screen.
- `artifacts/baseline_control_queue.json` contains the finite queue, timestamps, commands, and completion status.

Use `native_baseline --async-scheduling` in a new `control_async_` output directory.
Use `analysis_and_plots --native-family async --label NEW_LABEL` for a separate analysis snapshot.
Use `native_quality_pilot --async-scheduling` for the existing-option natural-output control.
The normal GPU guard and UUID checks remain mandatory.
