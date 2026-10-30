# Static Recurrence-Cut Performance Map

Status: finite mechanism study completed. This is not the nine-arm serving-capacity evaluation.

## Source and model contract

The numerical release is `3b7a722898dab5f6dd797b474c1ff461659d8fa1`.
The inherited RCC release is `b53a8fed36a1706ac74754b53753f20edb8b913e`.
Initial implementation checkpoint: `28b1cbf75b0b26a758e98e26c60af9860e7c498e`.
Code and raw-evidence release: `71453683a69d600a7cabc667386df4dd1b622f7a`.
The final summary identifies the final implementation revision.
Each run also preserves the actual overlay archive.

| Model | Checkpoint revision | Precision | Recurrent depth |
|---|---|---|---:|
| Ouro Base 1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | BF16 | R4 |
| Huginn-0125 | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | BF16 | R32 |

Ouro uses the released `down_norm` correction.
Huginn uses `down_norm_adapter`.
No weight, precision, or recurrent-depth change was introduced.
Native Mixed is off. An executor guard rejects combined Prefill/Decode recurrent batches.

Clean timing uses one 1P+1D pair.
The repeated map crosses GPU0/1 and GPU4/5.
The frozen-choice hold-out uses GPU6/7.
The initial Huginn correctness probes also use GPU2/3.
All mappings include physical UUID checks.
Owned burners and other experiment workers are absent from clean GPU measurements.
This is not a claim of complete CPU-host exclusivity.

## Correctness before timing

Both models passed the released internal baseline gate.
The new retained-state migration then passed independent GPU checks.
The checks cover all initial cuts at input length 128.
They also cover midpoint migration at lengths 512 and 1,536.
Outputs, valid Prompt KV, final hidden state, receiver bytes, and recurrent-row counts matched.
Cut 0 also matched the full-Prefill control.

The final map contains 1,208 measured requests in 151 finite cells.
All 151 full-depth work checks passed.
All 828 non-reference target comparisons matched their within-start output anchors.
Reference-cohort self-comparisons are not included in that 828 denominator.
The separate native-streaming comparison passed 48/48 paired output checks.
The detailed qualification denominators remain in the correctness report and JSON evidence.

These results do not establish original-HF losslessness or downstream accuracy.
GSM8K and MATH accuracy are NOT_MEASURED in this study.

## Workloads and scheduling

The main map uses eight fixed ShareGPT-derived token-prefix prompts per model.
Every prompt has 512 tokens under that model's tokenizer.
Token chunk size is 128. The scheduled token budget is also 128.
Each worker has at most four active compute slots.
Eight queued requests do not mean a fixed eight-row Decode batch.
The actual batch changes with admission and request completion.

| Profile | Arrival and output configuration | Interpretation |
|---|---|---|
| P-pressure | Eight arrivals, 1 ms apart, output cap16 | Short-output Prefill burst |
| Balanced | Eight arrivals, 90 ms apart for Ouro and 400 ms for Huginn, cap64 | Intermediate offered-work profile |
| Verified D-pressure | Two background requests, cap128, then six targets, cap32 | Targets enter only after both background requests generate four tokens |

Both background requests must remain active at target release.
The background requests always use Full Cut R.
Only the target cut changes in this loaded-D mechanism probe.
This is not a global static policy applied to all historical requests.
The profile names describe the experiment design. They do not assert exact GPU-utilization ratios.

Generation uses actual greedy outputs with `ignore_eos=True`.
This is controlled generation, not Natural-EOS completion or frozen-token replay.
The initial fixed-offset D-pressure pilot did not guarantee active Decode at target arrival.
Its data remain available, but it is excluded from the decisive repeated map below.

## Repeated L512 map

Values are mean all-request E2E in milliseconds.
Brackets contain start-bootstrap 95% intervals for that mean.
Each cell uses three independent starts and eight measured requests per start.
Full-Cut anchors run first for output-parity checks.
The candidate order changes across starts.
Thus the control position is not fully counterbalanced.
Per-cell warm-up reduces startup effects but does not prove the absence of order drift.
The intervals condition on the same frozen diagnostic prompts.
They do not represent benchmark-population uncertainty.

Full Cut R below is the matched completed-prompt snapshot control.
It is not the native chunk-streaming PD_G1 control.

### Ouro R4

| P cut | P-pressure E2E ms | Balanced E2E ms | Verified D-pressure E2E ms |
|---:|---|---|---|
| 0 | 2,923 [2,901, 2,939] | 4,696 [4,680, 4,714] | 4,389 [4,248, 4,539] |
| 1 | 2,749 [2,724, 2,782] | 4,343 [4,331, 4,358] | 3,954 [3,833, 4,109] |
| 2 | 2,655 [2,588, 2,739] | 4,417 [4,327, 4,533] | 3,843 [3,697, 4,009] |
| 3 | 2,799 [2,575, 3,065] | 4,455 [4,277, 4,677] | 3,827 [3,678, 3,974] |
| 4 | 3,510 [3,038, 3,910] | 4,501 [4,204, 4,755] | 3,509 [3,229, 3,750] |

The lowest point mean changes from Cut 2 to Cut 1 to Cut 4 across these profiles.
Intermediate cuts reduce Prefill-side pressure in the short-output profile.
Full Cut 4 better protects the loaded Decode worker in this experiment.
Point minima alone do not establish a reliable online selector.
Several intervals overlap, and the calibration ranking changes in later starts.

### Huginn R32

| P cut | P-pressure E2E ms | Balanced E2E ms | Verified D-pressure E2E ms |
|---:|---|---|---|
| 0 | 8,533 [8,451, 8,605] | 16,185 [16,058, 16,295] | 14,478 [13,514, 15,213] |
| 4 | 8,214 [8,181, 8,244] | 15,549 [15,437, 15,646] | 14,097 [13,240, 14,740] |
| 8 | 7,920 [7,890, 7,955] | 14,967 [14,871, 15,044] | 13,789 [13,012, 14,409] |
| 16 | 7,561 [7,206, 8,038] | 14,670 [13,890, 15,600] | 13,128 [12,259, 13,873] |
| 24 | 7,467 [7,092, 7,925] | 15,118 [14,759, 15,538] | 12,705 [12,203, 13,199] |
| 28 | 7,327 [6,836, 7,932] | 14,606 [14,528, 14,675] | 12,500 [12,112, 12,951] |
| 32 | 7,431 [6,885, 8,078] | 14,395 [13,854, 15,031] | 12,272 [11,923, 12,693] |

Cut 28 has a slightly lower P-pressure point mean than Cut 32.
The difference is small relative to the observed variation.
Full Cut 32 has the lowest point mean in the other two profiles.
The data do not show a large, reliable adaptive-cut advantage for Huginn.

## Long-prompt subset

This supplementary probe uses input length 1,536 and chunk/token budget256.
It tests only Cut 0, the midpoint, and Full Cut R.
Each cell has one independent start and eight measured requests.
No confidence interval or globally optimal long-context cut is claimed.
Length and chunk size both change from the L512 map.
This is not a pure context-length ablation.

| Model | P cut | P-pressure mean E2E ms | Verified D-pressure mean E2E ms |
|---|---:|---:|---:|
| Ouro R4 | 0 | 4,150 | 5,367 |
| Ouro R4 | 2 | 3,713 | 5,150 |
| Ouro R4 | 4 | 4,619 | 4,199 |
| Huginn R32 | 0 | 11,697 | 20,442 |
| Huginn R32 | 16 | 10,920 | 17,806 |
| Huginn R32 | 32 | 10,224 | 15,697 |

Ouro again shows a pressure-dependent midpoint versus Full Cut trade-off.
Huginn's Full Cut is best among the three tested long-context cuts.
The reserved KV pool can limit concurrent long requests before compute slots fill.
This runtime constraint must not be confused with a universal model property.

## Native PD_G1 control

The original PD_G1 path streams completed full-depth token chunks.
The matched Full Cut control sends one completed-prompt snapshot.
Both use the qualified numerical path and separate Prefill/Decode phases.
Their 48 paired output sequences matched.

The single-start control shows that snapshot versus streaming can change performance by itself.
For P-pressure, snapshot E2E speedups relative to native were 1.270x for Ouro and 1.222x for Huginn.
For verified D-pressure, native streaming was faster in that start.
These controls have one paired start and no qualifying confidence interval.
They are evidence of a protocol confound, not a new recurrent-partitioning gain.

The exact values are in `adaptive_recurrence_cut/native_pd_g1_vs_snapshot.csv`.
All nine-arm capacity claims remain NOT_MEASURED.

## Timing, communication, and memory interpretation

`static_cut_metrics_by_start.csv` preserves TTFT, TPOT, observed ITL, E2E, and finite throughput.
Its finite requests/s is not sustainable serving capacity.
Multi-token notifications do not supply individual token timestamps.
The analysis reports their count instead of inventing ITL samples.

`e2e_component_shares.csv` partitions every request's E2E into host-observed intervals.
These intervals include reservation, scheduling, compute, handoff, and generation observations.
They are not dispatch-versus-GEMM GPU percentages.
The profiler-only kernel tables use a separate evidence class.
Overlapping GPU or transfer durations are not summed as an E2E critical path.

`migration_payloads.csv` records actual NIXL bytes and transfer completions.
The snapshot segment accounting reconciles with the measured transport counters.
Control-message bytes and pure KV DMA latency are NOT_MEASURED.

Each Ouro worker stores 2,869,305,346 weight bytes and reserves 12,884,901,888 KV bytes.
Each Huginn worker stores 7,129,953,600 weight bytes and reserves 33,218,887,680 KV bytes.
These values do not shrink with the recurrent cut in this implementation.
The matched snapshot control retains continuation staging capacity.
The native streaming control needs a smaller registered hidden slab.
The memory CSV distinguishes structural slab sizes from measured Torch peaks.
Graph workspace is NOT_MEASURED separately.

## Gate and reproduction

The map supports a finite load-dependent trade-off, especially for Ouro.
It does not establish Dynamic SLO-goodput, Token-axis superiority, or Loop-specific novelty.
The oracle-headroom report explains the frozen-selector results and remaining uncertainty.
The token/loop/layer report states the same-boundary equivalence constraint.
The online-policy report identifies all NOT_RUN arms and workloads.

The prepared W1 calibration and test sets each contain 200 full-text prompts.
Their target output mean is 448 tokens. No W1 serving run is implied.
Four-GPU and eight-GPU serving, downstream accuracy, and main SLO calibration are NOT_RUN.

See `adaptive_recurrence_cut/reproduce.md` and `commands.jsonl` for actual execution commands.
All failed launches, raw outputs, source archives, and unused configurations remain available.
