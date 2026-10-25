# Three-model Looped MoE expert-efficiency discovery

## 1. Outcome and evidence classes

The overall verdict is **NO_GO_HEADROOM for the tested native B1 candidate grid**.
P3 is NOT_RUN because the measured-denominator discovery gate did not pass.
No actual candidate E2E acceleration is reported.
No EP, TP, PP, quantization, training, or proxy model entered the experiment.

LIVE_MEASURED denotes an actual GPU timing.
LIVE_PROBE_DERIVED denotes statistics from actual model tensors.
LIVE_FULL_MODEL_INTERVENTION denotes quality after a changed full-model trajectory.
ANALYTICAL denotes a conditional timing model.
NOT_MEASURED and UNRESOLVED remain missing evidence, not zero costs.

| Model | Best tested NLL-eligible arm | Omitted work | ΔNLL [95% CI], nats/token | Optimistic timing model | Achieved E2E speedup | P2 / P3 |
| --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | O1_drop_0.2_renorm0 | 16.84% | 0.0136 [0.0112, 0.0159] | 1.0462× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |
| IFM MoE Huginn | O1_drop_0.3_renorm0 | 19.06% | 0.0133 [0.0109, 0.0156] | 1.0525× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |
| Foil-1 | O1_drop_0.2_renorm0 | 9.16% | 0.0122 [0.0072, 0.0175] | 1.0151× | NOT_MEASURED | NO_GO_HEADROOM / NOT_RUN |


## 2. Table A — Checkpoint and architecture verification

| Model / revision | Learned parameters | Physical / recurrent sites | Passes / logical MoE calls | Experts / top-k | Training tokens | Backend / GPU |
| --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro / `f0e9c8af22c10f34ac716c52495ceb00525f3556` | 8,780,889,600 | 28 / 28 | 4 / 112 | 25 / 2 | 500B | xLLM FA4 / H100 SM90 |
| IFM MoE Huginn / `f1321d5272a11348d5093ad7ab1d310eb9de2e11` | 8,783,251,968 | 28 / 12 | 8 / 112 | 25 / 2 | 500B | xLLM FA4 / H100 SM90 |
| Foil-1 / `b3d128b44cbefd54e2c8784224a110186c8931f1` | 553,730,048 | 3 / 1 | 16 / 18 | 64 / 2 at core | 100B | Official Foil SDPA / H100 SM90 |


The runtime pins appear in [environment.json](environment.json) and [model_pins.json](model_pins.json).
All learned weights remain BF16 and GPU-resident.
Native FP32 reductions, logits, and positional buffers retain their original precision.
Each forward uses one visible GPU.
The eight workers are independent experiments, not a distributed model.

Ouro repeats 28 physical layers four times.
Huginn executes 8 prelude layers, 12 core layers eight times, and 8 coda layers.
Foil-1 repeats one shared expert site sixteen times with untied attention.
Its outer eight-expert sites are not recurrence statistics.
Foil has 553,730,048 learned parameters and 9,437,184 FP32 RoPE buffer elements.
Their persistent element count is 563,167,232, which explains the spec's rounded 563.2M count.
The [official model card](https://huggingface.co/ShourenWSR/how-to-loop-moe) also reports 553.7M learned parameters.

Both IFM checkpoints use the strict native xLLM loader.
Foil uses its official remote code and Foil-1 subfolder.
The final strict Foil load reports no missing, unexpected, or mismatched tensors.
No tracked upstream model source changed.

## 3. P0 numerical qualification

Native no-op observers preserve exact logits and short greedy outputs for all three models.
The q=0 interventions preserve shifted NLL and KL.
All three adapters match their official generation APIs on an eight-token smoke prompt.
Those checks compare the same backend and checkpoint.

| Model | Original BF16-math whole-model L2 | Original result | Strict FP32 operator max L2 | Strict FP32 whole-model L2 | Strict 8-token parity | First different token (1-based) |
| --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | 0.057401 | FAIL at original 0.05 | 0.002002 | 0.041632 | False | 4 |
| IFM MoE Huginn | 0.061840 | FAIL at original 0.05 | 0.001996 | 0.040615 | False | 4 |


The later strict diagnostic disables TF32 in its FP32 reference calculations.
Its operator error is small, but full-model greedy parity fails on that different prompt.
The report does not convert the original backend failure into a pass.
P0 is CONDITIONAL_BACKEND_FIDELITY for both IFM models and PASS for Foil-1.
No FA3-versus-FA4 benchmark accuracy equivalence is established.

## 4. Workloads and denominators

The sampling seed is 20261026.
Each domain has 8 smoke, 32 calibration, and 128 held-out IDs.
The domains are general language, GSM8K, and MATH-500.
Four synthetic examples test repeated and changing token patterns.
All models receive the same raw text, not assumed-identical token IDs.
Pretraining overlap is UNKNOWN.

The primary quality sweep uses 32 calibration and 128 held-out language passages per model.
Calibration prefixes have cap 128 and held-out prefixes have cap 512.
The stronger independent-site random-control follow-up uses cap 512 for both splits.
The summary does not duplicate its repeated O1 examples.
Actual token counts remain in quality_summary.csv.

P1 includes eight real GSM8K and eight real MATH prompts per model.
Those probes measure four free-running steps at the last position of each recomputed prefix.
Synthetic probes use prefix targets 128, 512, and 2,048.
Every model also has separate baseline generation timing at 16 and 64 output tokens.

The natural-generation accuracy pilot evaluates eight GSM8K and eight MATH questions per model.
It tests B0, O1 q=0.1, C1 threshold 0.2, and unconditional C2 reuse.
The broader 16-token fidelity grid runs all 27 primary arms on six prompts per model.
These capped rollouts are not full benchmark accuracy evaluations.
The 128-question downstream expansion did not run after the latency gate failed.
The default task-quality margin therefore remains UNRESOLVED, not PASS.

## 5. Table B — Clean native latency and the expert denominator

Each cell uses three independent model starts on physical GPU0.
Each start uses five warmups and twenty clean forward repetitions.
Intervals resample the three process-start means.
With three starts, interval coverage is only approximate.

| Model | Input tokens | Clean forward ms [95% CI] | Expert span ms | Observer / clean | Expert / observer | Expert / clean, optimistic | Peak allocated GiB | Validity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Foil-1 | 128 | 31.55 [30.26, 33.06] | 4.53 | 1.171 | 12.3% | 14.4% | 1.12 | PARTIAL_OBSERVER_OVERHEAD |
| Foil-1 | 512 | 31.03 [30.31, 32.30] | 5.03 | 1.106 | 14.6% | 16.2% | 1.20 | PARTIAL_OBSERVER_OVERHEAD |
| Foil-1 | 2048 | 32.02 [31.13, 33.09] | 5.93 | 1.072 | 17.3% | 18.5% | 1.49 | QUALIFIED_EVENT_SPAN |
| IFM MoE Huginn | 128 | 207.80 [206.35, 209.64] | 48.82 | 1.097 | 21.4% | 23.5% | 16.75 | QUALIFIED_EVENT_SPAN |
| IFM MoE Huginn | 512 | 215.89 [214.04, 218.68] | 56.54 | 1.067 | 24.5% | 26.2% | 17.69 | QUALIFIED_EVENT_SPAN |
| IFM MoE Huginn | 2048 | 229.46 [227.94, 231.54] | 69.44 | 1.069 | 28.3% | 30.3% | 21.40 | QUALIFIED_EVENT_SPAN |
| IFM MoE Ouro | 128 | 217.92 [212.20, 228.56] | 48.94 | 1.055 | 21.3% | 22.5% | 16.74 | QUALIFIED_EVENT_SPAN |
| IFM MoE Ouro | 512 | 219.80 [219.72, 219.93] | 57.68 | 1.065 | 24.6% | 26.2% | 17.68 | QUALIFIED_EVENT_SPAN |
| IFM MoE Ouro | 2048 | 234.04 [231.84, 235.87] | 71.94 | 1.114 | 27.6% | 30.7% | 21.40 | PARTIAL_OBSERVER_OVERHEAD |


Event spans and clean wall time come from separate forwards.
The event spans include internal launch gaps.
They are not summed kernel-service times.
Observer overhead above 10% produces a PARTIAL label.
Attention and routing spans appear separately in component_spans.csv.
Dispatch, combine, and unassigned host intervals remain UNRESOLVED as clean critical-path percentages.
No invented 100% E2E partition is provided.

### Baseline model-only generation

| Model | Prefix | Output tokens | E2E ms [95% CI] | Mean TTFT ms | Mean TPOT ms | Mean output tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| Foil-1 | 128 | 16 | 1000.4 [980.8, 1028.2] | 31.2 | 64.6 | 16.00 |
| Foil-1 | 128 | 64 | 3970.8 [3920.5, 4058.9] | 31.2 | 62.5 | 16.12 |
| Foil-1 | 512 | 16 | 998.5 [985.8, 1016.6] | 31.6 | 64.5 | 16.03 |
| Foil-1 | 512 | 64 | 4035.6 [3939.8, 4215.0] | 31.4 | 63.6 | 15.87 |
| IFM MoE Huginn | 128 | 16 | 3364.9 [3339.7, 3402.4] | 213.4 | 210.1 | 4.76 |
| IFM MoE Huginn | 128 | 64 | 13484.2 [13367.2, 13595.4] | 206.1 | 210.8 | 4.75 |
| IFM MoE Huginn | 512 | 16 | 3474.5 [3427.4, 3511.6] | 225.0 | 216.6 | 4.61 |
| IFM MoE Huginn | 512 | 64 | 13877.1 [13772.1, 14036.8] | 214.7 | 216.9 | 4.61 |
| IFM MoE Ouro | 128 | 16 | 3577.5 [3427.6, 3875.0] | 222.9 | 223.6 | 4.49 |
| IFM MoE Ouro | 128 | 64 | 14273.5 [13684.4, 15430.4] | 216.5 | 223.1 | 4.50 |
| IFM MoE Ouro | 512 | 16 | 3664.1 [3515.5, 3936.9] | 225.9 | 229.2 | 4.38 |
| IFM MoE Ouro | 512 | 64 | 14645.6 [14134.9, 15620.0] | 228.8 | 228.8 | 4.38 |


Both native runtime families recompute the entire prefix at every output token.
TPOT here is not cached-decode TPOT.
These hardware-shape runs disable EOS to fix output counts.
Natural-generation quality runs use real stop conditions instead.
Model loading, tokenizer work, network delivery, and request queueing are outside this timer.

The supplementary generation observer measures all sixteen actual steps.
It does not extrapolate the last replay or last forward.
It uses a separate representative prompt and process start.
Its clean latency need not equal the three-start macro table above.

| Model | Clean 16-token E2E ms | Observer E2E ms | Expert span ms | Expert / clean, optimistic | Expert / observer |
| --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | 3882.8 | 4171.5 | 914.5 | 23.6% | 21.9% |
| Foil-1 | 611.0 | 672.7 | 76.0 | 12.4% | 11.3% |
| IFM MoE Huginn | 3862.2 | 4163.5 | 907.6 | 23.5% | 21.8% |


### GPU concurrency control

The operational control repeats one independent IFM worker and eight independent workers three times each.
On GPU0, parallel/isolated forward time is 0.99308×.
Other GPUs run owned burners in the isolated condition and model workers in the parallel condition.
This control does not establish complete machine-idle timing or cross-model interference equivalence.
Primary timing remains in the sequential GPU0 lane.
GPU UUIDs, NVLink topology, temperatures, and worker allocation remain in raw provenance.

## 6. Table C — Routing does not imply redundant output

This table pools adjacent passes at the same physical site and token position.
It uses 128 held-out language sequences per model, with prefix cap 512.
Intervals resample sequences, not individual expert events.
Each metric retains its own matched-observation denominator.

| Model | Top-k Jaccard [95% CI] | Exact-set match | Shuffled-position Jaccard | Matched-input relative L2 | Matched-output cosine | Matched-output relative L2 |
| --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | 0.549 [0.545, 0.553] | 0.366 | 0.324 | 0.618 | 0.566 [0.560, 0.571] | 0.851 [0.844, 0.857] |
| IFM MoE Huginn | 0.450 [0.442, 0.458] | 0.250 | 0.226 | 0.521 | 0.690 [0.681, 0.700] | 0.798 [0.782, 0.813] |
| Foil-1 | 0.272 [0.264, 0.279] | 0.139 | 0.062 | 0.280 | 0.873 [0.871, 0.876] | 0.551 [0.540, 0.563] |


Input-relative-L2 conditions on the same expert.
The aggregate input cosine table instead covers all sampled tokens.
Those are different denominators.
Per-pass and gap-1/2/4/8 results appear in routing_summary.csv and representation_summary.csv.
Within-frame P10/P50/P90 averages are not pooled token quantiles.

Foil's late-pass output cosine can exceed 0.95 while relative L2 remains material.
Ouro's routing overlap falls in its final recurrent pass.
Huginn's final-pass output change also increases.
The negative examples show why an expert-ID match cannot certify output reuse.
Spatial and frequency-preserving shuffled controls remain separate from temporal comparisons.
Router entropy, confidence, margins, contribution fractions, and cancellation ratios appear in confidence_contribution_summary.csv.
Those confidence summaries use the smaller P1 smoke sample.
Mean relative contribution is 0.5 by top-2 normalization, so its distribution matters more than its mean.

## 7. Table D — Full-model quality versus work

| Model | Arm | Actual omitted expert edges | ΔNLL [95% CI] | Mean top-1 agreement | Optimistic timing model | Expert cache MiB | Measured E2E speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | B0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Ouro | q0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Ouro | O1_drop_0.2_renorm0 | 16.84% | 0.0136 [0.0112, 0.0159] | 0.9233 | 1.0462× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Ouro | O2_reuse_0.1 | 6.83% | 0.0103 [0.0086, 0.0121] | 0.9435 | 1.0183× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Ouro | C1_router_0.3 | 12.99% | 0.0102 [0.0083, 0.0121] | 0.9333 | 1.0353× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Ouro | C2_reuse_guardNone | 48.30% | 3.3565 [3.2088, 3.4991] | 0.3173 | 1.1452× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Ouro | N1_independent_site_drop_0.2 | 16.84% | 0.2874 [0.2627, 0.3131] | 0.7744 | 1.0462× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Huginn | B0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Huginn | q0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Huginn | O1_drop_0.3_renorm0 | 19.06% | 0.0133 [0.0109, 0.0156] | 0.9301 | 1.0525× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Huginn | O2_reuse_0.1 | 6.31% | 0.0082 [0.0065, 0.0100] | 0.9502 | 1.0168× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Huginn | C1_router_0.3 | 6.00% | 0.0028 [0.0015, 0.0042] | 0.9564 | 1.0160× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Huginn | C2_reuse_guardNone | 60.18% | 4.3506 [4.1751, 4.5341] | 0.2184 | 1.1871× | NOT_RECORDED | NOT_MEASURED |
| IFM MoE Huginn | N1_independent_site_drop_0.3 | 19.06% | 0.0856 [0.0782, 0.0938] | 0.8602 | 1.0525× | NOT_RECORDED | NOT_MEASURED |
| Foil-1 | B0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000× | NOT_RECORDED | NOT_MEASURED |
| Foil-1 | q0 | 0.00% | 0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000× | NOT_RECORDED | NOT_MEASURED |
| Foil-1 | O1_drop_0.2_renorm0 | 9.16% | 0.0122 [0.0072, 0.0175] | 0.8962 | 1.0151× | NOT_RECORDED | NOT_MEASURED |
| Foil-1 | O2_reuse_0.1 | 5.97% | 0.0019 [-0.0002, 0.0040] | 0.9364 | 1.0098× | NOT_RECORDED | NOT_MEASURED |
| Foil-1 | C1_router_0.3 | 4.08% | 0.0003 [-0.0015, 0.0022] | 0.9396 | 1.0066× | NOT_RECORDED | NOT_MEASURED |
| Foil-1 | C2_reuse_guardNone | 37.18% | 1.5990 [1.4918, 1.7087] | 0.4245 | 1.0641× | NOT_RECORDED | NOT_MEASURED |
| Foil-1 | N1_independent_site_drop_0.2 | 9.16% | 0.0316 [0.0255, 0.0377] | 0.8730 | 1.0151× | NOT_RECORDED | NOT_MEASURED |


O1 is a contribution-ranked hindsight heuristic, not an optimal oracle.
O2 ranks same-expert output-reuse opportunities using baseline error.
Both require offline information unavailable at deployment.
C1 uses current router weights before expert arithmetic.
C2 uses its own previous-pass output and optional input-delta guards.
Current gate weights always multiply the cached raw output.

The work denominator includes all expert executions, including unmodified outer sites.
Reuse cannot replace outputs on the first pass.
Changed-router constraints also reduce actual omissions below the requested quota.
Rows disappear before native expert GEMMs.
This proves logical row omission, not proportional hardware-instruction reduction.
The primary teacher-forced runs did not record expert-cache peaks.
Those cells are NOT_RECORDED, not zero.
Natural-generation runs separately record controller cache tensors, not total HBM or all temporary allocations.
Unmeasured cache traffic and selector overhead are not assumed free in an implementable system.
They are free only in the explicitly optimistic timing model.

The independent-site N1 control matches realized O1 row counts at every site and pass.
Its NLL penalty is generally larger than contribution-informed O1.
This supports contribution selectivity, not a novel recurrence-specific method.
Legacy random masks were correlated across sites and remain labeled separately.
Free-running random controls match nominal quota, not necessarily realized work after divergence.

### Native-weight granularity test

| Model | Prefix geometry | q=0 component ms [95% CI] | q=0.5 component ms [95% CI] | Ratio of means | Nonempty expert groups |
| --- | --- | --- | --- | --- | --- |
| IFM MoE Ouro | 128 | 0.3636 [0.3540, 0.3822] | 0.3610 [0.3495, 0.3814] | 1.007× | 9.0 → 7.0 |
| IFM MoE Ouro | 512 | 0.4380 [0.4336, 0.4460] | 0.4465 [0.4097, 0.5033] | 0.981× | 17.0 → 16.0 |
| IFM MoE Ouro | 2048 | 0.6605 [0.6267, 0.7220] | 0.5451 [0.5357, 0.5589] | 1.212× | 25.0 → 25.0 |
| IFM MoE Huginn | 128 | 0.3914 [0.3865, 0.4000] | 0.3827 [0.3748, 0.3911] | 1.023× | 16.0 → 14.0 |
| IFM MoE Huginn | 512 | 0.4576 [0.4484, 0.4672] | 0.4365 [0.4270, 0.4453] | 1.048× | 21.0 → 20.0 |
| IFM MoE Huginn | 2048 | 0.5896 [0.5856, 0.5951] | 0.5354 [0.5063, 0.5910] | 1.101× | 23.0 → 22.0 |
| Foil-1 | 128 | 0.2581 [0.2574, 0.2587] | 0.2165 [0.2160, 0.2173] | 1.192× | 52.0 → 41.0 |
| Foil-1 | 512 | 0.2737 [0.2734, 0.2740] | 0.2572 [0.2566, 0.2577] | 1.064× | 57.0 → 53.0 |
| Foil-1 | 2048 | 0.3449 [0.3396, 0.3531] | 0.2936 [0.2915, 0.2957] | 1.175× | 61.0 → 59.0 |


This microbenchmark covers one recurrent site per model.
It excludes row selection, compaction, routing, cache lookup, and combine.
Quota order changes between process starts.
The Huginn q=0 prefix-512 cell has a large between-start interval.
Its ratio of means must not be treated as a stable system speedup.
Half-row omission does not generally halve the measured component time.
The experiment did not measure tensor-core instruction counts or HBM bandwidth.

## 8. Natural-generation quality pilot

| Model | Dataset | Arm | Correct / N | Accuracy | Δ accuracy pp [95% paired CI] | Parse failures | Capped outputs | Token-exact / N | Peak expert cache MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Foil-1 | gsm8k | B0 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 2 | 2 | 8 / 8 | 0.00 |
| Foil-1 | gsm8k | C1_router_0.2 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 2 | 2 | 2 / 8 | 0.00 |
| Foil-1 | gsm8k | C2_reuse_guardNone | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 5 | 8 | 0 / 8 | 20.07 |
| Foil-1 | gsm8k | O1_drop_0.1_renorm0 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 2 | 2 | 1 / 8 | 0.00 |
| Foil-1 | math500 | B0 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 6 | 8 | 8 / 8 | 0.00 |
| Foil-1 | math500 | C1_router_0.2 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 4 | 8 | 4 / 8 | 0.00 |
| Foil-1 | math500 | C2_reuse_guardNone | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 8 | 8 | 0 / 8 | 11.88 |
| Foil-1 | math500 | O1_drop_0.1_renorm0 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 7 | 8 | 1 / 8 | 0.00 |
| IFM MoE Huginn | gsm8k | B0 | 6 / 8 | 75.0% | 0.0 [0.0, 0.0] | 0 | 0 | 8 / 8 | 0.00 |
| IFM MoE Huginn | gsm8k | C1_router_0.2 | 7 / 8 | 87.5% | 12.5 [0.0, 37.5] | 0 | 0 | 2 / 8 | 0.00 |
| IFM MoE Huginn | gsm8k | C2_reuse_guardNone | 0 / 8 | 0.0% | -75.0 [-100.0, -37.5] | 8 | 0 | 0 / 8 | 194.01 |
| IFM MoE Huginn | gsm8k | O1_drop_0.1_renorm0 | 7 / 8 | 87.5% | 12.5 [0.0, 37.5] | 0 | 0 | 2 / 8 | 0.00 |
| IFM MoE Huginn | math500 | B0 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 7 | 6 | 8 / 8 | 0.00 |
| IFM MoE Huginn | math500 | C1_router_0.2 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 6 | 7 | 0 / 8 | 0.00 |
| IFM MoE Huginn | math500 | C2_reuse_guardNone | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 8 | 2 | 0 / 8 | 89.73 |
| IFM MoE Huginn | math500 | O1_drop_0.1_renorm0 | 1 / 8 | 12.5% | 12.5 [0.0, 37.5] | 5 | 6 | 1 / 8 | 0.00 |
| IFM MoE Ouro | gsm8k | B0 | 4 / 8 | 50.0% | 0.0 [0.0, 0.0] | 1 | 0 | 8 / 8 | 0.00 |
| IFM MoE Ouro | gsm8k | C1_router_0.2 | 4 / 8 | 50.0% | 0.0 [0.0, 0.0] | 1 | 0 | 6 / 8 | 0.00 |
| IFM MoE Ouro | gsm8k | C2_reuse_guardNone | 0 / 8 | 0.0% | -50.0 [-87.5, -12.5] | 8 | 5 | 0 / 8 | 256.87 |
| IFM MoE Ouro | gsm8k | O1_drop_0.1_renorm0 | 5 / 8 | 62.5% | 12.5 [0.0, 37.5] | 1 | 0 | 2 / 8 | 0.00 |
| IFM MoE Ouro | math500 | B0 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 7 | 7 | 8 / 8 | 0.00 |
| IFM MoE Ouro | math500 | C1_router_0.2 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 7 | 6 | 0 / 8 | 0.00 |
| IFM MoE Ouro | math500 | C2_reuse_guardNone | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 7 | 7 | 0 / 8 | 147.42 |
| IFM MoE Ouro | math500 | O1_drop_0.1_renorm0 | 0 / 8 | 0.0% | 0.0 [0.0, 0.0] | 8 | 6 | 0 / 8 | 0.00 |


The maximum output length is 256 new tokens.
Greedy generation permits EOS and registered next-question stops.
GSM8K uses the official IFM eight-shot text format.
MATH uses a declared zero-shot Problem/Answer format and a strict answer parser.
The parser is not the official MATH mathematical-equivalence evaluator.
Raw outputs, tokens, first divergence, answers, and stop reasons remain preserved.

Eight questions cannot establish a 1 percentage point accuracy margin.
Foil's baseline is weak on both tasks.
MATH baselines are weak and often truncated for all models.
Matching zero scores is not evidence of preserved quality.
The report makes no paper-quality reproduction claim.

## 9. Analytical headroom and decision

The conditional timing model is S = 1 / (1 - f_expert × q).
The optimistic f uses expert event spans divided by clean forward time.
That fraction comes from a separate representative passage, not each held-out quality request.
The more conservative observer denominator remains in oracle_summary.csv.
The model assumes fixed other costs, proportional expert-time removal, and zero added overhead.
It is not a proven universal bound.

The selected points are post hoc summaries of a predeclared diagnostic sweep.
They are not newly tuned deployable thresholds.
Even this favorable frontier stays below 1.10× in both IFM models at prefix 512.
Using the larger prefix-2,048 expert fractions still yields less than 1.10× for those same q values.
That length transfer is an estimate, not a prefix-2,048 quality measurement.
Downstream accuracy equivalence remains unproven and can only weaken this gate.

Unconditional C2 reuse causes large NLL increases in all three models.
Stricter guards preserve more fidelity but admit little reusable work.
The same shared expert weights still compute different refinements.
There is no measured recurrence-specific acceleration opportunity sufficient to justify P3 here.

## 10. Limits and next research

The native reference paths are cache-free research runtimes.
Their large launch and host costs need not match optimized cached serving.
The IFM backend qualification and downstream sample size limit generalization.
No claim extends to larger batches, long-context production decoding, or EP.
No machine-wide idle baseline, end-to-end candidate speedup, or hardware bandwidth result is available.

A separate optimized-runtime study could first establish the cached-decode expert fraction.
It must use the same model semantics and repeat the quality-work frontier.
This experiment does not authorize a new engine or an EP method.
Do not repeat unconditional expert-output reuse without a different, testable error-control mechanism.

## 11. Artifact map

- [methodology.md](methodology.md): alignment, intervention, and statistical definitions.
- [correctness.md](correctness.md): numerical checks and quality limits.
- [timing_contract.md](timing_contract.md): timer and observer domains.
- [prior_overlap.md](prior_overlap.md): TEAM, xLLM, Foil, and novelty boundaries.
- [reproduce.md](reproduce.md): actual commands and archive reconstruction.
- [validation.json](validation.json): completed integrity tests.
- [plots/](plots/): editable SVG figures and PNG previews.
- [archive_manifest.json](archive_manifest.json): lossless raw files and failed attempts.
