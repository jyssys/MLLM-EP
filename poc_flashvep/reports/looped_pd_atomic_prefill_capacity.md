# Atomic Prefill PD: Independent Service Measurements

Status: measured capacity checkpoint. The online crossover search is still in progress.
Evidence type: MEASURED_FINITE_INDEPENDENT_SERVICE.
These service rates are not sustainable open-loop capacities.
They do not establish a Prefill-heavy regime by themselves.

## Measurement boundaries

Each reported rate is the mean of three repeated cohorts within one process start.
An independent-start confidence interval is not available for these capacity samples.
The complete batch curves remain in `looped_pd_atomic_prefill_qps_crossover/capacity_samples.csv` and `capacity_summary.csv`.
Those files preserve blocked batches and preparation methods.

RLT P-only timing ends before first Coda sampling.
RLT D-only timing starts from valid resident KV and includes first Coda sampling.
The barrier preserves the actual model state and verifies exact output against the unbarriered local runtime.
It does not fabricate KV or reuse another request's activations.
Native PD adds transport and cooperative worker progress that the local service timer does not include.

Native Qwen P timing includes its first sample.
Native Qwen prepared-D timing excludes preparation, cached-suffix recomputation, and first sampling.
The native PD path discards the P sample and samples on D from imported KV.
These boundary differences remain explicit. The P and D rates are not an additive E2E model.

## P service references

Units: requests/s. Values are finite within-start means.
The selected concurrency is the smallest tested batch within 95% of that chunk's best finite rate.

| Model | Input tokens | Full-Prompt rate | Selected P concurrency | Strongest selected C | Strongest selected rate | Best non-Full C available |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 3900 | 10.3589 | 1 | 3900 | 10.3589 | Not measured separately |
| Qwen3-8B | 8000 | 4.5732 | 1 | 8000 | 4.5732 | 1024 |
| Ouro R4 | 4096 | 4.5748 | 4 | 4096 | 4.5748 | Not measured separately |
| Ouro R4 | 8192 | 2.1331 | 4 | 4096 | 2.1905 | 4096 |
| Huginn R32 | 3500 | 0.8231 | 1 | 3500 | 0.8231 | Not measured separately |
| Huginn R32 | 3900 | 0.7362 | 1 | 2048 | 0.7556 | 2048 |

The best isolated P chunk need not be the best Serving chunk.
The Serving screen therefore tests additional qualified C settings.
It keeps wire-credit controls separate from token-chunk controls.

## Prepared D references

Units: requests/s. Each cell is the strongest measured legal finite cohort for that output cap.
The batch column is essential. A large all-ready batch may not form with staggered online arrivals.

| Model | Input tokens | Output cap | D batch | Finite D rate | Strongest P / finite D |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 3900 | 32 | 16 | 38.0846 | 0.272 |
| Qwen3-8B | 3900 | 64 | 16 | 18.2469 | 0.568 |
| Qwen3-8B | 3900 | 100 | 16 | 11.6238 | 0.891 |
| Qwen3-8B | 8000 | 100 | 32 | 13.4841 | 0.339 |
| Qwen3-8B | 8000 | 200 | 32 | 6.6554 | 0.687 |
| Ouro R4 | 4096 | 64 | 8 | 4.0653 | 1.125 |
| Ouro R4 | 4096 | 100 | 8 | 2.5845 | 1.770 |
| Ouro R4 | 8192 | 32 | 8 | 6.4635 | 0.339 |
| Ouro R4 | 8192 | 64 | 8 | 3.1815 | 0.689 |
| Ouro R4 | 8192 | 100 | 8 | 2.0276 | 1.080 |
| Ouro R4 | 8192 | 200 | 8 | 1.0059 | 2.178 |
| Huginn R32 | 3500 | 32 | 3 | 0.8864 | 0.929 |
| Huginn R32 | 3500 | 64 | 3 | 0.4360 | 1.888 |
| Huginn R32 | 3500 | 100 | 3 | 0.2772 | 2.969 |
| Huginn R32 | 3900 | 32 | 3 | 0.8664 | 0.872 |
| Huginn R32 | 3900 | 64 | 3 | 0.4268 | 1.771 |
| Huginn R32 | 3900 | 100 | 3 | 0.2716 | 2.782 |

Qwen 8000-token B32 uses the official resident-prefix preparation control.
Preparation runs sequentially outside the timed D interval.
Each actual cache hit must retain at least `input_length - block_size` tokens.
This avoids a large B32 Prefill activation peak in a Decode-capacity experiment.
Main Serving still disables prefix caching and uses NIXL import.
The preparation families are not pooled into one average.

Huginn B3 is measured, not interpolated between B2 and B4.
The measured 3900/32 B2 rate is approximately 0.6413 requests/s.
B3 improves the isolated rate, but three D reservations can leave no room for another P request's reserved D state.
The finite B3 result therefore does not certify simultaneous P/D operation at that rate.

## KV geometry and memory limits

Evidence type: MEASURED_ALLOCATION plus STRUCTURAL_ACCOUNTING.

| Model | Actual KV pool bytes per GPU | Pool blocks | Physical layers per storage depth | Storage depths | KV heads | Head dimension |
|---|---:|---:|---:|---:|---:|---:|
| Ouro R4 | 70,866,960,384 | 22528 | 24 | 4 | 16 | 128 |
| Huginn R32 | 66,437,775,360 | 24576 | 8 | 32 | 55 | 96 |

Huginn allocates 256 physical-layer/depth storage planes.
Its valid full-depth model structure uses 132 logical Attention layers.
The difference is 124 planes, or 48.4375% of the uniform allocation.
The allocated geometry corresponds to 5,406,720 KV bytes per token before block rounding.
The valid 132-layer geometry corresponds to 2,787,840 bytes per token.
These are storage calculations, not measured speedups from compaction.
This PoC does not implement compact KV storage.

Actual native handoff payload is separate from reserved HBM.
Measured 16-request calibration totals give the following payload per request.

| Model/input | Measured payload bytes/request | Measured fragment completions/request |
|---|---:|---:|
| Ouro 8192 | 6,442,455,040 | 97 |
| Huginn 3900 | 10,872,554,880 | 163 |
| Huginn 3500 | 9,757,418,880 | 146 |

Those payloads include the actual valid transfer state and hidden state.
Do not equate reserved invalid storage planes with transferred valid bytes.
The heavy byte checks compare valid KV and final hidden state, not uninitialized storage.

## Interpretation limits

Ouro 8192/32 and 8192/64 are promising independent-rate candidates.
That statement is not a measured online bottleneck classification.
Huginn's 32-token D-only rate is closer to P service than its 100-token rate.
KV reservation coupling can still prevent the larger D batch from forming online.
Qwen's longer-output exact parity fails across arrival rates.
Its finite service samples remain executed measurements, but they do not qualify its blocked Serving comparison.

Actual P queue growth, P kernel activity, D readiness, KV credit, and frozen Decode SLOs remain necessary.
The final report will separate those online results from the independent references above.
