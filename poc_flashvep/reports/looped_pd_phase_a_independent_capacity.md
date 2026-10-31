# Independent Prefill and Decode Capacity

## Scope and evidence

Status: FINAL_FINITE_CHARACTERIZATION.

This report covers Phase A of the bottleneck crossover study.
Phase B and the conditional Phase C have separate reports.
The study uses real checkpoints and valid KV states.
All generation uses fixed depth and BF16.
The benchmark controls output lengths and ignores EOS.
These results do not describe natural output-length distributions.

The main evidence directory is [looped_pd_bottleneck_crossover](looped_pd_bottleneck_crossover/).
The full matrix is in `capacity_raw.csv` and `capacity_summary.csv`.
Each measured cohort has a unique run directory, source archive, and configuration.
Failed namespaces remain intact.

| Model | Fixed recurrent depth | Unique loaded parameters | Runtime |
|---|---:|---:|---|
| Ouro Base | 4 | 1,434,652,673 | Qualified vLLM-RLT |
| Huginn-0125 | 32 | 3,564,976,800 | Qualified vLLM-RLT |
| Qwen3-1.7B | Not recurrent | 1,720,574,976 | Native vLLM 0.26 |

Qwen is an architecture diagnostic, not a matched-kernel causal control.
Its engine, attention backend, architecture, and numerical policies differ from RLT.
Do not attribute an absolute latency difference to recurrence alone.

## Measurement boundaries

The RLT Prefill phase ends before the first Coda execution.
The first-token phase runs Coda and sampling.
The remaining Decode phase generates exactly `output_cap - 1` tokens.
The native Decode boundary includes first Coda and all remaining Decode work.
Both boundaries appear in the raw data.

A local scheduling barrier preserves the real Prompt KV and hidden state.
It does not fabricate cache contents.
Each timed cohort matches an ordinary native execution with the same input and seed.
Issued recurrent rows must equal the complete fixed-depth work count.

Dedicated Prefill-only runs reserve Prompt KV only.
Their one-token cap is a measurement device, not a replacement Serving workload.
They do not reserve pages for the future Decode output.
They have no remote Decode worker or remote reservation backpressure.

Additional Decode-only runs use the actual PD engine and native NIXL import.
All source computation and incoming transfers finish before their Decode timer starts.
Their outputs match the qualified native reference.
These runs expose coordinator and worker costs that a local engine can hide.

`mu_P` and `mu_D` are finite service rates, in requests/s.
They are not measured maximum sustainable Serving capacity.
Within-start bootstrap intervals do not replace independent-start confidence intervals.

## Generic chunk tuning

The following values are measured finite-cohort Prefill service times per request.
The input contains 3,072 tokens.
The figures summarize the chunk sweep, not a Loop Split comparison.

| Model | C256 | C512 | C1024 | C2048 |
|---|---:|---:|---:|---:|
| Ouro | 1.1641 s | 0.6772 s | 0.4092 s | 0.2907 s |
| Huginn | 2.2624 s | 1.3855 s | 1.0598 s | 1.0236 s |

Matched available output sequences passed the cross-chunk comparison.
See `cross_chunk_output_parity.csv` for the exact denominators.
This finite check does not establish original-HF equivalence or downstream accuracy.

The sweep shows a substantial generic tuning effect.
A method must not claim this effect as a recurrent partitioning gain.
Phase B therefore includes native PD controls with C2048.
The sweep does not include whole-prompt C3500/4096 PD Serving.
Full P-side ownership must not be confused with atomic, unchunked Prefill.

## Decode capacity and engine concurrency

The actual native-PD Decode-only measurements below use Ouro at 3,072 input tokens and 128 output tokens.
They include first Coda, coordinator delivery, and complete Decode.
They exclude all Prefill and NIXL preparation.

| Active requests | Prepared Decode service | Finite completion rate |
|---:|---:|---:|
| 1 | 3.228 s | 0.310 requests/s |
| 2 | 3.333 s | 0.600 requests/s |
| 4 | 3.709 s | 1.078 requests/s |
| 8 | 4.058 s | 1.971 requests/s |

An additional generic engine configuration uses 32 compute slots and 22,528 KV blocks.
Its local cohorts passed full 128-token comparison against the canonical setting.
At 3,500/128, its local B24 rates were approximately 3.109 Prefill requests/s and 4.193 Decode requests/s.
The corresponding actual prepared-PD Decode rate was 3.706 requests/s over 6.476 s.
The difference is a runtime-path effect, not a model approximation.
These measurements are single-start finite cohorts.
Their SLO and actual queue behavior require Phase B validation.

## KV reservation and storage geometry

Ouro's default pool reserves 25,769,803,776 bytes.
Huginn's default pool reserves 49,828,331,520 bytes.
The actual reservation capacity depends on context, depth, and compute slots.
See `kv_geometry_accounting.csv` for each measured geometry.

Huginn allocates 256 layer-depth storage planes.
Its semantically used recurrent and boundary planes total 132.
The difference is 48.4375% of the plane reservation.
This is structural storage accounting, not measured reclaimed memory or speedup.
Native NIXL already excludes unused boundary planes from the relevant transfer path.
The storage difference must not be reported as equivalent wasted network traffic.

The table below derives per-token storage from the measured allocation geometry.
It is structural accounting, not a measured bandwidth or latency result.

| Model | Effective valid planes | Allocated planes | Valid KV bytes/token | Allocated KV bytes/token |
|---|---:|---:|---:|---:|
| Ouro R4 | 96 | 96 | 786,432 | 786,432 |
| Huginn R32 | 132 | 256 | 2,787,840 | 5,406,720 |
| Qwen3-1.7B | 28 | 28 | 114,688 | 114,688 |

Ouro uses 16 KV heads with head dimension 128.
Huginn uses 55 KV heads with head dimension 96.
Qwen uses eight KV heads with head dimension 128.
Shared recurrent weights do not imply a shared KV state across recurrent depths.
Parameter count alone therefore does not predict legal Decode concurrency.
The table does not isolate recurrence from attention architecture or logical layer count.

A 26,624-block Huginn pool actually loaded and completed valid B4 cohorts.
Its memory cost and capacity appear separately from the default pool.
No compact KV layout was implemented in this study.

## Instrumentation and profiling

The event-overhead ablation uses OFF/ON/OFF ordering within one server start.
Ouro B2 Prefill medians were 2.2165 s OFF and 2.2248 s ON.
Huginn B2 medians were 4.7393 s OFF and 4.7631 s ON.
These are within-start observations, not a population overhead bound.

CUDA forward spans include gaps between kernel submissions within a forward call.
They are not exclusive kernel-busy measurements.
CUPTI profiles are stored separately from clean Serving latency.
Heavy tracing substantially increases observed phase duration.
Its idle fraction cannot directly classify an unprofiled server.

The projection diagnostic uses actual checkpoint weights and captured live operands.
It compares prepared 256-row GEMM with native small-row GEMM.
It does not change the candidate runtime.
The Huginn down-projection sample comes from a Prelude module.
It must not be treated as a measured recurrent-core or E2E penalty.

## Correctness and failures

Both Looped models passed an independent 32-request native-PD qualification at 1,536 input tokens.
Both passed a two-request partial-tail qualification at 3,500 input tokens.
The checks include output sequences, valid Prompt KV, final hidden state, and full-depth work.
The initial barrier gates passed eight output and state comparisons per model.
These are internal runtime results, not original-model accuracy results.

The enlarged Ouro setting additionally passed 24 sequences at 3,500 input tokens and 128 output tokens.
Its observed and expected PD recurrent row counts both equal 348,192.
The enlarged Huginn setting passed four sequences at 3,072 input tokens and 128 output tokens.
Its observed and expected PD row counts both equal 409,472.
Both qualifications compare Prompt KV and hidden state with native execution.

The first Huginn chunk sweep failed while allocating its second cache pool.
A local reference retained the previous pool after engine closure.
The harness now releases that reference before allocating the next pool.
The complete three-group retry passed.
This was not a model-capacity failure or an engine-math change.

The first Qwen pilot disabled chunked prefill and emitted an official unsupported-mode warning.
That pilot is excluded from primary evidence.
The supported configuration enables chunked prefill.
Its native NIXL transport gate passed two exact output comparisons.

## Interpretation boundary

The capacity map can identify candidate pressure regimes.
It cannot prove a Prefill bottleneck from a ratio alone.
Phase B must check real queues, Decode KV credit, SLO margin, and clean GPU activity.
Only a verified Prefill-compute-limited regime can authorize Phase C.

The final Phase B review did not establish such a regime beyond generic tuning and admission effects.
It therefore left Phase C unrun.
This does not establish a structural impossibility for Looped models.
Exact 4K/100, longer Ouro contexts, and a faithful Qwen3-8B DuetServe control were not measured.
See the [final report](looped_pd_prefill_decode_bottleneck_final_summary.md) for all completion counts and decisions.
