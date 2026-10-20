"""Write new reports from preserved measurements. Do not modify prior reports."""
import csv
import json
from .contract import ROOT,REVISION,UUIDS
from .analyze import load


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    with (ROOT/'pp_controls_summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (ROOT/'memory_raw.csv').open() as f:memory=list(csv.DictReader(f))
    paired=load(ROOT/'pp_controls_paired.json')
    profiles={arm:load(ROOT/'profiles'/f'clean_mp_correctness_{arm}/summary.json') for arm in ['layer_pp4','loop_pp4']}
    TPprofile=load(ROOT/'profiles/clean2_tp_operator_diagnostic/summary.json')
    performance=table(['C','Runtime','Arm','Makespan s, median [min,max]','Output tok/s','TTFT P50 ms','TPOT P50 ms','E2E P50 ms'],[
        [r['C'],r['runtime'],r['arm'],f"{float(r['makespan_s_median']):.4f} [{float(r['makespan_s_min']):.4f},{float(r['makespan_s_max']):.4f}]",
        *[f'{float(r[k]):.2f}' for k in ['output_tokens_s_median','TTFT_p50_ms_median','TPOT_p50_ms_median','E2E_p50_ms_median']]] for r in summary])
    ratio_table=table(['C','Runtime','Layer / Loop makespan','95% paired bootstrap CI'],[
        [r['C'],r['runtime'],f"{r['layer_over_loop_speedup']:.3f}×",f"[{r['paired_cohort_bootstrap_95CI'][0]:.3f},{r['paired_cohort_bootstrap_95CI'][1]:.3f}]×"] for r in paired])
    memory_table=table(['Runtime','Arm','GPU','Weight GiB','Logical KV GiB','Allocated GiB','Reserved GiB','Peak allocated GiB'],[
        [r['runtime'],r['arm'],r['physical_GPU'],*[f'{int(r[k])/2**30:.4f}' for k in ['weight_bytes','KV_bytes','allocated_bytes','reserved_bytes','peak_allocated_bytes']]]
        for r in memory if r['index']=='1' and r['repeat']=='0'])
    profile_table=table(['Arm','Heavy window ms','Different-request kernel overlap ms','P2P copies','P2P MiB','P2P service sum ms'],[
        [arm,f"{p['window_ms']:.3f}",f"{p['different_request_kernel_overlap_ms']:.3f}",
         next(c['count'] for c in p['copies'] if c['kind']==10),
         f"{next(c['bytes'] for c in p['copies'] if c['kind']==10)/2**20:.6f}",
         f"{next(c['service_ms'] for c in p['copies'] if c['kind']==10):.3f}"] for arm,p in profiles.items()])
    rank_table=table(['Arm','GPU','Kernel service ms','Kernel/copy union ms','No recorded activity %'],[
        [arm,r['physical_GPU'],f"{r['kernel_service_ms']:.3f}",f"{r['activity_union_ms']:.3f}",f"{100*r['no_recorded_activity_fraction']:.2f}"]
        for arm,p in profiles.items() for r in p['ranks']])
    ext=[r for r in load(ROOT/'external_numerical_comparison.json') if r['rank']==0]
    external_table=table(['P/O','Max logit absolute error','Max logit mean error','Max hidden relative L2','Sampled KV relative L2','C2'],[
        [f"{r['P']}/{r['O']}",*[f'{r[k]:.6f}' for k in ['max_logit_abs','max_logit_mean','max_hidden_relative_L2','KV_sample_relative_L2']],r['PASS']] for r in ext])
    limit_table=table(['Quantity','Frozen limit'],[['Logit maximum absolute error',0.75],['Logit mean absolute error per step',0.08],
        ['Hidden relative L2 per step',0.03],['KV relative L2',0.03],['Operator relative L2',0.03],['Stable reference margin',0.5],['Stable argmax mismatches',0]])
    gpu_table=table(['Physical GPU','Logical rank','UUID'],[[i+4,i,'GPU-'+uuid] for i,uuid in enumerate(UUIDS)])
    TPcomm_table=table(['GPU','Actual operator All-Reduce kernels','NCCL kernel-service sum ms'],[
        [r['physical_GPU'],r['NCCL_kernel_count'],f"{r['NCCL_kernel_service_ms']:.6f}"] for r in TPprofile['ranks']])
    apis=table(['Profile','CUDA API','Count','Host service sum ms'],[
        [arm,k,v['count'],f"{v['host_service_ms']:.3f}"] for arm,p in profiles.items() for k,v in p.get('CUDA_API_diagnostics',{}).items()])
    reports={}
    reports['environment.md']=f'''# Environment and provenance

The experiment uses only physical GPUs 4, 5, 6, and 7.
The launcher sets `CUDA_VISIBLE_DEVICES=4,5,6,7` for every GPU job.
TP asserts a four-rank NCCL world and checks each loaded GPU UUID.

{gpu_table}

All four devices are H100 80GB HBM3 GPUs. `nvidia-smi` reports 81,559 MiB per device.
Every permitted pair reports NV18. The driver is 570.211.01.
The raw topology and NVLink status are in `environment_preflight.json`.
The preflight also records jobs on other GPUs. This experiment did not signal those jobs.

| Runtime | Python | PyTorch | Transformers | CUDA build | NCCL |
| --- | --- | --- | --- | --- | --- |
| Custom TP and PP | 3.11 | 2.8.0+cu128 | 4.57.1 | 12.8 | 2.27.3 |
| Isolated native vLLM | 3.14 | 2.11.0+cu129 | 5.16.1 | 12.9 | 2.28.9 |

The custom interpreter is `/home/esjung/.venvs/llada2-mini-reference/bin/python`.
The external interpreter is `/home/esjung/.venvs/cp-composition-vllm026/bin/python`.
No package installation or replacement was necessary.
We set no NCCL algorithm or protocol override.
The custom operator trace identifies `ncclDevKernel_AllReduce_Sum_f32_RING_LL`.

The model and tokenizer use `ByteDance/Ouro-1.4B`, revision `{REVISION}`.
This is the base checkpoint, not the Thinking checkpoint.
The loaded model has 24 physical layers, R4, and 96 independent logical KV slots.
The hidden size is 2048. There are 16 query heads and 16 KV heads.
The head size is 128. The MLP size is 5632.
The model has 1,434,652,673 parameters. All loaded model tensors use BF16.
The weight-file SHA256 is `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.

The base code commit is `addd81ee64d040527aac2b27243a671c59d4e175`.
The branch is `flashvep/ouro-tp4-parallelism-discovery`.
`preflight.json` records all protected source and report hashes.
`commands/*.json` records the source hashes for each execution.
The preservation test checks those protected hashes.

The requested spec path was absent in this worktree.
The full spec existed in `/home/esjung/MLLM-EP-github/poc_flashvep/reports/`.
`spec_frozen.md` preserves that document. `preflight.json` records its SHA256.
This experiment did not modify the main repository.

## Process lifecycle incident

The native vLLM driver initially used `os._exit`, which left four idle worker processes resident.
The cleanup audit linked all four workers to this task's recorded process group and user ID.
Only that process group received a termination signal.
`excluded_runs.json` lists later measurements that could contain this memory-residency confounder.
Those files remain unchanged. We repeated the clean PP controls and profiles after the workers exited.
The launcher now refuses occupied GPUs and cleans its own surviving worker group.
The external driver now requests an engine shutdown.
The user's burner script restarted after GPU tests. See `burner_restore.json` and the final occupancy snapshot.
'''
    reports['implementation.md']='''# Execution and ownership

| Arm | Dense weight ownership | Full KV ownership | Execution unit |
| --- | --- | --- | --- |
| Layer PP4 | Six physical layers per GPU, auxiliary weights replicated | 24 slots per GPU across four loops, all 16 heads | One request, six layers |
| Loop PP4 | All 24 physical layers on every GPU | 24 slots per GPU for one loop, all 16 heads | One request, one complete loop |
| Custom TP4 | QKV/head and SwiGLU shards, auxiliary weights replicated | All 96 slots per GPU, four heads per slot | Four ranks jointly execute one request |
| Native vLLM TP4 | Native TP layers and vocabulary shards | Native paged cache, 96 slots, four local heads | Engine-managed |

Layer PP4 visits GPU4→5→6→7 four times. It needs 15 hidden-state handoffs per full forward.
Loop PP4 visits that path once. It needs three hidden-state handoffs.
Both return the selected scalar token from GPU7 to GPU4 before the next token.
Both preserve the original loop-end norm and gate evaluation.
Neither custom arm implements adaptive exit.

## True TP4

`tp_model.py` reuses the official HF decoder algebra.
It replaces only projection modules and cache storage.
Each Q/K/V weight shard has shape `[512,2048]`.
Each attention-output weight shard has shape `[2048,512]`.
Each gate/up weight shard has shape `[1408,2048]`.
Each MLP-down weight shard has shape `[2048,1408]`.
Attention-output and MLP-down use actual NCCL SUM All-Reduce.
Norms, residuals, embedding, output head, and the unused fixed-depth gate remain replicated.
The custom TP model therefore uses more auxiliary memory than vocabulary-sharded native vLLM.

Weights, KV, and persistent hidden tensors use BF16.
The final diagnostic variant retains GEMM accumulators and row reductions in FP32 before a BF16 cast.
Its collective operands are FP32, not BF16. This is a disclosed implementation difference.
The earlier BF16-reduction variant remains in the execution logs and correctness results.

Each physical layer retains its original parameters across all four loops.
KV index is `24 * loop + physical_layer`.
Each rank owns one contiguous four-head slice at every index.
Prefill uses the entire prompt. Decode uses the exact current causal key length.
Graph scratch never exposes future KV positions to attention.
Owner-local cache staging and graph-input copies remain inside the execution path.
The implementation does not claim a KV-update optimization.

## Common multiprocess PP control

`mp_pp.py` creates one GPU-affine process and one FIFO queue per rank.
CPU queues carry request ID, step, route visit, tensor kind, and readiness metadata.
Startup exchanges persistent CUDA IPC buffer handles.
The receiver copies the hidden tensor directly from the previous GPU into a reusable local buffer.
Serving messages do not carry a hidden tensor through host memory.
Each worker waits for its own CUDA completion event before it publishes the buffer.
The receiving worker can then use the published tensor safely.
No global barrier occurs between stages or loops.
Startup, finalization, and IPC lifetime management use a Gloo group.
Gloo is not a model-compute collective in these PP paths.

The common graph path prepares position-only RoPE constants before admission.
Both placements use the same BF16 decoder, exact-length SDPA, scratch policy, and local publication rule.
Each request owns a separate KV namespace and separate peer export buffers.
No stage merges requests into one tensor batch.
The final scalar argmax still requires host control.
Per-boundary event waits, queues, and IPC copies remain. This is not an optimized production engine.

## Batching boundary

The custom TP serving prototype uses batch-one round-robin admission.
It does not merge requests or simulate PP overlap.
The qualification guard disables its serving benchmark because full C2 qualification failed.
Native vLLM uses different kernels, paged KV, vocabulary sharding, and scheduler code.
The external run uses eager numerical probes only. It is not an optimized serving result.
'''
    reports['correctness.md']=f'''# Correctness and numerical qualification

## Frozen limits

These limits preceded the first GPU probe. No limit changed after a failure.

{limit_table}

Relative L2 is `norm(observed-reference) / norm(reference)`.
Raw files also retain maximum relative errors.
Elementwise relative errors near zero can be large. They are not the primary gate.

## Gate results

| Gate | Scope | Result |
| --- | --- | --- |
| C0 | Protected prior code, physical mapping, two locked legacy R4 cases | PASS with timing drift |
| C1 | Layer-0 attention, MLP, norm, full layer, K/V head slices, all loops and ranks | 192/192 checks PASS |
| C2 short | P128/O8, two prompt streams, aligned numerical and real greedy feedback | PASS in final eager and graph variants |
| C2 longer | P512/O32, full aligned logits/hidden/owner KV | FAIL against frozen internal limits |
| C2 P2048 | Custom implementation | NOT_RUN after P512 failure |
| C3 PP | Four concurrent KV namespaces and physical cross-request kernel overlap | PASS on bounded probes |
| C3 TP | Four-rank sharding, collective participation, short graph replay | PASS on bounded probes, full serving unqualified |

C1 contains eight operator inputs: four loops and prefill/decode positions.
The prefill length is 16. Decode adds position 16 with populated KV.
Each input checks six component values on each rank.
These checks do not prove full-depth numerical equivalence at every prompt length.

The MP correctness probe uses C4/P128/O8.
It duplicates two gold prompt streams into four distinct request namespaces.
Both PP placements match final logits, hidden values, tokens, and every owner KV element bitwise.
This is not four independent benchmark-quality samples.

## Failure and diagnosis

The first BF16-row-reduction probe failed on the second P128 request.
Its maximum hidden relative L2 was 0.034708, above 0.030000.
All eight aligned argmax IDs still matched the reference.

FP32 row accumulation and reduction corrected the short-probe gate failure.
The P512/O32 probe still failed.
Its maximum logit absolute error was 5.210938.
Its maximum per-step logit mean error was 0.802111.
Its maximum hidden relative L2 was 0.469359.
The full owner-KV relative L2 was 0.024168 on rank 0.
All 32 aligned argmax IDs matched on all four ranks.
The largest hidden deviations occurred at steps 18, 19, 26, and 27, using zero-based indices.

An unsharded custom-cache control matched the HF reference bitwise for all 32 P512 steps.
That result does not support a common cache-index or reference-adapter error.
FP32 column accumulation did not remove the long-shape failure.
A diagnostic SDPA head-geometry control reduced the maximum hidden L2 to 0.098947.
It still failed. That diagnostic repeats local heads and is not a performance baseline.
The results implicate numerical execution geometry as a candidate. They do not establish a unique root cause.

Native vLLM also passed the short probe and failed the longer internal limits.
Its bounded free-running outputs matched all 72 reference tokens across three shape probes.
The internal values below summarize rank 0. All four rank records remain available.

{external_table}

This does not prove a model-quality regression or a mathematically incorrect TP decomposition.
It proves that numerical equivalence under this frozen contract remains unresolved.
We did not broaden the limits because tokens happened to agree.
The custom P512 natural-generation lane did not execute after the aligned numerical assertion failed.
The external natural probes permit EOS. All three reached their 8/32/32-token caps.
No GSM8K, MATH, or long-reasoning accuracy claim follows from these capped probes.

## Evidence

- `correctness/*rank*.json`: rank-local component, numerical, KV, and token observations.
- `tp_numerical_diagnostics.csv`: compact rank-0 numerical summary.
- `external_numerical_comparison.json`: all four external ranks and sampled KV comparisons.
- `correctness/unsharded_cache_probe.json`: bitwise unsharded control.
- `logs/`: assertions, failed attempts, and observer failures.

Full native tensor captures remain in the ignored `.tensor_cache/` directory.
`native_reference.json` records their hashes and reproduction source.
The repository contains numerical errors, token arrays, shapes, and raw timing evidence.
It does not commit the multi-GiB redundant tensor dumps.
'''
    reports['benchmark_summary.md']=f'''# Benchmark status and bounded PP controls

The requested TP4-versus-PP serving comparison is **NOT_RUN**.
Full-model C2 numerical qualification failed before that comparison.
`primary_matrix_status.csv` lists all 21 frozen R4 rows and their status.
No TP latency, TPS, crossover, or serving-speedup value is available.

The following measurements are bounded PP-only replication and transport controls.
They are not a qualified three-way architecture benchmark.
All rows use R4, P128, O32, whole prompts, and physical GPUs 4–7.
Each cell summarizes three independent physical cohort repeats.

{performance}

{ratio_table}

`pp_controls_raw.csv` retains all 24 cohort measurements.
`pp_controls_summary.csv` also reports P90 request metrics.
The bootstrap uses 5,000 resamples of paired cohort repeats, not ranks, tokens, or layers.
Three pairs provide weak interval coverage. They do not justify a broad significance claim.
The MP Layer-PP C4 measurements span 0.6289–1.0529 s. This dispersion limits interpretation.

The C4 legacy replication differs from the old 1.8765/1.0621 s comparison.
The new legacy medians are 1.4412/0.8538 s.
We report this drift instead of treating historical absolute latency as a matched baseline.
The old study remains unchanged.

![PP controls](pp_controls.png)

## Memory

These are C4/P128/O32, repeat-0 allocator observations.
They are not a maximum-capacity sweep or an NVML peak measurement.

{memory_table}

The custom TP weight allocation is 1,019,617,282 bytes per rank.
This comes from loaded shard ownership, not a qualified serving run.
Its long-shape graph-workspace peak remains NOT_MEASURED.
Native vLLM reserved a 4 GiB KV pool per rank for the numerical probes.
That configurable pool is not a measured minimum KV requirement.

Logical KV, graph scratch, allocator reserve, and total driver memory are different quantities.
`memory_raw.csv` records scratch and IPC export storage separately where available.
The scratch sum does not include every CUDA Graph private allocation.
CUDA IPC can create peer contexts in several processes. NVML can therefore show more process entries than model owners.
We make no maximum-concurrency, OOM-boundary, or serving-capacity claim.
'''
    reports['transport_control.md']=f'''# Common PP transport control

The common control removes the central Python scheduler at intermediate boundaries.
Each GPU has an independent process, FIFO queue, stream, and reusable IPC buffer bank.
Both placements use this same path.
It does not remove local completion waits, graph KV staging, scalar token control, or pipeline dependencies.

At C4/P128/O32, the legacy Layer/Loop median ratio is 1.688×.
The four-process ratio is 1.166×.
The corresponding median makespans change from 1.4412/0.8538 s to 0.6326/0.5424 s.
At C1, the median ratio changes from 1.480× to 1.252×.
The MP implementation does not improve every absolute latency. C1 setup/publication effects remain visible.
Legacy warm-up executes a complete request. MP warm-up executes local graph stages before the first cohort.
The MP C1 clock can therefore include first-use queue or peer-copy effects.
C4 follows a complete C1 cohort in each MP process, but this is not an identical end-to-end warm-up protocol.
The control also changes position-constant preparation and buffer reuse.
These controls show an important implementation contribution to the old PP gap.
They do not isolate one exclusive host-cost component or prove an architecture advantage over TP4.

## Physical overlap and transport

These separate heavy runs use C4/P128/O8 and correctness observers.
They do not supply clean serving latency.

{profile_table}

The P2P counts include 28 scalar token returns in each arm.
Hidden transfers account for 480 versus 96 copies.
Their hidden payloads are 33,177,600 versus 6,635,520 bytes.
Scalar token returns add 224 bytes in each arm.
All payloads match runtime receive counters.
The observer also copies final hidden/logit values to CPU for comparison.
Those D2H copies are diagnostics, not intermediate hidden-state transport.

The hidden byte count is `C * (P + O - 1) * 2048 * 2 * handoffs`.
The handoff count is 15 for Layer PP and three for Loop PP.
This exact accounting does not convert byte reduction into latency reduction.
P2P kernel-service sums cannot establish exposed E2E communication cost.

{rank_table}

The last column is the complement of recorded kernel/copy intervals in the heavy window.
It mixes pipeline dependencies, host gaps, waiting, and observer effects.
It is not clean GPU utilization or a measured architectural bubble fraction.
Stage readiness, dispatch, safe publication, and CUDA event brackets remain in each traced rank JSON.
These local brackets do not establish an exclusive E2E critical-path decomposition.

{apis}

These API times are cumulative host service, not additive E2E shares.
The heavy lane has extra correctness copies and events.
It is not shape-matched to the O32 clean control. A quantitative observer-overhead ratio remains NOT_AVAILABLE.
Both unprofiled and profiled C4/O8 correctness runs pass the same token and full-owner-KV checks.

![Layer PP physical kernels](profiles/clean_mp_correctness_layer_pp4/kernel_timeline.png)

![Loop PP physical kernels](profiles/clean_mp_correctness_loop_pp4/kernel_timeline.png)
'''
    reports['tp4_external_baseline.md']=f'''# Native vLLM TP4 attempt

The installed isolated environment contains vLLM 0.26.0 with native Ouro support.
The [official Ouro project](https://ouro-llm.github.io/) identifies this release as the last native Ouro release.
The environment required no package replacement.
The first attempt failed because the child PATH did not include the installed `ninja` executable.
The retry used the interpreter's existing bin directory. It completed the numerical captures.

The native model has 24 physical layers and 96 separate attention/KV slots.
All four workers own four local attention/KV heads and execute target-model compute.
`external_ownership.json` records UUIDs, weights, cache shapes, and storage.
The native model uses actual TP collectives. It is not a replicated-model substitute.

## Execution contract

- BF16, fixed R4, TP4, EP disabled, PP1, DP1.
- Maximum model length 4096 and KV pool 4 GiB per rank.
- Whole-prompt prefill, no chunked prefill or prefix caching.
- Eager execution for numerical observation, not an optimized CUDA Graph serving benchmark.
- FlashAttention 3, native paged KV and fused operations.
- Native vocabulary sharding, unlike the custom replicated output head.
- Separate PyTorch, Transformers, NCCL, scheduler, and kernel versions.

The engine initialization took {load(ROOT/'external_setup.json')['wall_s']:.3f} s.
That value is setup time, not request latency.

{external_table}

The sampled KV comparison covers all 96 slots and three positions: 0, P−1, and P+O−2.
It does not compare every stored external KV element.
Free-running greedy IDs match 8/8, 32/32, and 32/32 tokens on the three shape probes.
All three outputs stop at their length cap.
These are bounded numerical probes, not reasoning accuracy or serving performance.

The external serving matrix is NOT_RUN because the longer numerical comparisons fail the frozen gate.
Consequently, there is no production-oriented TP4-versus-PP performance result.
The eager observer result must not be called an optimized TP latency baseline.
Native vLLM's internal differences also preclude an engine-matched claim.
'''
    reports['timing_contract.md']='''# Timing and evidence contract

## Evidence classes

LIVE_MEASURED_PP_ONLY_CONTROL means actual clean PP execution with pretrained weights on the four permitted GPUs.
DIAGNOSTIC_NSIGHT_ACTIVITY means kernels and copies from a separate heavy execution.
NUMERICAL_OBSERVER means actual values with potentially intrusive output copies.
STRUCTURAL means a shape-derived quantity or an execution counter, not a measured network rate.
NOT_RUN means the experiment did not execute.

## Clean clocks

All requests enter as one finite cohort. There is no prompt chunking or tensor batching across requests.
Makespan includes prefill, decode, hidden copies, head work, queueing, publication waits, and scalar token control.
The MP origin is rank 0's admission timestamp on the shared host monotonic clock.
Its endpoint is the final token publication at rank 3.
Worker-stop messages and teardown are outside that endpoint.
The legacy result uses the preserved controller clock. Its stop overhead is small but not identically defined.
That clock difference limits exact legacy-versus-MP causal attribution.

TTFT ends at first token publication and includes admission queueing.
E2E ends at each request's last token publication.
TPOT is `(last_token_time-first_token_time)/(O-1)` for each request.
P50 and P90 refer to requests inside one cohort.
Reported cells are medians of three physical repeats.
Output tokens/s is `C*O/makespan`. Requests/s is `C/makespan`.

Weights, communicator setup, graph capture, buffer setup, and warm-up precede admission.
Raw files record preparation time. All qualified PP controls record zero graph captures inside the serving clock.
Legacy warm-up runs a complete request. MP initially warms local graph stages, not the entire queue/peer path.
First-use queue or peer setup can remain in the MP C1 clock. C4 follows the complete C1 cohort.
This warm-up difference prevents a strict engine-matched comparison with legacy serving latency.
The clean lane records one completion event per stage for safe publication.
It has no per-layer timing events and no Nsight profiler.
The cold process startup time is not the serving clock.

## Output control

The frozen workload stores exact input and feedback token IDs.
The MP hardware control computes the real head and argmax, then uses the frozen token as the next input.
Every computed argmax in the qualified MP controls also matches that frozen stream.
The legacy C0 replication uses live greedy feedback. Its observed IDs match the frozen output stream on every measured request.
These controls therefore execute identical token sequences, but their feedback-control code is not byte-identical.
The separate PP numerical lane uses real free-running feedback.
The TP numerical lane separates aligned comparisons from free-running capped generation.

## Profiler clock

Nsight supplies the common timeline for physical cross-rank kernel overlap.
We do not compare independent CUDA event timestamps across GPUs.
Per-stage CUDA brackets measure local elapsed intervals only.
Stage metadata records host readiness, dispatch, and safe publication separately.
Each receiver reads a tensor only after its producer's completion event permits publication.

Kernel-service sums, activity unions, and clean wall times remain separate.
An All-Reduce kernel can include another rank's readiness delay.
Its service time is not pure payload movement or an exclusive E2E cost.
We report no true bandwidth or additive critical-path breakdown.

## Statistical limits

Paired bootstrap resamples complete physical cohort pairs 5,000 times.
The sampling unit is not a token, layer, loop, or rank.
Only three pairs exist per bounded PP case. Confidence intervals remain exploratory.
No task-level quality confidence interval exists because no accuracy benchmark ran.
'''
    reports['analysis.md']=f'''# Analysis and final verdict

## Verdict: BLOCKED

The full TP4 serving comparison remains blocked by numerical qualification gate C2.
The experiment does not establish that Loop PP4 is faster or slower than optimized TP4.
It also does not establish a general architectural NO-GO.

We implemented and executed actual four-rank TP4.
It shards QKV heads and SwiGLU tensors and performs actual NCCL All-Reduce.
All 192 bounded component checks pass. Short full-model eager and graph probes also pass.
Longer full-depth hidden/logit values exceed the frozen limits, although observed token IDs match.
Native vLLM reproduces this distinction between token agreement and internal numerical equivalence.
We therefore preserve the failure and do not publish unqualified TP latency.

## What the valid PP control shows

At C4/P128/O32, the common four-process path narrows the Layer/Loop median ratio from 1.688× to 1.166×.
This supports a substantial runtime-implementation contribution to the old PP advantage.
It does not show that all remaining cost is architectural.
The three Layer-PP repeats vary widely. The residual advantage needs more samples and a qualified TP baseline.
Warm-up protocols and position/buffer preparation also differ between legacy and MP controls.
The ratio change does not isolate the effect of multiprocessing alone.

Both MP placements preserve bitwise owner KV and tested natural greedy outputs.
Both show different-request GPU kernel overlap on four physical GPUs.
Their transport uses direct peer copies, not hidden-state host staging.
Their process-local FIFO queues remove intermediate central-controller/GIL scheduling.
Local publication waits and per-stage graph/layout work remain.

{ratio_table}

## Physical communication evidence

{profile_table}

The heavy PP traces have nearly equal per-rank kernel-service totals.
Their P2P service difference is much smaller than the historical wall-time difference.
Transfer count alone cannot explain serving latency.
The current heavy windows also include correctness observers and cannot replace clean clocks.

The bounded TP operator trace contains the following actual collectives:

{TPcomm_table}

These 32 reductions per rank come from eight component inputs, not one complete model forward.
The four ranks jointly produce 128 recorded All-Reduce kernels.
The kernel name identifies FP32 SUM, RING, and LL.
Python call logs retain each operation's actual tensor shape and logical operand bytes.
The nominal 192 reductions per full R4 core are visible in short graph capture logs.
They are not a measured serving collective profile.
The operator observer has substantial rank-readiness skew. Its NCCL service sums are not production link costs.

## Interpretation and decision scores

| Question | Result |
| --- | --- |
| True TP4 implementation and four-GPU participation | Demonstrated on operators and bounded model probes |
| Full-model numerical qualification | FAIL against frozen limits at P512 |
| Common multiprocess PP control | Implemented, bitwise bounded correctness, physical overlap |
| Complete 21-row three-way serving comparison | NOT_RUN |
| Optimized external TP4 serving baseline | NOT_RUN, native eager numerical diagnosis only |
| Loop PP advantage over TP4 | UNRESOLVED |
| Novel loop-specific execution mechanism | NOT_ESTABLISHED |
| Dynamic-depth follow-up | NOT_RUN, not justified by this result |

The smallest next action is a numerical-qualification study, not a scheduler implementation.
It should compare per-layer FP32 reference errors across the existing BF16 single-GPU and TP backends.
It must define any revised acceptance criterion prospectively on independent calibration inputs.
It must not retroactively convert this failed frozen gate into PASS.
If that study qualifies TP4, the preserved 21-row workload can support a fresh matched serving campaign.

## Limitations

The final verdict is a contract gate, not proof that correct TP inference is impossible.
Bounded token agreement is not benchmark accuracy or arbitrary-length generation parity.
The external engine differs in kernels, batching, cache layout, and software versions.
Only two short PP hardware shapes received three paired repeats.
No exclusive E2E attribution or quantitative observer correction is available.
We did not test a maximum-concurrency or HBM capacity boundary.
Excluded runs and failed attempts remain preserved and explicitly identified.

See [benchmark_summary.md](benchmark_summary.md), [correctness.md](correctness.md), and [transport_control.md](transport_control.md).
'''
    reports['reproduce.md']='''# Reproduction

Run commands from `/home/esjung/MLLM-EP-ouro-request-pp`.
Use the pinned environments. Do not install packages into another experiment's environment.
Every GPU launcher requires physical GPUs 4–7 to be free.
Stop only recognized project burners. Never signal an unrelated process.
Use a new label for each run. Do not overwrite preserved evidence.

## Inspect and check without GPU execution

```bash
git status --short
/home/esjung/.venvs/llada2-mini-reference/bin/python -m unittest poc_flashvep.ouro_tp4.test_evidence -v
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.analyze
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.make_reports
```

Analysis reads existing raw results. It writes only this new report directory.
Do not rerun `prepare.py` over the frozen preflight.

## GPU lifecycle

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.gpu_lease status
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.gpu_lease stop
```

The wrapper refuses to stop an unrecognized GPU job.
After experiments, restore the user's burner script:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.gpu_lease restore
```

## Exact recorded commands

`commands/*.json` contains each argv, cwd, selected environment variables, and source SHA256.
The paired completion file records the driver return code and process ID.
`logs/*.log` contains stdout, stderr, and failed assertions.
These records are authoritative for variants that changed during diagnosis.
The current TP source is the final FP32-accumulator diagnostic variant.
It must not be described as the exact source of the earlier BF16-reduction attempt.

To run a new component check, use a new label:

```python
from poc_flashvep.ouro_tp4.launch import tp
tp('new_operator_check', ['--mode', 'operators'])
```

To run a new short numerical check:

```python
from poc_flashvep.ouro_tp4.launch import tp
tp('new_short_check', ['--mode', 'correctness', '--smoke'])
```

To repeat a bounded PP control:

```python
from poc_flashvep.ouro_tp4.launch import execute, PYTHON
execute('new_layer_control', [PYTHON, '-u', '-m', 'poc_flashvep.ouro_tp4.mp_pp',
    '--arm', 'layer_pp4', '--indices', '0,1', '--label', 'new_layer_control'])
```

Use `loop_pp4` with the same settings for the other placement.
`--correctness` selects the C4/P128/O8 real-feedback probe instead of the hardware control.
The launcher supplies `CUDA_VISIBLE_DEVICES=4,5,6,7`.
It sets OMP/OpenBLAS thread counts to one and offline model access.
Do not use the unqualified TP serving benchmark. Its guard refuses the current failed C2 evidence.

## Profiler reproduction

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.profile_diagnostics --prefix new_
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_flashvep.ouro_tp4.profile_analysis
```

Nsight exports SQLite locally. The committed `.nsys-rep` files can recreate those exports.
Use `--kill=none` to prevent profiler shutdown from terminating distributed workers after capture.
The `clean_tp_operator_diagnostic` attempt exited nonzero after all operator checks passed.
`clean2_tp_operator_diagnostic` repeats that check and exits normally.
Only the latter supplies the final TP operator timeline.

Native vLLM numerical reproduction uses `external_fidelity.py` through the isolated interpreter and scoped launcher.
Its initialization parameters remain in `external_engine_contract.json`.
It needs the local reference tensors from `native_reference.py`.
It does not run an optimized serving benchmark.
'''
    reports['status.md']='''# Completion status

Verdict: BLOCKED at C2 full-model numerical qualification.
This is the specified stop condition, not a completed three-way serving campaign.

## Implemented and physically tested

- Four-process NCCL TP4 with actual weight/head shards and full depth-indexed KV.
- Operator and short-model numerical checks, including CUDA Graph replay.
- Longer aligned numerical diagnosis and an unsharded bitwise cache control.
- Common four-process Layer PP and Loop PP with persistent CUDA IPC transport.
- Bitwise bounded PP correctness and physical different-request kernel overlap.
- Twenty-four qualified PP-only cohort measurements across two shapes and two runtime paths.
- Separate native vLLM TP4 numerical and capped natural-generation probes.
- Nsight captures, CUDA API counts, communication accounting, and allocator memory records.
- Fifteen CPU evidence tests. See `tests.json` for the actual test run.

## Failed attempts retained

- BF16 row reduction: short hidden-state tolerance failure.
- FP32 row/column controls: longer hidden/logit tolerance failure.
- Attention head-geometry diagnostic: reduced error but still failed.
- Native vLLM first launch: child PATH could not find existing ninja.
- Native TP4 longer internal values: frozen gate failure despite token agreement.
- Orphan worker incident: later control runs excluded and repeated after cleanup.
- First clean TP Nsight attempt: profiler shutdown interrupted worker teardown. The retry completed.

## NOT_RUN

The qualified 21-row TP4-versus-PP serving matrix and optimized graph-mode vLLM serving did not run.
DP4 context, dynamic-depth runtime, new adaptive scheduler, capacity sweep, and task accuracy benchmarks did not run.
No R2/R3 sweep ran.

The source base commit is in `preflight.json`.
The result commit is the commit that adds this status file and the scoped new directories.
Use `git log -1 --format=%H -- poc_flashvep/reports/ouro_tp4_parallelism_discovery/status.md` to obtain it.
The final handoff identifies the exact result commit.
Only this PoC's files enter the new commit.
The user's final instruction authorizes push. `preflight.json` records that interpretation.
'''
    for name,text in reports.items():(ROOT/name).write_text(text)
    print('REPORTS_WRITTEN',len(reports),flush=True)


if __name__=='__main__':main()
