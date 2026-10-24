"""Build concise reports from retained measurements. Never fill absent results."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .analyze import confidence

def read(name):return list(csv.DictReader((ROOT/name).open()))
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def put(name,text):(ROOT/name).write_text(text.strip()+'\n')
def main():
    baseline=read('baseline_summary.csv');ops=json.loads((ROOT/'operator_summary.json').read_text())
    raw=read('attention_operator_pairs.csv');pairs=read('paired_comparisons.csv');local=read('local_segmentation.csv')
    counts=json.loads((ROOT/'analysis_counts.json').read_text())
    windows=json.loads((ROOT/'profile_complete_token_windows.json').read_text())
    b1=[r for r in baseline if int(r['N'])==1]
    baseline_table=table(['Prompt tokens','Starts','Mean TTFT, ms','Mean delivered ITL, ms','Logical KV, GiB','Peak allocated, GiB'],
        [[r['P'],r['starts'],f"{float(r['TTFT_mean_ms']):.2f}",f"{float(r['ITL_mean_ms']):.3f}",f"{float(r['logical_KV_GiB']):.3f}",f"{float(r['peak_GiB']):.3f}"] for r in b1])
    trace=[]
    for name in sorted({r['run'] for r in windows}):
        rows=[r for r in windows if r['run']==name and r['complete_R4_geometry']]
        if not rows:continue
        a=np.mean([r['attention_exclusive_GPU_service_ms'] for r in rows]);t=np.mean([r['output_completion_interval_ms'] for r in rows])
        trace.append([name,len(rows),f'{a:.3f}',f'{t:.3f}',f'{a/t*100:.2f}%'])
    trace_table=table(['Heavy trace','Complete R4 windows','Attention exclusive GPU service, ms','Sampling interval, ms','Trace-local ratio'],trace)
    op_rows=[]
    for r in ops:
        if r['B']!=1 or r['placement']!='balanced':continue
        ci=r['speedup_wall_95CI']
        op_rows.append([r['L'],r['P'],'pressure' if r['cache_pressure'] else 'repeated KV',r['independent_starts'],
            f"{r['local_wall_ms']:.4f}",f"{r['distributed_wall_ms']:.4f}",f"{r['speedup_wall']:.3f}× [{ci[0]:.3f},{ci[1]:.3f}]"])
    op_table=table(['Visible keys','GPUs','Cache condition','Process starts','Local graph wall, ms','Distributed graph wall, ms','Operator speedup, 95% CI'],op_rows)
    layout_rows=[]
    for p,n,lane in sorted({(int(r['P']),int(r['N']),r['lane']) for r in pairs}):
        rs=[r for r in pairs if (int(r['P']),int(r['N']),r['lane'])==(p,n,lane)]
        values=[float(r['speedup']) for r in rs];ci=confidence(values)
        layout_rows.append([p,n,lane,len(values),f"{np.mean([float(r['native_ITL_ms']) for r in rs]):.3f}",
            f"{np.mean([float(r['layer_major_ITL_ms']) for r in rs]):.3f}",f'{np.mean(values):.3f}× [{ci[0]:.3f},{ci[1]:.3f}]'])
    layout_table=table(['Prompt','B','Lane','Start pairs','Native ITL, ms','Layer-major ITL, ms','Speedup, 95% CI'],layout_rows)
    local_rows=[]
    for length,batch,pressure in sorted({(int(r['L']),int(r['B']),r['cache_pressure']) for r in local}):
        choices=[]
        for split in (2,4,8,16):
            rs=[r for r in local if (int(r['L']),int(r['B']),r['cache_pressure'],int(r['S']))==(length,batch,pressure,split)]
            if rs:choices.append((np.mean([float(r['segmented_wall_ms']) for r in rs]),split,rs))
        if not choices:continue
        latency,split,rs=min(choices,key=lambda x:x[0])
        local_rows.append([length,batch,pressure,split,len(rs),f'{latency:.4f}',f"{np.mean([float(r['speedup_wall']) for r in rs]):.3f}×"])
    local_table=table(['Visible keys','B','128 MiB pressure','Best measured local S','Process starts','Local segmented wall, ms','Native S1 / segmented'],local_rows)
    numeric_max=max(float(r['max_relative_L2']) for r in raw) if raw else None
    abs_max=max(float(r['max_absolute']) for r in raw) if raw else None
    put('baseline_audit.md',f'''# Native baseline audit

The checkpoint, source, and physical topology are pinned in `source_lock.json`.
`source_audit.md` lists the actual cache and attention call sites.
The model executes 24 shared layers four times.
The cache contains 96 logical layer/depth namespaces.

The baseline is native RLT Async/Mixed with FA4 and CUDA Graphs.
It uses a common 18,432-block physical pool and 16-token pages.
The allocator keeps all depth planes in HBM.
Prefix caching is disabled.
Each measured request therefore performs its full prefill again.

{baseline_table}

Evidence: `LIVE_MEASURED_SERVING`, bounded hardware lane.
The output cap is 64 tokens, and EOS is ignored in this table.
TTFT includes request prefill in a resident, warmed engine.
It excludes model loading and deployment startup.
ITL uses actual frontend-delivered output tokens, not recurrent callbacks.
The two starts use disjoint corpus families and are not paired quality samples.
`baseline_summary.csv` contains all tested B1/B2/B4/B8 cells.
''')
    put('kv_memory_accounting.md',f'''# Exact KV memory accounting

The model has 1,434,652,673 parameters.
Unique weight storage is 2,869,305,346 bytes.
BF16 K and V require `2 × 2 × 24 × 4 × 16 × 128` bytes per context token.
The result is 786,432 bytes, or 0.75 MiB per token per sequence.
One physical pool block stores 16 positions for all 24 physical layers.
It occupies 3 MiB across K and V.
The measured 18,432-block pool occupies 54 GiB total.
It already includes all four depths.

{baseline_table}

B1/16K has 12 GiB of logical KV.
B1/32K has 24 GiB.
B1/65,408 has 47.906 GiB.
B2 at that last length needs 95.812 GiB before weights and workspace.
That payload cannot fit on one H100 in this experiment.
B4/32K and B8/16K each need 96 GiB before other allocations.
These are payload constraints, not allocator-fragmentation claims.

Small B/L states leave much of the common pool empty.
Reducing that reservation does not compress valid KV.
It also reduces the number of requests that the pool can admit.
The size-aware pool control charges this capacity trade-off.

`memory_events.csv.gz` distinguishes actual allocator bytes from issued written-prefix metadata.
Async metadata can precede producer completion.
It is not a hardware traffic measurement.
The fixture capture reads initialized pages in the producing CUDA stream.
`capacity_admission.json` records actual allocator acceptance and first rejection.
Those empty reservations are not completed concurrent inference.
The successful grid records the largest completed batch within the tested sweep.
''')
    put('single_gpu_critical_path.md',f'''# Single-GPU critical path

## Clean bounded hardware inference

{baseline_table}

## Separate heavy GPU trace

{trace_table}

Each retained window spans consecutive GPU sampling completions.
It contains 96 actual attention kernels.
Partial capture boundaries do not enter this table.
Attention-exclusive GPU service excludes overlap with other recorded kernels.
The ratio uses the same heavy-trace sampling interval.
It is not a clean host E2E fraction.

Long context creates a substantial measured attention interval.
The old Thinking-model eager result of approximately 7.3% does not describe this native Base-model configuration.
The clean ITL also increases with context.
These observations support long-context attention diagnosis.
They do not establish an integrated distributed speedup.

`profile_kernel_families.json` retains GEMM, KV-write, metadata, copy, and pointwise service.
QKV and FFN GEMMs remain unresolved within their combined family.
Pointwise kernels also include normalization, RoPE, and residual work.
No exclusive clean FFN percentage is claimed.
The trace shows many small kernels even though CUDA Graphs reduce CPU submission count.
Kernel count alone is not removable E2E time.

`counters/` retains the absolute-path Nsight Compute attempt.
Unavailable HBM counters remain missing.
Logical KV bytes divided by latency are not achieved DRAM bandwidth.

For an interval fraction f, the conditional Amdahl speedup is `1 / (1-f+f/s)`.
Using trace-local f gives a trace-based model only.
It does not measure clean request improvement.
Cold request E2E also contains prefill, which is outside the decode-only operator.
''')
    put('single_gpu_layout.md',f'''# Exact single-GPU controls

The native cache stores physical layers inside each page block.
Its per-layer view has a 1.5 MiB block stride for K alone.
The layer-major control reduces that stride to 64 KiB.
It preserves the public tensor axes and depth-specific page ownership.
It does not reduce valid KV payload or model execution depth.

{layout_table}

The confidence interval resamples independent matched process-start pairs.
Technical repeats and model layers are not independent statistical samples.
Quiet pairs use the same physical GPU within each comparison.
The parallel discovery lane remains separate.

The dedicated initialized-page test compares three physical layers at all four depths.
Its twelve sampled Q/K/V/attention snapshots are byte-exact across layouts.
The corresponding two-token hardware outputs and depth arrays also match.
`layout_correctness.json` retains every comparison.
This limited probe is not benchmark quality evidence.

## Strong generic local attention

Pinned FA4 rejects internal SplitKV on SM90.
The study preserves that failure and tests a generic alternative.
It exposes context-page segments as extra FA4 batch rows.
The segments share the original K/V storage without a KV repack.
Query replication and the stable merge execute inside the measured CUDA Graph.
This is a Flash-Decoding-style control, not a loop-aware method.

{local_table}

The best S is a measured tuning envelope over S2/S4/S8/S16.
It is not a validated online controller.
The control uses the same source fixtures and registered numerical tolerances.
It tests whether CP gains reflect missing generic local sequence parallelism.
No full-model integration or accuracy gain is inferred from this operator.
''')
    put('distributed_attention_correctness.md',f'''# Distributed attention correctness

The tested groups contain two, four, or eight physical GPUs.
Each rank has its own CUDA context and NCCL process group.
Rank identity files record physical GPU IDs and UUIDs.
The rank-local valid K/V hashes match original model-derived BF16 values.
Each valid context position belongs to exactly one rank.
The final partial page remains length-masked.

The operator uses actual query broadcast and output/LSE all-gather.
Each rank executes optimized FA4 only over its local KV.
The stable merge combines normalized local outputs through logsumexp weights.
The query has one causal position.
Every retained prefix key is visible to that position.
This validates decode attention, not multi-query prefill masks.

Completed paired operator rows: {len(raw)}.
Maximum recorded relative L2: {numeric_max}.
Maximum recorded absolute error: {abs_max}.
The registered limits are 0.03 and 0.125, respectively.
The limits did not change after measurement.
Both eager and captured distributed outputs enter the error maximum.

The main matrix samples four depths and physical layers 0, 12, and 23.
It does not run all 24 layers as a distributed model.
Batch B uses B copies of one real captured query/KV state.
It measures operator geometry, not independent natural serving requests.
The fixtures come from original pretrained forward execution, not random weights.

The edge campaign tests zero-length local shards and partial pages.
Its cropped prefixes are primitive correctness tests.
They are not model quality evaluations.
All-empty global attention is outside the model's valid causal state.
No attention or KV approximation is used.
''')
    put('distributed_scaling.md',f'''# Physical distributed scaling

{op_table}

Evidence: `LIVE_MEASURED_OPERATOR`.
The speedup compares local FA4 with the full distributed graph wall interval.
NCCL, stable merge, query transfer, and completion are included.
The interval uses the slowest measured rank for each paired repetition.
The 95% interval resamples process-start means.
It does not treat loop, layer, or repeated kernel calls as independent requests.

Repeated-KV tests can favor a shard that fits in L2.
The pressure control touches a separate 128 MiB buffer before each timed operation.
That buffer operation is outside the timing interval for both arms.
The test is cache-pressure sensitivity, not counter-certified cold-cache behavior.

`capacity_by_gpu_count.csv` reports actual live operator KV payload and page padding.
Full-model memory equivalents multiply by 96 and remain `MODEL_UPPER_BOUND`.
They do not prove that an integrated 96-namespace distributed server fits or runs.
Reserved allocator memory can exceed valid shard payload.
Later confirmatory records also retain pre-timing allocated and reserved CUDA bytes.

The primary balanced split and 75%-on-rank0 split use the same original state.
They test generic shard imbalance.
No added loop-aware planner improves that generic implementation here.

The pinned RLT model has no matched TP2/4/8 path.
Those baselines are `UNSUPPORTED_TP_BASELINE_IN_PINNED_RLT`.
This is not a claim that Ouro lacks TP in every runtime.
`DP_replica_scaling.csv` records actual independent replica cohorts.
Their throughput is scale-out hardware throughput, not B1 memory extension or SLO capacity.

There are two collectives per tested attention operator.
Repeating them over 24 layers and four loops would require 192 collectives per token.
That multiplication is structural accounting, not a measured full-model collective count.
Application payload is not measured NVLink wire traffic.
The full serving integration remains `NOT_RUN_FULL_SERVING`.
''')
    put('correctness.md',f'''# Correctness summary

The original model weights, BF16 precision, RoPE, and Static R4 remain unchanged.
Native generation asserts exit depth four for every returned token.
Complete GPU windows contain 96 attention executions.
The cache manager uses four distinct depth planes.
Initialized fixture hashes preserve the original BF16 K/V values.

The exact layout probe passes twelve sampled physical-layer/depth comparisons.
Its Q, K, V, and attention outputs are byte-exact.
The two-token output and depth arrays also match.
The full timing cohorts retain output tokens for paired analysis.
The cohort comparison must not replace full-answer accuracy evaluation.

The distributed operator has {len(raw)} retained paired rows.
Its maximum relative L2 is {numeric_max}.
Its maximum absolute error is {abs_max}.
`distributed_attention_correctness.md` states the batch and mask limitations.
The edge results remain in their own archived rank records.

No integrated distributed natural generation was performed.
GSM8K/MATH-500 accuracy is `NOT_RUN`.
There is no claim of statistical quality equivalence.
''')
    put('serving_summary.md',f'''# Serving scope

The native model completed {counts['timing_cohorts']} retained timing cohorts.
The primary lane is warmed-engine, bounded hardware inference.
It records actual frontend timestamps and generated token IDs.
It does not use frozen output-token replay.
It does ignore EOS to control hardware output length.

Separate EOS-enabled book-continuation probes retain their natural stop outcomes and 128-token cap.
They are not an unrestricted long-answer benchmark.
The report makes no sustainable serving-capacity claim from those finite probes.

The distributed path is an operator, not a model server.
Its timing cannot populate TTFT, TPOT, or natural SLO tables.
Those candidate serving metrics are `NOT_RUN`.
The pause on native-vLLM Loop PP remains in force.
''')
    put('phase_gates.md',table(['Phase','Status','Evidence and limit'],[
        ['P0','PASS','Pinned native runtime, model, topology, 1/2/4/8 contention, and ownership checks.'],
        ['P1','CONDITIONAL','Long-context attention and KV capacity measured. Clean exclusive attribution and hardware counters remain qualified.'],
        ['P2','CONDITIONAL','Exact generic layout, allocator, tiering, and local sequence-parallel controls. No loop-specific advantage established.'],
        ['P3','PASS','Real two-rank exact BF16 sharding and NCCL merge. Operator only.'],
        ['P4','PASS','Real four/eight-rank operator scaling. Full-cache serving and TP comparison remain unimplemented.'],
        ['P5','NO_GO','No measured increment beyond the identical generic operator or allocator.'],
        ['P6','NOT_RUN','No qualifying loop-aware full-model candidate. No benchmark accuracy or sustainable SLO result.']])+
        '\nGate G5 is `NO_GO_INCREMENTAL` for the current loop-specific claim.\nGeneric operator results remain valid and separate.\n')
    put('next_actions.md','''# Next actions

Do not implement a new loop-aware serving engine from the present evidence.
First compare the strong generic local segmentation control with the distributed pressure control.
Resolve hardware counter availability and local backend support before a broad architecture claim.
If generic attention remains promising, request a separate bounded integration study.
That study must compare optimized local sequence parallelism and legal native TP.
It must preserve exact paged KV and captured-graph correctness.
Only a qualified natural-generation candidate should trigger GSM8K/MATH and SLO expansion.
Do not resume the paused Loop PP project.
''')
    put('analysis.md',f'''# Loop-aware KV residency and distributed attention discovery

## Verdict

**NO_GO_INCREMENTAL for a new loop-specific method.**
**CONDITIONAL for generic attention engineering, not a serving speedup claim.**

Long context exposes a real attention interval and substantial exact KV payload.
Real context-sharded attention runs on 2/4/8 physical H100 GPUs.
Short-context collective overhead and long-context operator scaling differ substantially.
However, the implementation is generic exact context parallelism.
No added loop-aware placement or prefetch policy demonstrates an incremental benefit.
The study does not convert that generic operator signal into a new method claim.

## Scope and completed work

Model: ByteDance/Ouro-1.4B Base, BF16, original Static R4.
Checkpoint: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Runtime: unmodified RLT `4a350344027960a48384626aa646f1765fda6d4f`.
The baseline uses native Async/Mixed, paged KV, FA4, and CUDA Graphs.
Physical GPUs 0–7 supplied independent workers and real distributed rank groups.
No Loop PP implementation or scheduling-policy research was resumed.

Completed model jobs: {counts['completed_model_jobs']}.
Retained model timing cohorts: {counts['timing_cohorts']}.
Paired distributed operator rows: {len(raw)}.
The raw archive retains failed attempts as well as completed attempts.

## Native hardware baseline

{baseline_table}

These are real generated-token measurements with EOS ignored and cap 64.
They are not a natural accuracy or sustainable-serving benchmark.
The 65,408-token prompt remains below the 65,536 total-position limit.
An exact 65,536-token prompt plus generation was skipped.

## Physical attention interval

{trace_table}

The table uses heavy-trace GPU intervals, not clean E2E percentages.
The clean critical-path fraction remains `UNRESOLVED` where causal attribution is incomplete.
The increasing attention interval is nevertheless a physical long-context observation.
The old eager Thinking-model 7.3% result is not reused as the present denominator.

## Exact generic controls

{layout_table}

Layer-major storage is a generic allocator change.
Smaller physical pools save empty reservation, not valid KV bytes.
Host/peer movement tests charge original BF16 bytes and completion waits.
Already resident HBM does not supply a free prefetch opportunity.

The internal FA4 SplitKV option fails on this SM90 backend.
The study therefore tests generic local sequence segmentation with the same FA4 kernel.
It retains K/V in place and merges exact partial attention.
The control separates missing local kernel parallelism from a need for additional GPUs.

{local_table}

## Physical distributed result

{op_table}

These are operator speedups, not full-model E2E speedups.
The main matrix covers real captured states at four depths and three physical layers.
Batch expansion copies a captured state and does not represent independent natural requests.
The pressure control tests an important warm-cache sensitivity.
An integrated model cycles through 96 namespaces rather than one repeated fixture.

## Answers to the research questions

1. Exact KV/attention becomes physically important at long context. The report keeps trace-local shares separate from clean E2E attribution.
2. Payload grows with context and concurrency. B2 near 64K or B4 at 32K exceeds one H100 before non-KV allocations.
3. Exact generic layout and local sequence-parallel controls are measured. They do not establish a loop-specific memory method.
4. The distributed implementation is itself the generic CP control. No distinct loop-aware method beats it.
5. Per-rank operator KV shrinks through real ownership. Full-model capacity, natural throughput, and E2E improvement remain unmeasured.
6. Static R4 reads the same valid context extent at each depth. No causal state pair proves an additional placement advantage.
7. We preserve original BF16 KV and test numerical error. Full distributed natural generation and benchmark accuracy remain `NOT_RUN`.
8. FlashLoop, HLA, LLA, ResidualQuant, and MELT alter representation or computation. Preserving BF16 distinguishes scope, not automatic novelty.

## Limitations and decision

The strongest same-resource integrated TP baseline is absent from the pinned RLT engine.
Native TP support in another engine is not ruled out.
The DP calibration measures independent hardware cohorts, not SLO capacity.
The complete clean FFN/attention/host causal partition remains partial.
Initial discovery groups lack an at-start experiment-source snapshot.
Later confirmations retain rank-specific source snapshots.
The measured attention/merge graph definition remains unchanged across those confirmations.

The current evidence does not justify a new loop-specific distributed runtime.
It can justify a separate generic attention-baseline integration question after the strong local control.
No new model engine, KV approximation, training, quantization, or hidden-state prediction was implemented.

Branch: `flashvep/ouro-loop-aware-kv-residency-poc`.
The final Git commit identifies the exact delivery.
`reproduce.md` contains commands and artifact locations.
''')

if __name__=='__main__':main()
