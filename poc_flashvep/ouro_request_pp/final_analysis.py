"""Produce bounded conclusions with explicit evidence classes and controls."""
import collections
import json
import numpy as np
from .contract import ROOT,write
from .analyze import read,paired_ratio
from .make_report import table,median


def main():
    data=json.loads((ROOT/'results_raw.json').read_text())
    profiles=json.loads((ROOT/'profile_index.json').read_text())
    primary=json.loads((ROOT/'paired_comparison.json').read_text())
    control_rows=[];controls=[]
    for prefix in ['receive_pool_','receive_pool_rope_']:
        groups=collections.defaultdict(dict)
        for r in data:
            # Do not mix the two nested name prefixes.
            expected='raw/'+prefix+r['placement']+'/'
            if not r['file'].startswith(expected):continue
            groups[(r['depth'],r['C'],r['P'],r['O'])].setdefault(r['placement'],[]).append(r)
        for key,g in sorted(groups.items()):
            if len(g)!=2:continue
            a,b=g['layer_pp4'],g['loop_pp4']
            row=dict(control=prefix.rstrip('_'),R=key[0],C=key[1],P=key[2],O=key[3],
                layer_s=median(a,'wall_s'),loop_s=median(b,'wall_s'),
                **paired_ratio([x['wall_s'] for x in a],[x['wall_s'] for x in b]))
            controls.append(row)
            control_rows.append([row['control'],*key,f"{row['layer_s']:.4f}",f"{row['loop_s']:.4f}",
                                 f"{row['layer_over_loop_speedup']:.3f}×"])
    write(ROOT/'backend_controls.json',controls)
    text='''# Analysis and final verdict

## Decision

**CONDITIONAL.** Both exact request-level pipelines work on four physical GPUs.
The experiment shows a repeatable implementation-level advantage for Loop-PP in most fixed-R4 cases.
It does not establish an architecture advantage over an optimized multiprocess engine, TP4, or DP4.
The evidence does not justify a new adaptive scheduler or a general serving-performance claim.

Gate A, executable correctness: PASS on the tested fixed-depth inputs.
Gate B, actual request-level pipeline overlap: PASS in both placements.
Gate C, paired discovery measurements: PASS, with the timing and backend limitations below.
Gate D, true dynamic depth: NOT_RUN.

## Primary result

There are 162 physical cohort measurements: 27 paired cases, two arms, and three repeats.
Loop-PP has the lower median makespan in 25 cases.
The median speedup range is 0.971× to 2.044× across all cases.
The two median Layer-PP wins occur at R2, C8/C16, P128, O32.
Both intervals include 1.000×. They do not establish a reliable crossover.
The complete values and intervals appear in [benchmark_summary.md](benchmark_summary.md).

Representative fixed-depth results:

'''
    rows=[]
    for R,C,P,O in [(4,1,128,32),(4,4,128,32),(4,16,128,32),(4,16,512,128),(2,16,128,32),(3,16,128,32)]:
        r=next(x for x in primary if (x['depth'],x['C'],x['P'],x['O'])==(R,C,P,O))
        rows.append([R,C,P,O,f"{r['layer_median_s']:.4f}",f"{r['loop_median_s']:.4f}",f"{r['layer_over_loop_speedup']:.3f}×"])
    text+=table(['R','C','P','O','Layer-PP makespan s','Loop-PP makespan s','Loop/Layer speedup'],rows)
    text+='''
The input and output lengths describe hardware shapes, not public-benchmark quality settings.
TTFT includes whole-prompt prefill and queueing.
The supplied CSV also records mean, P50, and P90 request metrics.
No stage combines several requests into a larger GEMM batch.
Longer Loop-PP stages therefore do not increase the tensor batch size of each decoder GEMM.

## Actual overlap and physical transport

These values come from the separate heavy profiler lane.
The headline comparison does not use these wall times.
Kernel overlap requires different request IDs on different GPUs at the same profiler time.
Stage brackets alone do not provide this proof.

'''
    rows=[]
    for p in profiles:
        name=p['source'].split('/')[-1].replace('.sqlite','')
        peer=next((x for x in p['copy_categories'] if x['copyKind']==10),dict(count=0,bytes=0,service_ms=0))
        rows.append([name,f"{p['window_s']*1000:.3f}",f"{p['different_request_kernel_overlap_ms']:.3f}",
            p['max_simultaneous_kernel_ranks'],peer['count'],f"{peer['bytes']/2**20:.3f}",f"{peer['service_ms']:.3f}"])
    text+=table(['Profile','Heavy window ms','Different-request kernel overlap ms','Max active kernel ranks',
                'Actual P2P copies','P2P MiB','Sum P2P activity ms'],rows)
    text+='\nPer-GPU activity in the same heavy lane:\n\n'
    text+='The activity union includes recorded kernels and copies. Its complement includes pipeline bubbles, host gaps, and waiting.\n'
    text+='It does not isolate architectural bubbles from observer or host-scheduler costs.\n\n'
    text+=table(['Profile','GPU','Kernel service ms','Kernel/copy activity union ms','No recorded activity %'],
        [[p['source'].split('/')[-1],r['physical_GPU'],f"{r['kernel_service_ms']:.3f}",
          f"{r['recorded_activity_union_ms']:.3f}",f"{100*r['no_recorded_activity_fraction']:.2f}"]
          for p in profiles for r in p['ranks']])
    text+='''
The R4/C4/O8 Layer-PP trace contains 480 hidden copies and 28 token returns.
The corresponding Loop-PP trace contains 96 hidden copies and 28 token returns.
Nsight reports these operations as Peer-to-Peer copies.
Its peer-copy bytes exactly match the executed runtime payload counter.
H2D traffic contains only four 128-token inputs. D2H traffic contains 32 scalar token IDs.
There is no hidden-state or KV staging through host RAM.

R4 hidden payloads fall by 80% because the handoff count changes from 15 to 3 per forward.
Total hidden bytes include a full-prompt tensor on prefill and one hidden row on later forwards.
For uniform shapes, they equal `C * (P + O - 1) * 2048 * 2 * handoffs`.
Token return and scalar control are additional measured traffic.
Each KV slot stays on its owner GPU for the whole request.

Do not interpret P2P service sums as an E2E critical-path share.
Different links and compute can overlap.
The physical transfer service difference is much smaller than the observed cohort wall-time difference.
This rules out a payload-bandwidth-only explanation in the traced condition.

## Stage granularity and runtime cost

The principal measured distinction is the number of stage boundaries.
R4 Layer-PP executes 16 stage items per forward. Loop-PP executes four.
Each item includes a queue operation, input publication, graph replay, local KV staging, and completion wait.
Layer-PP also re-enters GPU4 three times per forward.
The heavy trace records 512 versus 128 graph launches for C4/O8.
It records 520 versus 136 event-synchronization calls, including workload-clock checks.
The per-rank kernel service times are broadly similar. The gaps between active GPU intervals are substantial.

The native model computes position embeddings once for a forward pass.
The initial common pipeline path recomputes those embeddings at each stage.
That adds more redundant RoPE preparation to Layer-PP.
The backend controls below remove that cost with immutable position constants.
They preserve every input-dependent decoder operation and exact RoPE values.

Python threads, GIL contention, event publication, allocator behavior, and local graph staging can affect the gap.
The experiment does not isolate each of those costs as a non-overlapped E2E component.
Nsight node tracing substantially changes timing.
Its activity fractions describe that heavy execution only.
They are not clean-run GPU utilization measurements.

## Backend controls

The first control preallocates destination activation buffers before admission.
It changes no placement, model math, queue discipline, or payload byte count.
The second control also caches position-only RoPE constants outside graph replay.
Each control passes a separate single-GPU reference comparison before performance measurement.
The controls use three physical repeats per row.
Control execution order differs from the main alternating matrix. Treat them as sensitivity checks.

'''+table(['Control','R','C','P','O','Layer-PP s','Loop-PP s','Loop/Layer speedup'],control_rows)
    text+='''
These controls do not remove the single-process host scheduler or publication waits.
They cannot establish performance for an optimized four-process transport implementation.
Do not attribute the full remaining gap to link bandwidth or to a fundamental loop-specific bottleneck.

## Low depth and pipeline bubbles

At R2, Loop-PP schedules no decoder work on GPUs 6 and 7.
At R3, it schedules no decoder work on GPU7.
Layer-PP uses all four GPUs for every executed loop at either depth.
The depth traces record actual omitted work, not a post-hoc exit label.
Idle late GPUs alone do not prove a throughput crossover.
In this prototype, shorter routes offset some loss of distributed compute capacity.
The R2 primary medians approach parity. The R4 gap is larger.

The finite cohort includes initial fill and final drain.
The same FIFO policy does not imply the same order of eligible work across different stage granularities.
The trace retains every ready-queue wait with request, layer range, loop, and GPU identifiers.
No per-request global synchronization prevents a different request from progressing.

The following short trace records sum stage brackets on each GPU.
Queue time sums waits across distinct stage items. It can exceed cohort wall time.
Compute brackets include launch gaps. They are not pure kernel service.

'''
    rows=[]
    for arm in ['layer_pp4','loop_pp4']:
        d=read(ROOT/'raw'/f'probe_{arm}_v5'/'case_0000.json.gz')
        for rank in range(4):
            es=[e for e in d['events'] if e['rank']==rank]
            pre=[e for e in es if e['step']==0];dec=[e for e in es if e['step']>0]
            rows.append([arm,rank+4,len(es),f"{sum(e['compute_ms'] for e in pre):.2f}",
                f"{sum(e['compute_ms'] for e in dec):.2f}",f"{sum(e['queue_ms'] for e in es):.2f}",
                f"{np.quantile([e['queue_ms'] for e in es],.9):.2f}"])
    text+=table(['Arm, R4 C4 P128 O8','GPU','Stage calls','Prefill compute brackets ms',
                'Decode compute brackets ms','Sum ready-queue waits ms','P90 stage queue wait ms'],rows)
    paths=json.loads((ROOT/'prefill_paths.json').read_text())
    rows=[]
    for arm in ['layer_pp4','loop_pp4']:
        rr=[r for r in paths if r['file']==f'raw/probe_{arm}_v5/case_0000.json.gz']
        rows.append([arm,f"{np.median([r['prefill_path_host_ms'] for r in rr]):.2f}",
                     f"{np.median([r['initial_entry_queue_ms'] for r in rr]):.2f}",
                     f"{np.median([r['TTFT_host_ms'] for r in rr]):.2f}"])
    text+='\nPrefill path spans include internal stage waits but exclude the initial admission wait.\n'
    text+='These trace-lane values are separate from the clean TTFT table.\n\n'
    text+=table(['Arm','Median prefill path wall ms','Median initial entry wait ms','Median TTFT ms'],rows)
    text+='''

## Memory trade-off

Layer-PP uses 0.9493 GiB of weights per GPU, including auxiliary replicas.
Loop-PP uses 2.6722 GiB per GPU.
The extra weight cost is approximately 1.7229 GiB per GPU.
The same full LM head and embedding are present in both arms.
For R4, owner-local KV storage is equal across placements at the same request shape.
There is no intrinsic logical-KV compression.
At R2/R3, Loop-PP concentrates all live KV on its active depth ranks.

The prototype retains separate graph storage for exact attention lengths.
That storage is much larger than necessary in a production engine.
Peak PyTorch allocation is not a model-weight-plus-KV capacity estimate.
The memory plot explicitly separates weights, owner-local KV, and other allocations.
No maximum serving-concurrency claim follows from this matrix.

## Observer validity and correctness scope

'''
    observer=json.loads((ROOT/'observer_overhead.json').read_text())
    text+=table(['Arm','Clean median s','Event-trace median s','Trace/clean ratio'],
        [[r['arm'],f"{r['clean_median_s']:.4f}",f"{r['trace_median_s']:.4f}",f"{r['trace_over_clean']:.3f}"] for r in observer])
    text+='''
Three short control repeats give a noisy overhead estimate, not a universal correction factor.
The analysis applies no correction factor to clean latency.
The pipeline's safe-publication event also exists in clean mode.
Heavy profiling and per-stage trace data remain outside the headline lane.

The final runtime matches the single-GPU reference bitwise on the tested numerical probes.
Both placements also match every primary generated token ID.
This is not an accuracy evaluation on GSM8K or MATH.
EOS-enabled capped probes remain separate from the forced-length matrix.
See [correctness.md](correctness.md) for actual stop outcomes and preserved failures.

## Dynamic depth and future architecture

True online token-level dynamic exit: NOT_IMPLEMENTED.
Fixed R2/R3 do not establish correctness for a variable-depth request's missing historical KV.
An exit-aware runtime needs separate gate, active-token, and ragged-KV correctness checks.
The earlier adaptive `last_exited` CDB implementation is not silently substituted here.

Full weight replicas make alternative compute ownership possible in principle.
They do not make depth-indexed KV available on an arbitrary GPU.
Moving a later decode stage can require KV transfer, duplication, or a new cache policy.
None of those costs can be omitted from a future proposal.
Neither fixed Loop-PP nor generic PP is a novelty claim in this report.

The smallest useful next experiment is a common multiprocess/event-driven transport control.
Keep the checked placements, exact caches, frozen inputs, and no-chunking rule.
Remove avoidable Python scheduling overhead before designing any new adaptive placement.
Then compare against a same-four-GPU DP or optimized serving baseline.
Proceed to true dynamic depth only after that control preserves the advantage and numerical correctness.

## Artifacts

- [Environment and ownership](environment.md)
- [Correctness and failures](correctness.md)
- [Complete performance tables](benchmark_summary.md)
- [Timing definitions](timing_contract.md)
- [Reproduction commands](reproduce.md)
- [Raw aggregate CSV](results.csv)
- [Per-stage data](stage_metrics.csv)
- [Transport edges](transport_edges.csv)
- `raw/`: losslessly compressed inputs, generated IDs, token times, memory, and counters.
- `profiles/`: Nsight reports, exported activity, four-GPU kernel timelines, and per-rank summaries.
- `commands/` and `logs/`: exact invocations, source hashes, status, and failure logs.
- `source_v5.tar.gz`: source snapshot before optional transport/RoPE controls.
'''
    (ROOT/'analysis.md').write_text(text)
    (ROOT/'report.md').write_text('# Ouro request-level PP discovery\n\nFinal verdict: **CONDITIONAL**.\n\n'+
        'Read [analysis.md](analysis.md) for the complete result and limitations.\n'+
        'Read [benchmark_summary.md](benchmark_summary.md) for all 27 measured paired cases.\n')
    write(ROOT/'decision.json',dict(verdict='CONDITIONAL',gates=dict(A='PASS',B='PASS',C='PASS_WITH_LIMITATIONS',D='NOT_RUN'),
        reason='Exact physical request pipelines work. Loop-PP usually wins in this prototype. Host/backend costs prevent a general architecture claim.',
        true_dynamic_depth='NOT_IMPLEMENTED',benchmark_accuracy='NOT_RUN',backend_controls=controls))
    print('FINAL_ANALYSIS',len(controls),'control rows')


if __name__=='__main__':main()
