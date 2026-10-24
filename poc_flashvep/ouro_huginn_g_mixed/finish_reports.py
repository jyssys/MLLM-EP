"""Complete environment, profiling, scale-out, and experiment coverage reports."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,MODELS,UUIDS,write
from .report import text,table
from .analyze import csvfile

def main():
    facts={m:json.loads((ROOT/'raw'/f'support_{m}_s0'/'model_facts.json').read_text()) for m in MODELS}
    rows=[[m,MODELS[m].name,facts[m]['parameters'],facts[m]['parameter_bytes'],4 if m=='ouro' else 16,
        'async' if m=='ouro' else 'sync',65536 if m=='ouro' else 4096] for m in MODELS]
    text('environment.md','''# Environment and model provenance

The hardware is eight physical H100 80-GB devices.
The initial GPU query and topology appear in `environment/`.
`gpu_uuid_map.json` identifies physical GPUs 0–7.
The topology reports NV18 links between every GPU pair.
No model-parallel TP/PP execution belongs to this study.

'''+table(['Model','Checkpoint revision','Unique parameters','Parameter bytes','Fixed R','Execution','Configured context limit'],rows)+'''

The model facts come from the actual loaded pretrained modules.
Huginn changes its inference recurrence from checkpoint-default R32 to the registered R16.
No checkpoint tensor changes.
Its fixed prelude injection, adjacent-pair RoPE, and prelude/coda KV remain intact.

The source pin is RLT `4a350344027960a48384626aa646f1765fda6d4f`.
The isolated worktree and every removable source diff remain in provenance and worker identities.
The GPU Python is `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python`.
The recorded environment uses Torch 2.11.0+cu129 and FlashAttention4 4.0.0b30.
`validation/hardened_tests/packages.json` contains the complete package lock.

Workers receive distinct CUDA contexts and physical-device masks.
Each worker has a separate CPU-affinity range, CUDA/JIT cache path, log, and raw attempt directory.
The dispatcher uses one measurement process per GPU.
It preserves completed hashes and refuses failed-attempt overwrite.
The resource ledger records verified burner sessions and GPU ownership checks.
Quiet groups reserve the other devices to exclude burner interference.
Other independent stages restore a finished device's burner while its peers continue.
Final handoff restores all eight owned burners.

The shared KV pool uses the same block count and layout within each model's paired cell.
Peak HBM and actual later cache shapes appear in raw rows and the DP export.
Reserved pool capacity is not initialized live KV payload.
''')
    calibration=[];baseline=None;device8=[]
    for width in [1,2,4,8]:
        path=ROOT/'raw'/f'contention_w{width}_gpu0_s0'/'results.json'
        values=[r['wall_s'] for r in json.loads(path.read_text())]
        median=float(np.median(values))
        if baseline is None:baseline=median
        calibration.append([width,3,f'{median*1000:.3f}',f'{median/baseline:.6f}'])
    for gpu in range(8):
        values=json.loads((ROOT/'raw'/f'contention_w8_gpu{gpu}_s0'/'results.json').read_text())
        device8.append(float(np.median([r['wall_s'] for r in values])))
    text('gpu_contention.md','''# Shared-resource contention control

The calibration uses Native Mixed P256/C128 with two background requests.
Every process completes its warm-up before a shared start barrier.
The table uses GPU0 across 1/2/4/8 concurrent workers.
Each worker count has one process start and three technical cohorts.
This is a narrow interference control, not a universal zero-interference claim.

'''+table(['Concurrent workers','Technical cohorts','Median cohort wall ms','Ratio to solo GPU0'],calibration)+f'''

The eight-worker cross-GPU median-wall CV is {np.std(device8)/np.mean(device8):.6f}.
The primary confirmation runs one model process at a time in a reserved quiet lane.
It crosses physical GPUs and rotates policy order across three starts per model/width.
The exploratory grid uses parallel independent workers and remains a separate lane.
Its outliers do not replace the quiet paired results.
The telemetry records sampled clocks, power, memory, CPU load, and host I/O.
Those samples do not directly measure exclusive GPU idle time.
''')
    dp=list(csv.DictReader((ROOT/'dp_scaleout.csv').open()))
    table_rows=[[r['model'],r['DP'],r['arm'],r['condition'],r['cluster'].split('_')[-1],
        f"{float(r['makespan_s']):.4f}",f"{float(r['requests_per_s']):.3f}",
        f"{float(r['requests_per_s_per_GPU']):.3f}",f"{float(r['makespan_speedup_vs_DP1_native']):.3f}×" if r['makespan_speedup_vs_DP1_native'] else 'NOT_AVAILABLE_matched_DP1',
        f"{r['completed']}/{r['offered']}"] for r in dp]
    text('dp_scaleout.md','''# Actual DP scale-out

Each replica owns one complete model and its local KV state.
The dispatcher assigns global request i to replica i modulo DP width.
All replicas use one common offered burst timestamp.
Each cluster receives the same 32 short arithmetic chat requests.
The table measures actual execution, not trace replay or a throughput simulator.

'''+table(['Model','DP','Arm','Warm-up condition','Start block','Makespan s','Requests/s','Requests/s/GPU','Speedup vs matched model DP1 Mixed','Complete/offered'],table_rows)+'''

Makespan includes actual owned-engine drain after the final response.
TTFT and ITL use delivered frontend timestamps.
All-offered request records remain in `dp_request_metrics.csv`.
Ouro honors its registered chat-end token in the frontend.
Huginn uses its native EOS behavior.

The initial complete sweep has one cluster start per configuration.
The Huginn DP8 positive point receives two additional independent starts with crossed arm order.
That confirmation is not a full capacity experiment.
The work has no registered Poisson-rate or stability-qualified 90% SLO crossing.
Do not call these finite-burst rates sustainable goodput.

The Hybrid arm is the fixed event hybrid control.
It is not OURS_ADAPT_G_MIXED.
Any positive static-hybrid result remains a control result, not evidence of adaptive superiority over the best fixed hybrid.
DP scaling is generic replication and does not establish a new multi-GPU parallelism mechanism.
''')
    h8=[r for r in dp if r['model']=='huginn' and r['DP']=='8']
    pairs=[]
    confirmations={}
    for condition,blocks in [('LIBRARY_WARM_ONLY',['r1','r2','r3']),('MATCHED_CHAT_WARM',['r4','r5','r6','r7','r8'])]:
        pairs=[]
        for block in blocks:
            a=next((r for r in h8 if r['arm']=='mixed' and r['cluster'].endswith(block)),None)
            b=next((r for r in h8 if r['arm']=='hybrid' and r['cluster'].endswith(block)),None)
            if a and b:pairs.append(float(a['makespan_s'])/float(b['makespan_s']))
        if len(pairs)>=3:
            rng=np.random.default_rng(20261025)
            ci=np.percentile(np.array(pairs)[rng.integers(0,len(pairs),size=(10000,len(pairs)))].mean(axis=1),[2.5,97.5])
            confirmations[condition]=dict(speedup_ratios=pairs,mean=float(np.mean(pairs)),
                paired_start_CI95=ci.tolist(),independent_starts=len(pairs),capacity_qualified=False)
            with (ROOT/'dp_scaleout.md').open('a') as f:f.write(f'\nHuginn DP8, {condition}: paired makespan speedup {np.mean(pairs):.4f}×, 95% interval [{ci[0]:.4f}, {ci[1]:.4f}].\nThis interval covers {len(pairs)} starts and one repeated short-chat source family.\n')
    write(ROOT/'dp_huginn_confirm.json',confirmations)
    events=list(csv.DictReader((ROOT/'gpu_event_intervals.csv').open()))
    group=defaultdict(list)
    for row in events:group[(row['model'],row['arm'],row['G'],row['operation'])].append(float(row['GPU_stream_interval_ms']))
    profile_rows=[list(key)+[len(values),f'{np.median(values):.4f}',f'{np.percentile(values,95):.4f}'] for key,values in sorted(group.items())]
    text('profiling.md','''# Separate heavy profiling

The profile uses producer-stream CUDA events after a complete warm-up workload.
Its raw records retain interval start, interval end, and elapsed milliseconds.
Host submission plans remain a different clock.
The figures identify their evidence lane.

'''+table(['Model','Arm','Nominal G','Operation','Intervals','Median stream ms','P95 stream ms'],profile_rows)+'''

A recurrent interval can contain multiple required prompt passes.
It includes GPU stream elapsed time and possible launch gaps.
It is not a hardware-counter measurement of kernel-only busy time.
CPU scheduling can overlap GPU execution.
The report does not sum CPU and GPU intervals into an E2E speedup.
It does not sum overlapping stream intervals into a 100% critical-path table.

Grouped prefill retains native per-pass attention/GEMM execution.
The patch does not fuse CUDA kernels or create a multi-loop CUDA Graph.
Mixed G1 can co-pack prompt and decode rows at different depths.
Its query width can exceed graph support and use the native eager fallback.
The clean raw rows preserve actual capture/replay/fallback increments.

`observer_parity.csv` contains three ON/OFF blocks per model.
Their elapsed-time ratios describe the light plan observer, not the heavy tensor audit.
The clean confirmation excludes that observer.
Exclusive GPU idle, kernel launch attribution, and HBM counters remain unresolved in this scope.
''')
    states=[]
    for plan_path in (ROOT/'plans').glob('*.json'):
        plan=json.loads(plan_path.read_text())
        blocks=plan.get('groups',[dict(jobs=plan.get('jobs',[]))])
        for block in blocks:
            for job in block['jobs']:
                folder=ROOT/'raw'/job['id']
                status='PASS' if (folder/'completed.json').exists() else 'FAILED' if (folder/'failure.json').exists() else 'NOT_RUN'
                states.append(dict(plan=plan_path.name,job=job['id'],model=job['model'],gpu=job['gpu'],
                    kind=job['kind'],P=job.get('P'),N=job.get('N'),C=job.get('C'),status=status))
    states.append(dict(plan='checkpoint_constraint',job='huginn_P8192',model='huginn',gpu='',
        kind='unsupported_context',P=8192,N='',C='',status='NOT_RUN_CONFIG_LIMIT'))
    csvfile('experiment_matrix.csv',states)
    csvfile('slo_attainment_by_rate.csv',[dict(model=m,status='NOT_RUN',reason='No promoted adaptive method or qualified capacity experiment') for m in MODELS])
    csvfile('quality_cases.csv',[dict(model=m,dataset=d,status='NOT_RUN',N=0,correct='',accuracy='') for m in MODELS for d in ['GSM8K','MATH-500']])
    print('SUPPLEMENTAL_REPORTS_WRITTEN',len(dp),'DP cohorts',len(events),'event intervals')

if __name__=='__main__':main()
