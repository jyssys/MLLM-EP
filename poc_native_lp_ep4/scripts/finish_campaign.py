"""Finite execution plan: correctness -> discovery -> frozen final -> trace.

This is not a production scheduler. Each subprocess has a hard timeout and
the GPU-session wrapper restores only the authorized existing burners.
No failed stage is automatically retried or silently relabeled.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from poc_native_lp_ep4.contract import atomic_json, groups_for, read_json
from poc_native_lp_ep4.scripts.analyze import clean_rows, summarize, trace_rows
from poc_native_lp_ep4.scripts.aggregate import component_waves, component_summary

ROOT=Path('poc_native_lp_ep4')
VERSION='v2'


def launch(name, arms, repeats, *, trace=False, light=False,
           correctness=False, paired=False, overhead=False, decode=96, timeout=2700,
           repeat_offset=0):
    output=ROOT/'artifacts/runs'/name
    if output.exists():
        if not (output/'status.json').exists() or not read_json(output/'status.json')['ok']:
            raise RuntimeError(f'failed/incomplete existing run: {output}; choose a new version')
        return output
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,4,5',VLLM_NO_USAGE_STATS='1',
        PYTHONPATH=f'{Path.cwd()}/poc_native_lp_ep4/hooks:{Path.cwd()}',
        NATIVE_LP_ENABLE='1',NATIVE_LP_OUTPUT=str(output.resolve()),
        MLLM_COMM_OUTPUT=str(output.resolve()),
        MLLM_COMM_CONTROL=str((ROOT/'artifacts/control.json').resolve()),
        NATIVE_LP_TRACE='1' if trace else '0',
        NATIVE_LP_LIGHT_TRACE='1' if light else '0',
        NATIVE_LP_CORRECTNESS='1' if correctness else '0')
    cmd=[sys.executable,str(ROOT/'scripts/gpu_session.py'),'--timeout',str(timeout),
         sys.executable,str(ROOT/'scripts/run.py'),'--output',str(output),
         '--arms',','.join(arms),'--modalities','VISUAL_PREFILL,TOKEN_MATCHED_TEXT',
         '--repeats',str(repeats),'--decode-tokens',str(decode),'--inject-at','16',
         '--preprocess-before-injection','--repeat-offset',str(repeat_offset)]
    if trace:cmd+=['--trace']
    if VERSION in ('v5','v6','v7'):cmd+=['--atomic-wave-admission']
    if VERSION in ('v6','v7'):cmd+=['--uniform-resident-prefill']
    if trace and not light:cmd+=['--route-stats']
    if correctness:cmd+=['--correctness']
    if paired:cmd+=['--pair-baseline']
    if overhead:cmd+=['--interleave-instrumentation']
    ledger=ROOT/'artifacts/campaign_ledger.jsonl'
    with ledger.open('a') as f:f.write(json.dumps({'event':'START','name':name,
        'wall_ns':time.time_ns(),'command':cmd})+'\n')
    with (ROOT/'artifacts'/f'{name}.log').open('w') as log:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                              timeout=timeout+100,start_new_session=True)
    with ledger.open('a') as f:f.write(json.dumps({'event':'END','name':name,
        'wall_ns':time.time_ns(),'exitcode':result.returncode})+'\n')
    if result.returncode or not read_json(output/'status.json')['ok']:
        raise RuntimeError(f'{name} failed; raw evidence retained; no automatic retry')
    return output


def check_correctness(run, name):
    target=ROOT/'artifacts/correctness'/name
    if target.exists():
        result=read_json(target)
        assert result['status']=='PASS' and result['source_run']==str(run)
        return result
    subprocess.run([sys.executable,str(ROOT/'scripts/check_correctness.py'),
                    '--run',str(run),'--output',str(target)],check=True)
    return read_json(target)


def main():
    global VERSION
    p=argparse.ArgumentParser()
    p.add_argument('--initial-correctness',type=Path,required=True)
    p.add_argument('--wait-for-initial-seconds',type=int,default=0)
    p.add_argument('--version',default='v2')
    p.add_argument('--wait-for-run',type=Path)
    args=p.parse_args()
    VERSION=args.version
    if args.wait_for_run:
        deadline=time.monotonic()+1800
        while not (args.wait_for_run/'status.json').exists():
            if time.monotonic()>=deadline:raise TimeoutError('preceding diagnostic run did not end')
            time.sleep(5)
    deadline=time.monotonic()+args.wait_for_initial_seconds
    while not (args.initial_correctness/'status.json').exists():
        if time.monotonic()>=deadline:
            raise TimeoutError('initial correctness run has no final status')
        time.sleep(5)
    if not read_json(args.initial_correctness/'status.json')['ok']:
        raise RuntimeError('native correctness execution did not complete')
    check_correctness(args.initial_correctness,'coordination_v3_validation.json')
    # The initial run may have written its status just before GPU teardown.
    # Subsequent launch() calls return only after their wrapper cleanup.
    from poc_native_lp_ep4.scripts.gpu_session import compute_pids,owned_experiment
    deadline=time.monotonic()+90
    while any(owned_experiment(pid) for pid in compute_pids()):
        if time.monotonic()>=deadline:raise TimeoutError('initial GPU teardown did not finish')
        time.sleep(2)
    time.sleep(3)
    extra=launch('correctness_partition_completion',['FULL','LP_G8','LP_G24'],3,
                 trace=True,correctness=True,timeout=1500)
    check_correctness(extra,'partition_completion_validation.json')
    if VERSION!='v2':
        release=launch('release_correctness_'+VERSION,
            ['FULL','LP_AUTO','LP_G8','LP_G16','LP_G24','LP_G48'],3,
            trace=True,light=True,correctness=True,timeout=2000)
        check_correctness(release,'release_validation_'+VERSION+'.json')
    discovery=launch('discovery_native_'+VERSION,[
        'DECODE_ONLY','FULL','CP256','CP512','CP1024','LP_AUTO',
        'LP_G4','LP_G8','LP_G16','LP_G24','LP_G48'],3,timeout=2700)
    discovery_rows=clean_rows(discovery)
    summary=summarize(discovery_rows)
    visual=[r for r in summary if r['modality']=='VISUAL_PREFILL' and r['arm'].startswith('LP_')]
    assert all(r['n']==3 for r in visual)
    acceptable=[r for r in visual if r['peak_itl_ratio']['mean']<=1.5]
    best=min(acceptable,key=lambda r:r['ttft_ms']['mean']) if acceptable else min(
        visual,key=lambda r:r['peak_vsel_ms']['mean'])
    freeze={'selection_contract':str(ROOT/'artifacts/manifests/analysis_contract_v2.json'),
        'frozen_ns':time.time_ns(),'selection_run':str(discovery),
        'best_arm':best['arm'],'best_groups':groups_for(best['arm'],2029),
        'slo_eligible':bool(acceptable),'discovery_summary':summary}
    atomic_json(ROOT/'artifacts/manifests'/('frozen_final_selection_'+VERSION+'.json'),freeze)
    atomic_json(ROOT/'artifacts'/('discovery_native_'+VERSION+'.json'),{'clean_waves':discovery_rows,'clean_summary':summary})
    primary=list(dict.fromkeys(['FULL','CP512','LP_AUTO',best['arm']]))
    final=launch('final_paired_native_'+VERSION,primary,10,paired=True,decode=128,timeout=6000)
    final_rows=clean_rows(final)
    atomic_json(ROOT/'artifacts'/('final_clean_'+VERSION+'.json'),{'clean_waves':final_rows,'clean_summary':summarize(final_rows)})
    components=launch('component_overhead_native_'+VERSION,primary,3,trace=True,light=True,
                      paired=True,overhead=True,decode=128,timeout=3300)
    layers=trace_rows(components);cw=component_waves(layers)
    serving=clean_rows(components)
    atomic_json(ROOT/'artifacts'/('component_trace_'+VERSION+'.json'),{'trace_layers':layers,
        'component_waves':cw,'component_summary':component_summary(cw),
        'overhead_serving_rows':serving})
    routes=launch('route_accounting_native_'+VERSION,['FULL','CP256','CP512','CP1024',
        'LP_AUTO','LP_G8','LP_G16','LP_G24','LP_G48'],3,trace=True,timeout=2700)
    rl=trace_rows(routes);rw=component_waves(rl)
    atomic_json(ROOT/'artifacts'/('route_accounting_'+VERSION+'.json'),{'trace_layers':rl,
        'route_waves':rw,'route_summary':component_summary(rw),
        'timing_status':'ROUTE_DIAGNOSTIC_NOT_HEADLINE; heavy trace timing excluded'})
    atomic_json(ROOT/'artifacts'/('campaign_status_'+VERSION+'.json'),{'status':'MEASUREMENTS_COMPLETE',
        'reporting':'PENDING','runs':[str(x) for x in (args.initial_correctness,extra,
        discovery,final,components,routes)]})
    print('MEASUREMENTS_COMPLETE',flush=True)


if __name__=='__main__':main()
