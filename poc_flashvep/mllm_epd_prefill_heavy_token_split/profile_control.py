"""Measure a fixed-work clean/profile/clean sequence on actual online services."""
import argparse
import asyncio
import json
from pathlib import Path
import random
import time

from .common import ROOT, digest, preserve, write
from .client import execute, resolve_manifest, load_replay_counts
from .encoder_capture_control import rpc
from .metrics import summarize


async def run(args):
    preserve()
    path=resolve_manifest(args.manifest)
    manifest=json.loads(path.read_text());rows=manifest['requests']
    random.Random(args.order_seed).shuffle(rows)
    counts=load_replay_counts(args.replay_from)
    assert all(r['request_id'] in counts for r in rows)
    servers={name:json.loads((ROOT/'state'/f'server_{name}.json').read_text()) for name in args.server_names.split(',')}
    assert all(r['ready'] and not r['finished'] for r in servers.values())
    config=dict(time=time.time(),manifest=str(path),manifest_SHA256=digest(path),
                reference=str(args.replay_from),reference_SHA256=digest(args.replay_from),
                QPS=args.qps,arrival_seed=args.seed,order_seed=args.order_seed,
                servers=servers,source_SHA256=digest(__file__),
                observer_interference_max=.05,primary_quality_status='PENDING',
                evidence='DIAGNOSTIC until both observer and model-quality gates pass',
                observer_kind=args.observer_kind,
                scheduler_state_sampling_interval_ms=getattr(args,'state_interval_ms',0),
                client_host_observer=args.observe_client_host,
                freeze_client_static_heap=args.freeze_client_static_heap,
                metrics=['duration_seconds','TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms'])
    write(ROOT/'configs'/f'{args.name}.json',config)
    stages={};rpc_results={};opened=[]
    async def measure(stage):
        run_name=args.name+'_'+stage
        result=await execute(rows,[args.url],run_name,args.qps,args.seed,counts,False,
            metadata=dict(manifest=str(path),manifest_SHA256=digest(path),evidence='DIAGNOSTIC',
                procedure='Equal-output clean/profile/clean',stage=stage,
                observer=stage=='profile',quality_qualification='PENDING',
                replay_reference_SHA256=digest(args.replay_from),order_seed=args.order_seed),
            observe_host=args.observe_client_host,freeze_static_heap=args.freeze_client_static_heap)
        assert len(result)==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in result)
        assert all(r['output_tokens']==counts[r['request_id']] for r in result)
        stages[stage]=summarize(run_name)
        print('CPC_STAGE_COMPLETE',stage,stages[stage]['duration_seconds'],flush=True)
    await measure('clean_before')
    try:
        for url in args.worker_urls.split(','):
            reply=await asyncio.to_thread(rpc,url,args.observer_kind+'/begin',dict(name=args.name,include_copies=True,
                state_interval_ms=getattr(args,'state_interval_ms',0)))
            rpc_results[url]=dict(begin=reply);opened.append(url)
        await measure('profile')
    finally:
        for url in reversed(opened):
            try:rpc_results[url]['end']=await asyncio.to_thread(rpc,url,args.observer_kind+'/end',{})
            except Exception as error:
                rpc_results[url]['close_error']=type(error).__name__+': '+str(error)
        write(ROOT/'state'/f'{args.name}_collector_rpc.json',rpc_results)
    await measure('clean_after')
    comparison={}
    for key in config['metrics']:
        left,right,profile=(stages[s][key] for s in ['clean_before','clean_after','profile'])
        center=(left+right)/2
        comparison[key]=dict(clean_before=left,profile=profile,clean_after=right,
            profile_relative_difference=profile/center-1,
            clean_relative_spread=abs(right-left)/center,
            within_5_percent=abs(profile/center-1)<=.05 and abs(right-left)/center<=.05)
    if args.observer_kind=='profile':
        collectors_passed=all(item['passed'] for value in rpc_results.values() for item in value['end'])
    elif args.observer_kind=='gpu_enclosures':
        collectors_passed=all(item['events']>0 and item['clock_scale']>0
            for value in rpc_results.values() for item in value['end'])
    else:
        collectors_passed=all(value['end']['api']['events']>0 and all(r['events']>0 for r in value['end']['workers'])
            for value in rpc_results.values())
    value=dict(time=time.time(),name=args.name,stages=stages,comparison=comparison,
               collector_counters_passed=collectors_passed,
               observer_gate_passed=collectors_passed and all(x['within_5_percent'] for x in comparison.values()),
               model_quality_gate='PENDING',evidence='DIAGNOSTIC',observer_kind=args.observer_kind,
               interpretation='CUPTI qualification does not qualify model outputs. Fixed output work does not guarantee identical answer strings.')
    write(ROOT/'analysis'/f'{args.name}_qualification.json',value)
    preserve()
    print('CPC_GATE',value['observer_gate_passed'],json.dumps(comparison),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--url',required=True);p.add_argument('--worker-urls',required=True)
    p.add_argument('--server-names',required=True);p.add_argument('--manifest',type=Path)
    p.add_argument('--replay-from',type=Path,required=True);p.add_argument('--qps',type=float,default=1)
    p.add_argument('--observer-kind',choices=['profile','timeline','gpu_enclosures'],default='profile')
    p.add_argument('--state-interval-ms',type=float,default=0,help='Scheduler state sampling only. GPU kernels remain fully recorded.')
    p.add_argument('--observe-client-host',action='store_true')
    p.add_argument('--freeze-client-static-heap',action='store_true')
    p.add_argument('--seed',type=int,default=2026112402);p.add_argument('--order-seed',type=int,default=2026112404)
    asyncio.run(run(p.parse_args()))


if __name__=='__main__':main()
