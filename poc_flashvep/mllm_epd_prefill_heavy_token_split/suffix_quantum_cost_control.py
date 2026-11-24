"""Real imported-KV suffix costs, not prefix probes or online serving speedup."""
import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np

from . import token_correctness_control as single
from .batch_token_correctness_control import headers,post
from .client import payload
from .common import ROOT,digest,preserve,write,experiment_deadline
from .frontend_supply_control import retire
from .ipc_ec_control import stop_proxy,ownership_empty
from .llm_probe_control import rpc
from .source_snapshot import snapshot
from .runtime_endpoint_identity import require_free_ports
from .token_split import PLAN_KEY,POLICY_KEY

TAG='suffix_QUANTA_C41'
INDICES=[0,6,31]
QUANTA=[128,256,512,1024,2048]


async def trial(session,row,quantum,run):
    prefix=row['total_input_tokens']-quantum;body=payload(row,run,1,False)
    encoded=dict(body,kv_transfer_params={POLICY_KEY:dict(suffix_fraction=0.,fixed_prefix=prefix)})
    encoder_begin=time.monotonic()
    result=await post(session,18840,'/v1/chat/completions',encoded,headers(run,'E:full'))
    assert result['usage']['prompt_tokens']==row['total_input_tokens']
    encoder_end=time.monotonic()
    plan=dict(prefix_tokens=prefix,full_prompt_tokens=row['total_input_tokens'])
    source=dict(body,kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,
        remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None,**{PLAN_KEY:plan}))
    source_begin=time.monotonic()
    output=await post(session,18841,'/v1/chat/completions',source,headers(run,'P'))
    source_end=time.monotonic();params=output['kv_transfer_params']
    assert params['remote_num_tokens']==prefix and params['remote_block_ids']
    destination=dict(body,kv_transfer_params=dict(params,**{PLAN_KEY:plan}))
    destination_begin=time.monotonic()
    measured=await post(session,18842,'/v1/chat/completions',destination,headers(run,'D'))
    destination_end=time.monotonic()
    assert measured['usage']['completion_tokens']==len(measured['choices'][0]['token_ids'])==1
    assert measured['usage']['prompt_tokens']==row['total_input_tokens']
    write(ROOT/'raw'/f'{run}_isolated_response.json',dict(time=time.time(),E_response=result,P_response=output,
        D_response=measured,D_begin=destination_begin,D_end=destination_end,request_headers=headers(run,'D'),
        evidence='ISOLATED_GPU_MEASUREMENT or DIAGNOSTIC after qualification',task_completion=False,
        scope='Non-streaming one-token completion wall. Not a C3 SSE TTFT or an end-to-end client request.'))
    import hashlib
    released=await post(session,18840,'/campaign/ec_ipc/release',dict(key=hashlib.sha256(run.encode()).hexdigest()))
    assert released[0]['live_images']==released[0]['outstanding_exports']==0
    return dict(run=run,request_id=row['request_id'],input_tokens=row['total_input_tokens'],prefix_tokens=prefix,
        suffix_tokens=quantum,D_first_token_HTTP_ms=1000*(destination_end-destination_begin),
        E_HTTP_ms=1000*(encoder_end-encoder_begin),P_HTTP_ms=1000*(source_end-source_begin),
        logical_prefix_KV_bytes=prefix*28*2*4*128*2,
        logical_suffix_GEMM_rows=28*quantum,logical_suffix_attention_pairs_per_layer=quantum*prefix+quantum*(quantum+1)//2,
        task_completion=False,actual_client_requests_in_flight=1,
        scope='ISOLATED_GPU_MEASUREMENT if observer gate passes. Non-streaming one-token D completion includes '
            'actual NIXL import, embedding consumption, admission, full28 suffix, sampling and delivery. '
            'Not C3 SSE TTFT or GPU-exclusive time.',
        KV_bytes_scope='STRUCTURAL_ACCOUNTING logical prefix entries. Not actual padded DMA bytes.')


def wait_previous(tag='interference_PROBE_C40R1'):
    value=json.loads((ROOT/'state'/f'{tag}_controller.json').read_text());proc=Path('/proc',str(value['pid']))
    while proc.exists():
        assert proc.stat().st_uid==os.getuid()
        assert b'mllm_epd_prefill_heavy_token_split.interference_probe_control' in (proc/'cmdline').read_bytes()
        assert time.time()<experiment_deadline();time.sleep(1)
    assert (ROOT/'state'/f'{tag}_controller_complete.json').exists(), 'Inspect the failed predecessor. Do not automatically proceed after its failure.'
    for path in (ROOT/'state').glob(f'server_{tag}*.json'):
        if path.name.endswith('.source_snapshot.json'):continue
        lease=json.loads(path.read_text());assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()


async def costs(pool):
    import aiohttp
    records=[];collector=None
    base=[(index,quantum) for index in INDICES for quantum in QUANTA]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
        for stage in ['warmup','clean_before','profile','clean_after']:
            schedule=base.copy() if stage=='warmup' else base*9
            random.Random(2026112511).shuffle(schedule)
            if stage=='profile':collector=dict(begin=rpc(18842,'gpu_enclosures/begin',dict(name=TAG,state_interval_ms=50)))
            try:
                for repetition,(index,quantum) in enumerate(schedule):
                    assert time.time()<experiment_deadline()
                    run=f'{TAG}_{stage}_i{index}_q{quantum}_r{repetition:03d}'
                    records.append(dict(stage=stage,index=index,**await trial(session,pool[index],quantum,run)))
            finally:
                if stage=='profile':
                    collector['end']=rpc(18842,'gpu_enclosures/end',{})
                    write(ROOT/'state'/f'{TAG}_collector.json',collector)
            print('SUFFIX_COST_STAGE',stage,len(schedule),flush=True)
    checks=[]
    for index,quantum in base:
        values={stage:[r['D_first_token_HTTP_ms'] for r in records if r['index']==index and r['suffix_tokens']==quantum and r['stage']==stage]
            for stage in ['clean_before','profile','clean_after']}
        assert all(len(v)==9 for v in values.values())
        for percentile in [50,95]:
            left,profile,right=[float(np.percentile(values[stage],percentile)) for stage in ['clean_before','profile','clean_after']]
            center=(left+right)/2;delta=profile/center-1;spread=abs(left-right)/center
            checks.append(dict(index=index,input_tokens=pool[index]['total_input_tokens'],suffix_tokens=quantum,percentile=percentile,
                clean_before_ms=left,profile_ms=profile,clean_after_ms=right,
                observer_difference=delta,clean_spread=spread,within5percent=abs(delta)<=.05 and spread<=.05))
    for label,rows in [('requests',records),('qualification',checks)]:
        with (ROOT/'analysis'/f'{TAG}_{label}_v0.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    passed=all(r['within5percent'] for r in checks) and collector['end'][0]['events']>0
    write(ROOT/'analysis'/f'{TAG}_cost_v0.json',dict(time=time.time(),observer_gate_passed=passed,checks=checks,
        evidence='ISOLATED_GPU_MEASUREMENT' if passed else 'DIAGNOSTIC',independent_starts=1,
        samples_per_cell_per_stage=9,task_completion=False,online_interference=False,
        transfer_DMA_exclusive_time='NOT_MEASURED',GPU_exclusive_suffix_kernel_time='NOT_MEASURED',
        scope='Actual imported prefix KV and original MLLM visual embeddings. HTTP cost and separate CUDA-event '
            'forward enclosures are not exclusive operator or transfer costs. No future idle-window or C3 speedup claim.'))
    return passed


def main():
    global TAG
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-c40',action='store_true')
    parser.add_argument('--tag',default=TAG);parser.add_argument('--predecessor-tag',default='interference_PROBE_C40R1')
    args=parser.parse_args()
    assert re.fullmatch(r'suffix_QUANTA_C41(?:R[0-9]+)?',args.tag)
    assert re.fullmatch(r'interference_PROBE_C40R[0-9]+',args.predecessor_tag)
    TAG=args.tag;preserve();assert time.time()+1800<experiment_deadline()
    pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    source=snapshot(TAG)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_ACTUAL_SUFFIX_COSTS',source_archive_SHA256=source['archive_SHA256'],
        GPU_group=[4,5,6],indices=INDICES,input_tokens=[pool[i]['total_input_tokens'] for i in INDICES],suffix_quanta=QUANTA,
        geometry='Original prompt remains unchanged. Prefix=L-quantum, all28 KV imported by NIXL, exact MRoPE and visual embedding continuation.',
        correctness='All15 same-chunk full teachers and token executions must pass strict all28 KV and natural full greedy output before cost timing.',
        timing='Separate max1 ignoreEOS discarded samples. D timing starts only after P prefix is ready. No other client is active.',
        observer='D-only CUDA-event forward enclosures in clean/profile/clean. All30 cell-percentile checks must pass5percent.',
        no_Main100_change=True,no_new_modality_method=True,no_online_speedup_claim=True,
        wait_for_c40=args.wait_for_c40,predecessor_tag=args.predecessor_tag,prefix_probe_cost_not_reused_as_suffix_cost=True))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_C40' if args.wait_for_c40 else 'STARTING'))
    if args.wait_for_c40:wait_previous(args.predecessor_tag)
    require_free_ports([18840,18841,18842,18848,18849])
    assert time.time()+1800<experiment_deadline()
    single.TAG=TAG;single.EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
    names=[];teacher=None;proxy=None;comparisons=[]
    try:
        single.launch_models(names,fixed_reduction=True,deterministic_reductions=True)
        teacher,teacher_url=single.launch_proxy(TAG+'_teacher_proxy','e_pd',18848,names)
        for index in INDICES:
            for quantum in QUANTA:
                assert time.time()<experiment_deadline()
                row=pool[index];prefix=row['total_input_tokens']-quantum
                reference_run=f'{TAG}_N1_i{index}_q{quantum}_TEACHER'
                actual_run=f'{TAG}_N1_i{index}_q{quantum}_TOKEN'
                reference=single.capture(row,reference_run,teacher_url,[(18841,'P')],prefix)
                proxy,url=single.launch_proxy(actual_run+'_proxy','epd',18849,names,prefix=prefix)
                try:
                    actual=single.capture(row,actual_run,url,[(18841,'P'),(18842,'D')])
                    comparisons.append(single.compare(row,prefix,reference_run,actual_run,reference,actual))
                finally:stop_proxy(proxy);proxy=None
        stop_proxy(teacher);teacher=None
        write(ROOT/'analysis'/f'{TAG}_correctness_v0.json',dict(time=time.time(),passed=True,cases=comparisons,evidence='CORRECTNESS_ONLY'))
        passed=asyncio.run(costs(pool))
        ownership_empty(TAG,'cuda_ipc')
        write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),status='COMPLETE',observer_gate_passed=passed))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_correctness_cases=len(comparisons),restart_automatically=False))
        raise
    finally:
        if proxy is not None:stop_proxy(proxy)
        if teacher is not None:stop_proxy(teacher)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])
    preserve()


if __name__=='__main__':main()
