"""Actual embedding-ready prefill probes. They do not complete client requests."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time
import urllib.request

from .client import execute,payload,resolve_manifest,load_replay_counts
from .common import ROOT,REPO,PYTHON,digest,experiment_deadline,preserve,write
from .encoder_llm_memory_control import info
from .frontend_supply_control import ROLES,wait_ready,retire
from .llm_probe_control import rpc
from .ipc_ec_control import launch as launch_stock,ownership_empty,stop_proxy as stop_original_proxy
from .metrics import summarize
from .native_common_control import set_common_ipc_policy
from .prefix_probe_contract import KEY
from .prefix_quantum_correctness_control import compare
from .source_snapshot import snapshot
from .runtime_endpoint_identity import require_identity,require_free_ports
from .token_confirmation_control import smoke
from .token_split import TEACHER_KEY

TAG='interference_PROBE_C40R1'
URL='http://127.0.0.1:18845'
QUANTA=[128,512]


def launch_native(tag,names):
    ec=Path('/dev/shm')/f'mllm_epd_20261124_{tag}';assert not ec.exists()
    for role,gpu,port,side in ROLES:
        name=f'{tag}_{role[0].upper()}{gpu}';names.append(name)
        args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
            '--name',name,'--gpu',str(gpu),'--role',role,'--port',str(port),'--side-port',str(side),
            '--ec-path',str(ec),'--observe','--ec-transport','cuda_ipc']
        if role=='encoder':args+=['--encoder-native-ops','--compiled-encoder','--encoder-sync-scheduling','--mm-cache-gb','0','--renderer-workers','2']
        else:args+=['--fixed-reduction','--deterministic-reductions']
        if role=='decode':args+=['--prefix-probe']
        subprocess.run(args,cwd=REPO,check=True)
    wait_ready(names);set_common_ipc_policy()
    return ec


def launch_proxy(tag,names):
    name=tag+'_proxy'
    command=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.interference_probe_proxy',
        '--name',name,'--port','18845']
    lease=json.loads((ROOT/'state'/f'server_{names[0]}.json').read_text())
    with (ROOT/'logs'/f'{name}.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=lease['executed_source_cwd'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state'/f'{name}.json',dict(time=time.time(),pid=process.pid,sid=process.pid,command=command))
    until=time.time()+30
    while True:
        assert process.poll() is None
        try:
            with urllib.request.urlopen(URL+'/health',timeout=2) as response:
                if response.status==200:
                    require_identity(response,name,process)
                    return process
        except OSError:pass
        assert time.time()<until,'Inspect the same live proxy. An observation timeout is not authorization to restart.'
        time.sleep(.2)


def stop_proxy(process):
    if process.poll() is not None:return
    path=Path('/proc',str(process.pid))
    assert path.stat().st_uid==os.getuid() and os.getsid(process.pid)==process.pid
    assert b'mllm_epd_prefill_heavy_token_split.interference_probe_proxy' in (path/'cmdline').read_bytes()
    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20)


async def post(session,url,body,headers=None):
    async with session.post(url,json=body,headers=headers) as response:
        value=await response.json();assert response.status==200,(response.status,str(value)[:1800])
        return value


async def prime(row,tag):
    import aiohttp
    body=payload(row,tag,1,False);key=hashlib.sha256(tag.encode()).hexdigest()
    rpc(18840,'ec_ipc/inline_policy',dict(enabled=False))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
            value=await post(session,'http://127.0.0.1:18840/v1/chat/completions',body,
                {'x-request-id':tag,'x-campaign-metadata-key':key,'x-campaign-metadata-mode':'grid_only'})
        assert value['usage']['prompt_tokens']==row['total_input_tokens']
    finally:rpc(18840,'ec_ipc/inline_policy',dict(enabled=True))
    return dict(body=body,key=key,images=row['image_count'],full_tokens=row['total_input_tokens'])


async def prefix(session,prepared,port,run,quantum,teacher=False,correctness=False):
    body=dict(prepared['body']);body['kv_transfer_params']={TEACHER_KEY if teacher else KEY:
        dict(prefix_tokens=quantum,full_prompt_tokens=prepared['full_tokens'])}
    begin=time.monotonic()
    value=await post(session,f'http://127.0.0.1:{port}/v1/chat/completions',body,
        {'x-request-id':run,'x-campaign-metadata-key':prepared['key'],'x-campaign-metadata-mode':'grid_only'})
    end=time.monotonic()
    ids=value['choices'][0]['token_ids']
    assert len(ids)==value['usage']['completion_tokens']==1
    assert value['usage']['prompt_tokens']==(prepared['full_tokens'] if teacher else quantum)
    record=dict(run=run,port=port,quantum=quantum,teacher=teacher,begin=begin,end=end,
        HTTP_wall_ms=(end-begin)*1000,output=value,task_completion=False,
        evidence='CORRECTNESS_ONLY' if teacher or correctness else 'INTERFERENCE_PROBE',
        scope='Actual original full28 prefix, real primed E embeddings, no imported prefix KV. Discarded sample.')
    write(ROOT/'raw'/f'{run}.json',record)
    return record


async def probe_loop(prepared,arm,quantum,run,finished):
    import aiohttp
    rows=[];attempts=0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
        while not finished.is_set() and len(rows)<32:
            await asyncio.sleep(.5)
            if finished.is_set():break
            attempts+=1
            lease=await post(session,URL+'/campaign/probe/acquire',dict(arm=arm))
            if not lease['accepted']:continue
            failed=True;sample=None
            try:
                sample=await prefix(session,prepared,18843 if arm=='E' else 18842,
                    f'{run}_PROBE_{len(rows):03d}',quantum)
                failed=False
            finally:
                response=dict(token=lease['token'],failed=failed)
                if sample is not None:response.update(output_token_ids=sample['output']['choices'][0]['token_ids'],completion_tokens=1)
                release=await post(session,URL+'/campaign/probe/release',response)
            rows.append(dict(arm=arm,quantum=quantum,lease=lease,sample=sample,release=release,
                original_encoding_priority_at_accept=lease['before']['native_state']['encoder_pending']==0 if arm=='E' else None))
    return dict(accepted=len(rows),attempted=attempts,rejected=attempts-len(rows),rows=rows,
        task_completion=False,evidence='INTERFERENCE_PROBE',
        caveat='Embedding-ready helper work. Priming/weights are not a serving optimization. No new real client request is completed.')


async def measure(tag,label,rows,output,prepared=None,arm=None,quantum=0):
    run=f'{tag}_{label}';finished=asyncio.Event();task=None
    if arm is not None:task=asyncio.create_task(probe_loop(prepared,arm,quantum,run,finished))
    try:
        values=await execute(rows,[URL],run,2.,2026112402,output,metadata=dict(
            evidence='DIAGNOSTIC_WARMUP' if 'warmup' in label else 'MEASURED_ONLINE with INTERFERENCE_PROBE',
            task_quality='PENDING',GPU_profiler='OFF',CPU_tensor_capture='OFF',probe_arm=arm,quantum=quantum,
            no_client_task_completed_by_probe=True,scope='Main100 still runs original online E/P/D. '
                'Only an unrelated calibration prefix is primed. This is not actual request offloading or throughput gain.'))
    finally:
        finished.set()
        if task is not None:
            probe=await task
            write(ROOT/'analysis'/f'{run}_probes.json',probe)
    assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' for r in values)
    metric=summarize(run)
    with urllib.request.urlopen(URL+'/campaign/probe/state',timeout=30) as response:state=json.load(response)
    assert not state['leases'] and state['failure'] is None
    print('INTERFERENCE_RUNTIME_METRICS',run,{k:metric[k] for k in
        ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','gate_attainment']},flush=True)
    return dict(label=label,arm=arm,quantum=quantum,metrics=metric)


def wait_for_study():
    value=json.loads((ROOT/'state/ec_DOCUMENT_C39R1_controller.json').read_text())
    path=Path('/proc',str(value['pid']))
    while path.exists():
        assert path.stat().st_uid==os.getuid()
        assert b'mllm_epd_prefill_heavy_token_split.document_stratum_control' in (path/'cmdline').read_bytes()
        assert time.time()<experiment_deadline();time.sleep(1)
    for file in (ROOT/'state').glob('server_ec_DOCUMENT_C39R1*.json'):
        if file.name.endswith('.source_snapshot.json'):continue
        lease=json.loads(file.read_text())
        assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()


def stock_start(index,rows,counts):
    tag=f'{TAG}_B{index}_STOCK';names=[];proxy=None;results=[]
    try:
        names,proxy=launch_stock(tag,'cuda_ipc',owned_names=names)
        set_common_ipc_policy();url='http://127.0.0.1:18847'
        asyncio.run(smoke(tag,'NativeStrong',url))
        for mode,output in [('warmup',counts),('natural',None),('replay',counts)]:
            run=f'{tag}_{mode}'
            values=asyncio.run(execute(rows,[url],run,2.,2026112402,output,metadata=dict(
                evidence='DIAGNOSTIC_WARMUP' if mode=='warmup' else 'MEASURED_ONLINE NativeStrong control',
                block=index,method='NativeStrong',task_quality='PENDING',no_probe=True,
                compiler='Unchanged stock',D_admission='Original, no extra cap',GPU_observer='OFF')))
            assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' for r in values)
            ownership_empty(run,'cuda_ipc');metrics=summarize(run)
            results.append(dict(label='stock_'+mode,metrics=metrics))
        write(ROOT/'analysis'/f'{tag}_summary.json',dict(time=time.time(),status='COMPLETE',index=index,results=results,
            scope='Prospectively matched start block for NativeStrong. Common/probe stages follow in a separate initialization.'))
        return results
    except BaseException as error:
        write(ROOT/'state'/f'{tag}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_cohorts=len(results),restart_automatically=False))
        raise
    finally:
        if proxy is not None:stop_original_proxy(proxy)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])


def run_start(index,rows,counts):
    import aiohttp
    tag=f'{TAG}_B{index}';names=[];helper=None;proxy=None;prepared=None;results=[]
    try:
        ec=launch_native(tag,names);proxy=launch_proxy(tag,names)
        asyncio.run(smoke(tag,'NativeCommon',URL))
        results.append(asyncio.run(measure(tag,'plain_warmup',rows,counts)))
        before={str(port):info(port) for port in [18840,18841,18842]}
        for mode,output in [('natural',None),('replay',counts)]:
            results.append(asyncio.run(measure(tag,'plain_'+mode,rows,output)))
        helper=tag+'_H4'
        args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
            '--name',helper,'--gpu','4','--role','pd','--port','18843','--side-port','19843',
            '--ec-path',str(ec),'--observe','--ec-transport','cuda_ipc','--co-resident-with',tag+'_E4',
            '--kv-cache-gib','8','--gpu-memory-utilization','.35','--prefix-probe',
            '--fixed-reduction','--deterministic-reductions']
        subprocess.run(args,cwd=REPO,check=True);wait_ready([helper])
        for port in [18843,18842]:
            rpc(port,'ec_ipc/ack_policy',dict(mode='file'));rpc(port,'boundary_teacher/install',{})
        after={str(port):info(port) for port in [18840,18841,18842,18843]}
        asyncio.run(smoke(tag+'_WITH_HELPER','NativeCommon',URL))
        row=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests'][6]
        prepared=asyncio.run(prime(row,tag+'_PRIMED_CALIBRATION'))
        async def correctness():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
                for arm,port in [('E',18843),('D',18842)]:
                    for quantum in QUANTA:
                        teacher=f'{tag}_{arm}_q{quantum}_TEACHER';actual=f'{tag}_{arm}_q{quantum}_PREFIX'
                        for label,name,is_teacher in [('teacher',teacher,True),('probe',actual,False)]:
                            rpc(port,'llm_probe/begin',dict(name=name,max_logits=1,include_decode_inputs=True))
                            try:await prefix(session,prepared,port,name,quantum,is_teacher,correctness=True)
                            finally:rpc(port,'llm_probe/end',{})
                        compare(teacher,actual,quantum)
        asyncio.run(correctness())
        results.append(asyncio.run(measure(tag,'co_resident_warmup',rows,counts)))
        # All conditions were frozen before the first probe result. Every
        # baseline and probe condition uses the same coordinator and D policy.
        conditions=[('clean_before',None,0),('E128','E',128),('E512','E',512),
            ('D128','D',128),('D512','D',512),('clean_after',None,0)]
        for label,arm,quantum in conditions:
            for mode,output in [('natural',None),('replay',counts)]:
                assert time.time()<experiment_deadline()
                results.append(asyncio.run(measure(tag,label+'_'+mode,rows,output,prepared,arm,quantum)))
                write(ROOT/'analysis'/f'{tag}_partial.json',dict(time=time.time(),results=results))
        release=rpc(18840,'ec_ipc/release',dict(key=prepared['key']))
        assert release[0]['live_images']==release[0]['outstanding_exports']==0;prepared=None
        memory=dict(before=before,with_helper=after,final={str(p):info(p) for p in [18840,18841,18842,18843]})
        write(ROOT/'analysis'/f'{tag}_summary.json',dict(time=time.time(),status='COMPLETE',index=index,
            results=results,memory=memory,evidence='MEASURED_ONLINE and INTERFERENCE_PROBE',
            real_client_prefill_offload=False,full_task_quality='PENDING',safe_headroom='REQUIRES_N3_GATE_RECALCULATION'))
        return results
    except BaseException as error:
        write(ROOT/'state'/f'{tag}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_cohorts=len(results),restart_automatically=False))
        raise
    finally:
        if proxy is not None:stop_proxy(proxy)
        if helper is not None and (ROOT/'state'/f'server_{helper}.json').exists():retire([helper])
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])


def main():
    global TAG
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-c39',action='store_true')
    parser.add_argument('--tag',default=TAG)
    args=parser.parse_args();assert re.fullmatch(r'interference_PROBE_C40R[0-9]+',args.tag)
    TAG=args.tag;preserve();assert time.time()<experiment_deadline()
    prior_idle=ROOT/'analysis/ec_LOW_CUPTI_C25_CPC_q0.5_v0_qualification.json'
    assert json.loads(prior_idle.read_text())['observer_gate_passed']
    prior_memory=ROOT/'analysis/encoder_LLM_MEMORY_C16.json'
    assert json.loads(prior_memory.read_text())['status']=='OBSERVED_CO_RESIDENCY_FIT'
    prior_prefix=ROOT/'analysis/prefix_QUANTA_C22R1_summary.json'
    assert json.loads(prior_prefix.read_text())['strict_cases']==10
    manifest=resolve_manifest(None);rows=json.loads(manifest.read_text())['requests'];random.Random(2026112404).shuffle(rows)
    reference=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl';counts=load_replay_counts(reference)
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    source=snapshot(TAG)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_ONLINE_INTERFERENCE',source_archive_SHA256=source['archive_SHA256'],
        manifest_SHA256=digest(manifest),model_revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',GPU_group=[4,5,6],
        independent_initializations=3,QPS=2.,quanta=QUANTA,max_probes_per_cohort=32,poll_interval_seconds=.5,
        main_encoding='Actual online original E on every Main100 request. Calibration probe alone uses a primed embedding.',
        no_task_speedup_claim=True,extra_full_LLM_on_E=True,additional_duplicate_vision_counted=True,helper_KV_GiB=8,
        conditions=['PlainNative','CoResidentNoProbeBefore','E128','E512','D128','D512','CoResidentNoProbeAfter'],
        additional_control='Fresh unchanged NativeStrong initialization in each prospective block, before Common Native/probe initialization.',
        prior_unstarted_plan='interference_PROBE_C40: preserved as pre-GPU NOT_RUN. Same dataset, quanta, sampling, SLO and probe acceptance rules.',
        numerical='Common compiled BI0,original BF16/weights/attention. Prefix probe max1 sample discarded.',
        correctness='Unmarked Native N1/N8 output regression, then E/D128/512 same-GPU teacher all28 KV strict before timing.',
        protection='No new probe when P has no HTTP work. E accepts only with no pending E HTTP request. '
            'D accepts only with zero active or pending D HTTP admission. Native D waits until actual probe completion.',
        nonpreemption='A new Encode arrival can wait for an already launched helper kernel. Stream priority is not preemption.',
        relative_gates='Paired N3 Encoder HTTP service p95, ITLp99 and TPOTp95 CI upper<=1.05. Absolute Decode attainment>=95%.',
        end_to_end_protection='Also report TTFTp50/p95 NI and SLO-goodput against each Native control. '
            'Encoder HTTP completion is not guaranteed GPU completion. Do not call it exclusive encoder service.',
        observer='OFF during timing. HTTP service enclosure includes queue and host cost, not GPU-exclusive time.',
        prior_evidence_SHA256=dict(qualified_low_load_idle=digest(prior_idle),memory_fit=digest(prior_memory),prefix_correctness=digest(prior_prefix)),
        evidence_limit='Qualified low-load idle does not prove QPS2 safe headroom. This experiment tests conservative '
            'current-state probes empirically. No future window is assumed. Main100 is not yet confirmed prefill-heavy.',
        exclusions='No imported prefix KV, suffix continuation,3way partition, actual Main task completion by probes, '
            'maximum encoder-batch claim or sustainable-capacity claim.',
        quality='PENDING. No external task-answer extraction without approval.',wait_for_c39=args.wait_for_c39))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_C39' if args.wait_for_c39 else 'STARTING'))
    if args.wait_for_c39:wait_for_study()
    require_free_ports([18840,18841,18842,18843,18845,18847])
    results=[]
    try:
        for index in range(3):
            assert time.time()<experiment_deadline()
            strong=stock_start(index,rows,counts)
            results.append(dict(NativeStrong=strong,common_and_probes=run_start(index,rows,counts)))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_starts=len(results),restart_automatically=False))
        raise
    write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE',starts=results,
        evidence='INTERFERENCE_PROBE',online_serving_gain='NOT_TESTED',encode_safe_headroom='REQUIRES_ANALYSIS'))
    write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),status='COMPLETE'))
    preserve()


if __name__=='__main__':main()
