"""Same-group Native EPD CPU-versus-CUDA-IPC transport correctness and A/B."""
import asyncio
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import urllib.request

from .common import ROOT,REPO,PYTHON,digest,preserve,write
from .client import execute,resolve_manifest,load_replay_counts
from .frontend_supply_control import wait_ready,retire,ROLES
from .llm_probe_control import rpc
from .metrics import summarize
from .source_snapshot import snapshot

URL='http://127.0.0.1:18847'


def launch(tag,transport):
    ec=Path('/dev/shm')/f'mllm_epd_20261124_{tag}'
    assert not ec.exists(),'Every worker group requires a new EC ownership epoch'
    names=[]
    for role,gpu,port,side in ROLES:
        name=f'{tag}_{role[0].upper()}{gpu}';names.append(name)
        args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
            '--name',name,'--gpu',str(gpu),'--role',role,'--port',str(port),'--side-port',str(side),
            '--ec-path',str(ec),'--observe','--ec-transport',transport,'--encoder-api','http://127.0.0.1:18840']
        if role=='encoder':
            args+=['--encoder-native-ops','--compiled-encoder','--encoder-sync-scheduling',
                   '--mm-cache-gb','0','--renderer-workers','2']
        subprocess.run(args,cwd=REPO,check=True)
    wait_ready(names)
    proxy_name=tag+'_proxy'
    args=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.proxy',
        '--mode','epd','--encoder','http://127.0.0.1:18840','--prefill','http://127.0.0.1:18841',
        '--decoders','http://127.0.0.1:18842','--port','18847','--name',proxy_name,'--processor-handoff','grid_only']
    if transport=='cuda_ipc':args+=['--release-encoder-ipc']
    with (ROOT/'logs'/f'{proxy_name}.log').open('xb') as log:
        first_lease=json.loads((ROOT/'state'/f'server_{names[0]}.json').read_text())
        proxy=subprocess.Popen(args,cwd=first_lease.get('executed_source_cwd',str(REPO)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state'/f'{proxy_name}.json',dict(pid=proxy.pid,sid=proxy.pid,command=args,models=names,transport=transport))
    until=time.time()+30
    while True:
        assert proxy.poll() is None,'Owned proxy terminated'
        try:
            with urllib.request.urlopen(URL+'/health',timeout=2) as response:
                if response.status==200:break
        except OSError:pass
        if time.time()>until:raise TimeoutError('Inspect the same owned proxy; do not start another one')
        time.sleep(.5)
    return names,proxy


def stop_proxy(proxy):
    if proxy.poll() is not None:return
    proc=Path('/proc',str(proxy.pid))
    assert proc.stat().st_uid==os.getuid() and os.getsid(proxy.pid)==proxy.pid
    assert b'mllm_epd_prefill_heavy_token_split.proxy' in (proc/'cmdline').read_bytes()
    os.killpg(proxy.pid,signal.SIGTERM);proxy.wait(timeout=20)


def ownership_empty(tag,transport):
    if transport!='cuda_ipc':return None
    with urllib.request.urlopen('http://127.0.0.1:18840/campaign/ec_ipc/state',timeout=60) as response:
        value=json.load(response)
    assert len(value)==1
    state=value[0]
    write(ROOT/'state'/f'{tag}_ownership.json',dict(time=time.time(),state=state))
    assert state['live_images']==state['outstanding_exports']==0,state
    assert state['saved_count']==state['released_count'],state
    return state


async def correctness(tag,transport):
    path=ROOT/'data/calibration_manifest_template_v2.json'
    all_rows=json.loads(path.read_text())['requests']
    teacher_path=ROOT/'raw/frontend_B5_cache0_T2_N8_natural_v0_requests.jsonl'
    teacher={r['request_id']:r for r in map(json.loads,teacher_path.read_text().splitlines())}
    comparisons=[]
    for suffix,indices in [('N1',[0]),('N8',[0,1,4,6,12,17,20,31])]:
        rows=[all_rows[i] for i in indices]
        values=await execute(rows,[URL],f'{tag}_{suffix}_natural_v0',.25,2026112402,None,True,
            metadata=dict(evidence='CORRECTNESS_ONLY',manifest_SHA256=digest(path),transport=transport))
        for value in values:
            ref=teacher[value['request_id']]
            comparisons.append(dict(phase=suffix,request_id=value['request_id'],status=value['status'],
                strict_tokens=value['status']=='MEASURED_ONLINE' and value['token_ids']==ref['token_ids'],
                strict_prompt=value.get('prompt_token_ids')==ref.get('prompt_token_ids')))
        proof=dict(time=time.time(),evidence='CORRECTNESS_ONLY',rows=comparisons,
            passed=len(values)==len(rows) and all(r['strict_tokens'] and r['strict_prompt'] for r in comparisons),
            teacher_SHA256=digest(teacher_path),scope='Serial same-group Native EPD. Not a full cross-batch task-quality gate.')
        write(ROOT/'analysis'/f'{tag}_{suffix}_correctness.json',proof)
        assert proof['passed'],'Stop timing if the exact transport changes the same-geometry output'
        ownership_empty(f'{tag}_{suffix}',transport)
    print('EC_TRANSPORT_N1_N8_PASS',tag,flush=True)


def kv_capture(tag,transport):
    path=ROOT/'data/calibration_manifest_template_v2.json'
    row=json.loads(path.read_text())['requests'][6];active=[]
    try:
        for port,role in [(18841,'P'),(18842,'D')]:
            rpc(port,'llm_probe/begin',dict(name=f'{tag}_KV_{role}',max_logits=16));active.append((port,role))
        values=asyncio.run(execute([row],[URL],tag+'_KV_Mantis6_v0',.25,2026112402,None,True,
            metadata=dict(evidence='CORRECTNESS_ONLY',scope='Extra CPU state copies exclude all performance timing.')))
        assert len(values)==1 and values[0]['status']=='MEASURED_ONLINE'
    finally:
        for port,role in reversed(active):write(ROOT/'state'/f'{tag}_KV_{role}_close.json',rpc(port,'llm_probe/end',{}))
    ownership_empty(tag+'_KV',transport)
    print('EC_TRANSPORT_KV_CAPTURED',tag,flush=True)


def compare_kv(cpu,ipc):
    import torch
    from safetensors.torch import load_file
    tensors={};manifests={}
    for tag in [cpu,ipc]:
        for role in ['P','D']:
            value=json.loads((ROOT/'raw/llm_reference'/f'{tag}_KV_{role}/manifest.json').read_text())
            assert digest(value['tensor_file'])==value['tensor_SHA256']
            tensors[tag,role]=load_file(value['tensor_file']);manifests[tag,role]=value
    fields={key:torch.equal(tensors[cpu,'P'][key],tensors[ipc,'P'][key]) for key in
        ['prompt_token_ids','full_prompt_mrope','inputs_embeds_0','positions_0','hidden_0']}
    layers=[]
    for layer in range(28):
        record=dict(layer=layer)
        for tag in [cpu,ipc]:
            imported=manifests[tag,'D']['snapshot']['event']['start']
            record[tag+'_import_strict']=torch.equal(tensors[tag,'P'][f'prompt_KV_{layer}'][:imported],
                tensors[tag,'D'][f'prompt_KV_{layer}'][:imported])
        record['cross_transport_prompt_KV_strict']=torch.equal(tensors[cpu,'D'][f'prompt_KV_{layer}'],
            tensors[ipc,'D'][f'prompt_KV_{layer}'])
        layers.append(record)
    proof=dict(time=time.time(),evidence='CORRECTNESS_ONLY',prepared_fields=fields,layers=layers,
        passed=all(fields.values()) and all(all(v for k,v in row.items() if k!='layer') for row in layers),
        scope='One real Mantis request, all28 layers and original MRoPE. Full task quality remains separate.')
    target=ROOT/'analysis'/f'{cpu}_{ipc}_KV_Mantis6_v0.json'
    assert not target.exists(),'Do not overwrite a prior KV comparison'
    write(target,proof)
    assert proof['passed'],'Transport changed a prepared input or prompt KV state'
    print('EC_TRANSPORT_ALL28_KV_PASS',flush=True)


async def pilot(tag,transport):
    path=resolve_manifest(None);rows=json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    reference=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    counts=load_replay_counts(reference);summaries=[]
    for suffix,qps,output in [('warmup',2,counts),('replay_q2',2,counts),('replay_q3',3,counts),('natural_q2',2,None)]:
        run=f'{tag}_{suffix}_v0'
        values=await execute(rows,[URL],run,qps,2026112402,output,
            metadata=dict(evidence='DIAGNOSTIC_WARMUP' if suffix=='warmup' else 'DIAGNOSTIC_C3',
                manifest_SHA256=digest(path),transport=transport,quality_gate='PENDING',
                scope='Common Native EPD transport control, not a new token-split method. One independent start per arm.'))
        assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' for r in values)
        value=summarize(run);summaries.append(value);ownership_empty(run,transport)
        print('EC_TRANSPORT_PILOT',run,{k:value[k] for k in ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','gate_attainment']},flush=True)
    write(ROOT/'analysis'/f'{tag}_pilot_summary.json',dict(time=time.time(),rows=summaries,
        independent_starts=1,evidence='DIAGNOSTIC_C3',quality_gate='PENDING',confidence_interval='NOT_AVAILABLE'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-ipc-tag');args=parser.parse_args()
    preserve();archive=snapshot(args.resume_ipc_tag or 'native_EC_transport_control_v0')
    plan=ROOT/'configs/native_EC_transport_control_v0.json'
    if not args.resume_ipc_tag:
        assert not plan.exists()
        write(plan,dict(time=time.time(),status='FROZEN_BEFORE_NATIVE_CANDIDATE',
        source_archive_SHA256=archive['archive_SHA256'],GPU_group=[4,5,6],
        arms=['ec_CPU_C6','ec_IPC_C7'],one_factor='Original E embedding transport',
        fixed=['Model and processor revision','BF16 BI0','Full28 LLM and original vision encoder',
            'E cache0/T2/B114688 sync','P/D B8192/S1024 async and KV42GiB/Block128','Grid metadata reuse',
            'Main100 and order/arrival seeds','Immediate client evidence recording','All SLO thresholds'],
        differences='CUDA IPC retains original E GPU backing storage until successful downstream completion. '
            'Fresh per-consumer descriptors and GPU clone-completion ACKs are mandatory. No unused D image copy is forced.',
        bound='Maximum12GiB live unique E backing storage; exceeding it fails the experiment without freeing live tensors.',
        hypotheses=['CPU staging may delay E supply.','IPC open/RPC costs may offset removed CPU copies.',
            'Original E computation or other host work may remain the bottleneck.'],
        no_generic_token_split=True,no_online_prefill_insertion=True,primary_quality='PENDING',independent_starts_per_arm=1))
    cpu='ec_CPU_C6';ipc=args.resume_ipc_tag or 'ec_IPC_C7'
    if args.resume_ipc_tag:
        assert plan.exists() and args.resume_ipc_tag.replace('_','').isalnum()
        assert json.loads((ROOT/'analysis'/f'{cpu}_N8_correctness.json').read_text())['passed']
        baseline=json.loads((ROOT/'analysis'/f'{cpu}_pilot_summary.json').read_text())
        assert len(baseline['rows'])==4 and all(r['completed_count']==100 for r in baseline['rows'])
        resume=ROOT/'configs'/f'{ipc}_resume.json';assert not resume.exists()
        write(resume,dict(time=time.time(),source_archive_SHA256=archive['archive_SHA256'],
            parent_plan_SHA256=digest(plan),preserved_failed_candidate='ec_IPC_C7',
            correction='Preserve the complete opaque Torch allocator handle. Real installed handles have66 bytes, not64.',
            readiness='Original producer-stream ready event followed by a dedicated export-stream dependency.',
            reused_CPU_baseline=cpu,baseline_SHA256=digest(ROOT/'analysis'/f'{cpu}_pilot_summary.json'),
            scope='Repeat N1/N8 and all28 KV gates before any timing. No SLO,model or dataset change.'))
    arms=[(ipc,'cuda_ipc')] if args.resume_ipc_tag else [(cpu,'cpu'),(ipc,'cuda_ipc')]
    for tag,transport in arms:
        names=[];proxy=None
        try:
            names,proxy=launch(tag,transport)
            asyncio.run(correctness(tag,transport));kv_capture(tag,transport)
            if transport=='cuda_ipc':compare_kv(cpu,ipc)
            asyncio.run(pilot(tag,transport))
        except BaseException as error:
            write(ROOT/'state'/f'{tag}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
                error=type(error).__name__+': '+str(error),preserve_failure=True))
            raise
        finally:
            if proxy is not None:stop_proxy(proxy)
            if names:retire(names)
    preserve()


if __name__=='__main__':main()
