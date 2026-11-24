"""Official renderer cache/thread interventions with fresh EC lifetimes."""
import asyncio
import json
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace
import urllib.request

from .client import execute,load_replay_counts,resolve_manifest
from .common import ROOT,REPO,PYTHON,digest,preserve,write
from .metrics import summarize
from .profile_control import run as profile_run
from .source_snapshot import snapshot

URL='http://127.0.0.1:18845'
ROLES=[('encoder',4,18840,19840),('prefill',5,18841,19841),('decode',6,18842,19842)]


def wait_ready(names):
    until=time.time()+1500
    while True:
        ready=True
        for name in names:
            path=ROOT/'state'/f'server_{name}.json'
            if not path.exists():ready=False;continue
            value=json.loads(path.read_text())
            assert not value.get('finished'),name+' is terminal. Do not restart under the same identity.'
            proc=Path('/proc',str(value['pid']))
            assert proc.exists() and proc.stat().st_uid==os.getuid()
            if not value['ready']:ready=False;continue
            with urllib.request.urlopen(f'http://127.0.0.1:{value["port"]}/health',timeout=3) as response:
                ready=ready and response.status==200
        if ready:return
        if time.time()>until:raise TimeoutError('Inspect the same live initialization. Observation expiry is not terminal.')
        time.sleep(2)


def retire(names):
    for name in names:
        value=json.loads((ROOT/'state'/f'server_{name}.json').read_text())
        proc=Path('/proc',str(value['guardian']))
        assert value['name']==name and value['gpu'] in [4,5,6]
        if proc.exists():
            assert proc.stat().st_uid==os.getuid()
            assert b'mllm_epd_prefill_heavy_token_split.servers' in (proc/'cmdline').read_bytes()
        write(ROOT/'state'/f'stop_server_{name}',dict(time=time.time(),reason='CONTROLLED_FRONTEND_CONFIGURATION_CHANGE'))
    until=time.time()+180
    while any(Path('/proc',str(json.loads((ROOT/'state'/f'server_{name}.json').read_text())['guardian'])).exists() for name in names):
        if time.time()>until:raise TimeoutError('Owned guards remain live. Do not launch another GPU lease.')
        time.sleep(1)


def launch(tag,cache,workers):
    ec_path=Path('/dev/shm')/f'mllm_epd_20261124_{tag}'
    assert not ec_path.exists(),'A cache0 encoder requires a fresh EC namespace'
    names=[]
    for role,gpu,port,side in ROLES:
        name=f'{tag}_{role[0].upper()}{gpu}';names.append(name)
        cmd=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.servers','start',
             '--name',name,'--gpu',str(gpu),'--role',role,'--port',str(port),
             '--side-port',str(side),'--ec-path',str(ec_path),'--observe']
        if role=='encoder':
            cmd+=['--encoder-native-ops','--compiled-encoder','--encoder-sync-scheduling',
                  '--mm-cache-gb',str(cache),'--renderer-workers',str(workers)]
        subprocess.run(cmd,cwd=REPO,check=True)
    wait_ready(names)
    return names


async def correctness(tag,names):
    path=ROOT/'data/calibration_manifest_template_v2.json'
    all_rows=json.loads(path.read_text())['requests']
    rows=[all_rows[i] for i in [0,1,4,6,12,17,20,31]]
    teacher_path=ROOT/'raw/processor_handoff_B2_N8_natural_v0_grid_only_requests.jsonl'
    teacher={r['request_id']:r for r in map(json.loads,teacher_path.read_text().splitlines())}
    values=await execute(rows,[URL],f'{tag}_N8_natural_v0',.25,2026112402,None,True,
        metadata=dict(evidence='CORRECTNESS_ONLY',workers=names,manifest_SHA256=digest(path)))
    comparison=[dict(request_id=r['request_id'],status=r['status'],
        strict_tokens=r['status']=='MEASURED_ONLINE' and r['token_ids']==teacher[r['request_id']]['token_ids'],
        strict_prompt=r.get('prompt_token_ids')==teacher[r['request_id']].get('prompt_token_ids')) for r in values]
    proof=dict(time=time.time(),evidence='CORRECTNESS_ONLY',comparison=comparison,
        passed=len(values)==8 and all(r['strict_tokens'] and r['strict_prompt'] for r in comparison),
        teacher_raw_SHA256=digest(teacher_path),
        scope='Serial same-group Native EPD geometry. Full task-quality and cross-batch tolerance gates remain independent.')
    write(ROOT/'analysis'/f'{tag}_N8_correctness.json',proof)
    assert proof['passed'],'Frontend identity/output mismatch. Stop before performance.'
    print('FRONTEND_N8_PASS',tag,flush=True)


async def measure(tag,names,qps=3):
    path=resolve_manifest(None);rows=json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    counts=load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    run=f'{tag}_replay100_q{qps}_v0'
    values=await execute(rows,[URL],run,qps,2026112402,counts,
        metadata=dict(evidence='DIAGNOSTIC_C3',workers=names,manifest_SHA256=digest(path),
            processor_handoff='grid_only',client_raw_policy='immediate',quality_gate='PENDING'))
    assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' for r in values)
    value=summarize(run)
    print('FRONTEND_SCREEN',run,{key:value[key] for key in ['TTFT_p95_ms','ITL_p99_ms','gate_attainment']},flush=True)
    return value


def main():
    preserve()
    base=ROOT/'configs/encoder_frontend_concurrency_discovery_plan_v0.json'
    assert json.loads(base.read_text())['status']=='FROZEN_BEFORE_TUNING_RESULTS'
    archive=snapshot('frontend_supply_discovery_v0')
    plan=ROOT/'configs/frontend_supply_execution_v0.json'
    assert not plan.exists()
    names=['metadata_B3_E4','metadata_B3_P5','metadata_B3_D6'];wait_ready(names)
    write(plan,dict(time=time.time(),base_plan_SHA256=digest(base),source_archive_SHA256=archive['archive_SHA256'],
        phases=[dict(tag='frontend_A_cache1_T1',cache=1,threads=1),
                dict(tag='frontend_B4_cache0_T1',cache=0,threads=1),
                dict(tag='frontend_B5_cache0_T2',cache=0,threads=2)],
        QPS=3,client_full_raw='immediate for every arm',output='Same per-request equal-output replay',
        primary_status='DIAGNOSTIC until observer and full model-quality gates pass',
        invariant='All images and original E processing remain. P/D cache1 consume the exact E token/hash/grid result. '
            'No cache0 identifiers or exports survive an encoder restart. No official thread/cache assertion is removed.'))
    results={}
    results['A']=asyncio.run(measure('frontend_A_cache1_T1',names))
    for tag,workers in [('frontend_B4_cache0_T1',1),('frontend_B5_cache0_T2',2)]:
        retire(names);names=launch(tag,0,workers)
        asyncio.run(correctness(tag,names))
        results[tag]=asyncio.run(measure(tag,names))
        information={}
        for role,gpu,port,side in ROLES:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/campaign/worker_info',timeout=60) as response:
                information[role]=json.load(response)
        write(ROOT/'analysis'/f'{tag}_worker_memory_and_scheduler.json',dict(time=time.time(),workers=information,
            evidence='WORKER_COUNTERS',scope='Quiescent post-screen allocator/device counters. Not a co-residency probe.'))
        if workers==2:
            args=SimpleNamespace(manifest=resolve_manifest(None),
                replay_from=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl',
                server_names=','.join(names),order_seed=2026112404,qps=3,seed=2026112402,
                name=tag+'_CPC_q3_v0',url=URL,worker_urls=','.join(f'http://127.0.0.1:{r[2]}' for r in ROLES),
                observer_kind='profile',observe_client_host=False,freeze_client_static_heap=False)
            asyncio.run(profile_run(args))
    write(ROOT/'analysis/frontend_supply_discovery_v0.json',dict(time=time.time(),results=results,
        evidence='DIAGNOSTIC_C3',independent_starts_per_cell=1,model_quality='PENDING',
        scope='A->B changes processor cache policy; B->C changes only renderer worker count. '
            'Different initializations preserve EC lifetimes. No primary multi-start serving claim.'))
    preserve()


if __name__=='__main__':main()
