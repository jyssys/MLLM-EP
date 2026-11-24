"""Frozen same-group N3 confirmation. Keep runtime evidence and task quality separate."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import time

from . import token_correctness_control as single
from .client import execute, load_replay_counts, resolve_manifest
from .common import ROOT, digest, experiment_deadline, preserve, write
from .frontend_supply_control import retire
from .ipc_ec_control import launch, ownership_empty, stop_proxy
from .metrics import summarize
from .native_common_control import set_common_ipc_policy
from .natural_token_correctness_control import INDICES
from .source_snapshot import snapshot
from .token_discovery_control import proxy_state
from .token_split import select_prefix

TAG='token_CONFIRM_C37'
METHODS=['NativeStrong','NativeCommon','Static10','Idle10']


def method_orders():
    rng=random.Random(2026112506)
    orders=[]
    for _ in range(3):
        order=METHODS.copy();rng.shuffle(order);orders.append(order)
    return orders


def workload():
    path=resolve_manifest(None);manifest=json.loads(path.read_text())
    rows=list(manifest['requests']);random.Random(2026112404).shuffle(rows)
    heldout=[row for row in manifest['requests'] if row['request_id'] in set(manifest['heldout_ids'])]
    random.Random(2026112508).shuffle(heldout)
    assert len(rows)==100 and len(heldout)==50
    assert set(manifest['discovery_ids']).isdisjoint(manifest['heldout_ids'])
    return path,rows,heldout


def wait_closed(tag):
    for role,gpu in [('E',4),('P',5)]:
        path=ROOT/'state'/f'server_{tag}_{role}{gpu}.json'
        lease=json.loads(path.read_text());proc=Path('/proc',str(lease['guardian']))
        while proc.exists():
            assert proc.stat().st_uid==os.getuid()
            assert tag.encode() in (proc/'cmdline').read_bytes()
            assert time.time()<experiment_deadline()
            time.sleep(1)
        assert json.loads(path.read_text())['finished']


def prior_reference(method,index):
    if method=='NativeStrong':
        path=ROOT/'raw/frontend_B5_cache0_T2_N8_natural_v0_requests.jsonl'
        pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
        return next(r for r in map(json.loads,path.read_text().splitlines()) if r['request_id']==pool[index]['request_id'])
    run=(f'ec_COMMON_C32_N8_{index}_NATIVE' if method=='NativeCommon' else
        f'token_FROZEN_C36_F10_N8_{index}_TEACHER')
    return json.loads((ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines()[0])


async def smoke(tag,method,url):
    pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    proof=[]
    for label,indices in [('N1',[0]),('N8',INDICES)]:
        rows=[pool[i] for i in indices]
        run=f'{tag}_SMOKE_{label}'
        values=await execute(rows,[url],run,.25,2026112402,None,serial=True,
            metadata=dict(evidence='CORRECTNESS_ONLY',scope='Unprofiled serial output regression before timing. '
                'This is not a new concurrent KV or full task-quality gate.'))
        for index,actual in zip(indices,values):
            reference=prior_reference(method,index)
            proof.append(dict(phase=label,index=index,request_id=actual['request_id'],
                status=actual['status'],strict_tokens=actual['token_ids']==reference['token_ids'],
                strict_prompt=actual.get('prompt_token_ids')==reference.get('prompt_token_ids')))
        ownership_empty(run,'cuda_ipc')
    value=dict(time=time.time(),method=method,checks=proof,
        passed=len(proof)==9 and all(r['status']=='MEASURED_ONLINE' and r['strict_tokens'] and r['strict_prompt'] for r in proof),
        concurrent_KV_gate='Reused fraction10 C36 N1/N8, not this serial smoke',task_quality='PENDING')
    write(ROOT/'analysis'/f'{tag}_smoke.json',value)
    assert value['passed'],'Preserve this initialization and inspect its output regression before timing.'


async def measurements(tag,method,url,path,main,heldout,counts):
    cells=[]
    schedule=[('warmup','Main100',main,counts,2026112402),
        ('natural','Main100',main,None,2026112402),('replay','Main100',main,counts,2026112402),
        ('natural','Heldout50',heldout,None,2026112507),('replay','Heldout50',heldout,counts,2026112507)]
    for mode,subset,rows,output,seed in schedule:
        assert time.time()<experiment_deadline()
        run=f'{tag}_{subset}_{mode}'
        values=await execute(rows,[url],run,2.,seed,output,metadata=dict(
            method=method,subset=subset,manifest_SHA256=digest(path),independent_start=tag,
            evidence='DIAGNOSTIC_WARMUP' if mode=='warmup' else 'MEASURED_ONLINE runtime confirmation',
            task_quality='PENDING',GPU_observer='OFF',CPU_tensor_capture='OFF',
            safety='Extra suffix begins only when all prior D HTTP streams finish. '
                'A new Native D request waits only until the suffix emits its first actual token.',
            selection='Fixed10% fraction from Discovery. No parameter changes after heldout measurements.',
            scope='Natural EOS and exact per-request output-count replay are separate. '
                'Main100 contains Discovery50 and Heldout50. Heldout50 is a separate arrival trace, not50 new independent starts.'))
        assert len(values)==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in values)
        ownership_empty(run,'cuda_ipc');state=proxy_state(url)
        metrics=summarize(run)
        write(ROOT/'state'/f'{run}_proxy_drained.json',state)
        if mode!='warmup':
            cell=dict(method=method,subset=subset,mode=mode,run=run,metrics=metrics)
            cells.append(cell)
            write(ROOT/'analysis'/f'{tag}_partial.json',dict(time=time.time(),cells=cells))
        print('TOKEN_CONFIRM_METRICS',run,{key:metrics[key] for key in
            ['TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput','gate_attainment']},flush=True)
    return cells


def run_arm(block,method,path,main,heldout,counts):
    tag=f'{TAG}_B{block}_{method}'
    names=[];proxy=None;teacher=None
    try:
        if method in ('NativeStrong','NativeCommon'):
            common=method=='NativeCommon'
            names,proxy=launch(tag,'cuda_ipc',fixed_reduction=common,
                deterministic_reductions=common,owned_names=names)
            set_common_ipc_policy();url='http://127.0.0.1:18847'
        else:
            single.TAG=tag;single.EC=Path('/dev/shm')/f'mllm_epd_20261124_{tag}'
            single.launch_models(names,fixed_reduction=True,deterministic_reductions=True)
            teacher,teacher_url=single.launch_proxy(tag+'_teacher_proxy','e_pd',18848,names)
            row=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests'][0]
            prefix=select_prefix(row['total_input_tokens'],.1)
            teacher_run=tag+'_KV_TEACHER';token_run=tag+'_KV_TOKEN'
            reference=single.capture(row,teacher_run,teacher_url,[(18841,'P')],prefix)
            proxy,url=single.launch_proxy(tag+'_proxy','epd',18849,names,suffix_fraction=.1,
                token_admission='decode_idle',token_policy='idle_fallback' if method=='Idle10' else 'static')
            # The idle rule correctly selects Native without P pressure. Check the
            # exact static10 path in its own short-lived proxy before testing it.
            check_proxy=None
            try:
                check_url=url
                if method=='Idle10':
                    check_proxy,check_url=single.launch_proxy(tag+'_KV_static_proxy','epd',18846,names,
                        suffix_fraction=.1,token_admission='decode_idle')
                actual=single.capture(row,token_run,check_url,[(18841,'P'),(18842,'D')])
                single.compare(row,prefix,teacher_run,token_run,reference,actual)
            finally:
                if check_proxy is not None:stop_proxy(check_proxy)
            stop_proxy(teacher);teacher=None
        asyncio.run(smoke(tag,method,url))
        cells=asyncio.run(measurements(tag,method,url,path,main,heldout,counts))
        write(ROOT/'analysis'/f'{tag}_complete.json',dict(time=time.time(),status='COMPLETE',block=block,
            method=method,cells=cells,evidence='MEASURED_ONLINE runtime confirmation',task_quality='PENDING'))
        return cells
    except BaseException as error:
        write(ROOT/'state'/f'{tag}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),restart_automatically=False))
        raise
    finally:
        if teacher is not None:stop_proxy(teacher)
        if proxy is not None:stop_proxy(proxy)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-tag',default='vision_BATCH_C38R1')
    args=parser.parse_args();assert args.wait_for_tag.isidentifier()
    preserve();assert time.time()<experiment_deadline()
    path,rows,heldout=workload()
    prior=ROOT/'analysis/token_FROZEN_C36_F10_N8_summary.json'
    assert json.loads(prior.read_text())['passed']
    assert json.loads((ROOT/'analysis/token_FROZEN_C36_F10_N1_summary.json').read_text())['passed']
    discovery=ROOT/'configs/token_DISCOVERY_C34_frozen_fraction_v0.json'
    assert json.loads(discovery.read_text())['fraction']==.1
    reference=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    counts=load_replay_counts(reference)
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    source=snapshot(TAG)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_HELDOUT_CONFIRMATION',
        source_archive_SHA256=source['archive_SHA256'],manifest_SHA256=digest(path),GPU_group=[4,5,6],
        methods=METHODS,independent_initializations_per_method=3,method_order_by_block=method_orders(),
        common_arrival='Main100 seed2026112402/order2026112404; Heldout50 seed2026112507/order2026112508. QPS2.',
        main_ids=[r['request_id'] for r in rows],heldout_ids=[r['request_id'] for r in heldout],
        generation='Original BF16 BI0,Greedy,NaturalEOS cap2048. Separate reference-count replay.',
        replay_reference_SHA256=digest(reference),fraction=.1,discovery_SHA256=digest(discovery),
        prior_concurrent_state_gate_SHA256=digest(prior),
        excluded_25_percent='C36 N8 KV tolerance failure. Preserve all evidence. Do not promote its discovery goodput.',
        quality='PENDING; task-score floors and unauthorized external answer extraction remain unresolved.',
        numerical_policy='NativeStrong unchanged. NativeCommon and candidates share deterministic compiled reductions. '
            'Native baselines retain original NIXL, no token adapter and no artificial D cap.',
        admission='Candidates are conservative zero-prior-D-stream admission, not an active-decode safe-window predictor.',
        gates=dict(relative_ITLp99_CI_upper=1.05,relative_TPOTp95_CI_upper=1.05,
            TTFTp50_CI_upper=1.05,absolute_all_ITL_ms=34,absolute_TPOT_ms=33,
            absolute_decode_attainment_min=.95,TTFT_ms=5000,minimum_serving_gain=.05),
        confidence='Paired independent-start bootstrap within each block. Requests are not independent starts.',
        warmup='Same Main100, replay counts, excluded from metrics. Parameters freeze before any heldout result.',
        stop='Stop and preserve the initialization on a state/output/execution failure. Do not restart with a new label.',
        wait_for_existing_owned_job=args.wait_for_tag))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_OWNED_JOB'))
    wait_closed(args.wait_for_tag)
    all_cells=[]
    for block,methods in enumerate(method_orders()):
        for method in methods:
            assert time.time()<experiment_deadline()
            all_cells.extend(run_arm(block,method,path,rows,heldout,counts))
            write(ROOT/'analysis'/f'{TAG}_partial_results.json',dict(time=time.time(),cells=all_cells))
    write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE',cells=all_cells,
        independent_initializations_per_method=3,task_quality='PENDING',decode_gate='RECALCULATION_REQUIRED',
        evidence='MEASURED_ONLINE runtime confirmation',final_GO='NOT_EVALUATED'))
    write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),pid=os.getpid(),status='COMPLETE'))
    preserve()


if __name__=='__main__':main()
