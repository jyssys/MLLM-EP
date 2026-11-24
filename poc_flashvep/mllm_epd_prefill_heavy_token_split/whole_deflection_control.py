"""A separately frozen whole-request control with actual online E and original NIXL."""
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

from .client import execute,payload,one,load_replay_counts,resolve_manifest
from .common import ROOT,REPO,PYTHON,digest,preserve,write,experiment_deadline
from .dataset_build import keys
from .frontend_supply_control import retire
from .ipc_ec_control import launch,ownership_empty,stop_proxy as stop_native_proxy
from .llm_probe_control import rpc
from .metrics import summarize
from .native_common_control import set_common_ipc_policy
from .natural_token_correctness_control import INDICES
from .numerical import embedding_metrics
from .source_snapshot import snapshot
from .runtime_endpoint_identity import require_identity,require_free_ports
from .token_confirmation_control import smoke
from .vision_batch_handoff import Session

TAG='whole_DEFLECTION_C43'
URL='http://127.0.0.1:18849'
METHODS=['NativeStrong','NativeOnDemand','WholeDeflection']


def fresh_holdout():
    parent=json.loads((ROOT/'data/main_manifest_template_v2.json').read_text())
    main=json.loads(resolve_manifest(None).read_text())['requests']
    calibration=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
    excluded=set().union(*(keys(r) for r in main+calibration))
    candidate=[r for r in parent['requests'] if r['request_id'] in set(parent['heldout_ids']) and not keys(r)&excluded]
    random.Random(2026112517).shuffle(candidate)
    # Do not filter outputs, model quality, observed latency or winning method.
    selected=candidate[:50]
    assert len({r['request_id'] for r in selected})==len(selected)
    assert all(not keys(r)&excluded for r in selected)
    return selected,dict(available_context_disjoint_requests=len(candidate),selected=len(selected),
        sampling_seed=2026112517,selection='First50 shuffled parent-heldout requests with no Main100/calibration image or document context.',
        source_counts={s:sum(r['source']==s for r in selected) for s in sorted({r['source'] for r in selected})},
        prior_exposure='Original single-GPU Natural length profiling exists. These are new to this campaign C3 method comparison.',
        not_a_new_Main100=True,no_output_or_performance_filter=True)


def orders():
    row=METHODS.copy();random.Random(2026112512).shuffle(row)
    return [row[i:]+row[:i] for i in range(3)]


def wait_previous(tag='suffix_QUANTA_C41'):
    value=json.loads((ROOT/'state'/f'{tag}_controller.json').read_text())
    proc=Path('/proc',str(value['pid']))
    while proc.exists():
        assert proc.stat().st_uid==os.getuid()
        assert b'mllm_epd_prefill_heavy_token_split.suffix_quantum_cost_control' in (proc/'cmdline').read_bytes()
        assert time.time()<experiment_deadline();time.sleep(1)
    assert (ROOT/'state'/f'{tag}_controller_complete.json').exists(),'Inspect failed predecessor before starting another GPU group'
    for path in (ROOT/'state').glob(f'server_{tag}*.json'):
        if path.name.endswith('.source_snapshot.json'):continue
        row=json.loads(path.read_text());assert row['finished'] and not Path('/proc',str(row['guardian'])).exists()


def launch_whole(tag,names,policy,correctness=False):
    command=[str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.whole_deflection_proxy',
        '--name',tag,'--port','18849','--policy',policy]
    if correctness:command+=['--correctness-only','--capture','--defer-release']
    lease=json.loads((ROOT/'state'/f'server_{names[0]}.json').read_text())
    with (ROOT/'logs'/f'{tag}.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=lease['executed_source_cwd'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state'/f'{tag}.json',dict(time=time.time(),pid=process.pid,sid=process.pid,command=command))
    until=time.time()+30
    while True:
        assert process.poll() is None
        try:
            with urllib.request.urlopen(URL+'/health',timeout=2) as response:
                if response.status==200:
                    require_identity(response,tag,process)
                    return process
        except OSError:pass
        assert time.time()<until,'Inspect the same proxy. Do not restart because an observation expired.'
        time.sleep(.2)


def stop_whole(process):
    if process.poll() is not None:return
    proc=Path('/proc',str(process.pid))
    assert proc.stat().st_uid==os.getuid() and os.getsid(process.pid)==process.pid
    assert b'mllm_epd_prefill_heavy_token_split.whole_deflection_proxy' in (proc/'cmdline').read_bytes()
    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=20)


async def teacher(row,cohort,base,run):
    import aiohttp
    # Reuse the actual E output, not a new encoder execution or a corrected tensor.
    request_headers={'x-campaign-metadata-key':hashlib.sha256(base.encode()).hexdigest(),'x-campaign-metadata-mode':'grid_only'}
    rpc(18841,'llm_probe/begin',dict(name=run+'_P',max_logits=1,include_decode_inputs=False))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),trust_env=False) as session:
            result=await one(Session(session,request_headers),'http://127.0.0.1:18841',row,
                payload(row,cohort),time.monotonic(),run,0)
        assert result['status']=='MEASURED_ONLINE'
        return result
    finally:rpc(18841,'llm_probe/end',{})


def compare(row,base,reference_run,actual_result,teacher_result):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2);records=[];values=[]
    for name in [base+'_D',reference_run+'_P']:
        path=ROOT/'raw/llm_reference'/name/'manifest.json';meta=json.loads(path.read_text())
        assert digest(meta['tensor_file'])==meta['tensor_SHA256']
        records.append(dict(path=str(path),SHA256=digest(path),mrope_delta=meta['snapshot']['mrope_delta']))
        values.append(load_file(meta['tensor_file']))
    actual,target=values
    fields={key:torch.equal(actual[key],target[key]) for key in ['prompt_token_ids','full_prompt_mrope']}
    fields['mrope_delta']=records[0]['mrope_delta']==records[1]['mrope_delta']
    layers=[dict(layer=i,**embedding_metrics(actual[f'prompt_KV_{i}'],target[f'prompt_KV_{i}'])) for i in range(28)]
    inputs=[]
    for key in sorted(k for k in actual if k.startswith(('inputs_embeds_','positions_'))):
        assert key in target
        inputs.append(dict(key=key,strict=torch.equal(actual[key],target[key])))
    tokens=actual_result['token_ids']==teacher_result['token_ids']
    passed=all(fields.values()) and all(r['strict'] for r in layers) and all(r['strict'] for r in inputs) and tokens
    proof=dict(time=time.time(),request_id=row['request_id'],base=base,fields=fields,layers=layers,prepared_inputs=inputs,
        strict_greedy_tokens=tokens,output_tokens=actual_result['output_tokens'],passed=passed,manifests=records,
        evidence='CORRECTNESS_ONLY',actual_D_batch='One full request, serialized by the correctness-only full-prefill admission',
        teacher_P_batch=1,scope='Same actual E embedding and original full-model query geometry. No second encode, output correction or KV repair. '
            'This conditional state proof is not a cross-encoder-geometry task-quality guarantee.')
    write(ROOT/'analysis'/f'{base}_same_E_comparison.json',proof)
    assert passed,'Same-E full D vs full P state/output gate failed. Do not time the whole-request candidate.'
    return proof


async def correctness_cohort(tag,rows):
    values=await execute(rows,[URL],tag,2.,2026112402,None,serial=len(rows)==1,
        metadata=dict(evidence='CORRECTNESS_ONLY',scope='Online concurrent E, one full D request at a time. '
            'CPU full-state captures invalidate timing. Keep E exports for the same-E full P reference.'))
    assert len(values)==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in values)
    results=[]
    for i,(row,actual) in enumerate(zip(rows,values)):
        base=f'{tag}_{i:04d}';reference_run=base+'_SHARED_E_TEACHER'
        reference=await teacher(row,tag,base,reference_run)
        results.append(compare(row,base,reference_run,actual,reference))
        released=rpc(18840,'ec_ipc/release',dict(key=hashlib.sha256(base.encode()).hexdigest()))
        assert released[0]['outstanding_exports']==0
    ownership_empty(tag,'cuda_ipc')
    return results


def correctness():
    tag=TAG+'_CORRECT';names=[];native=None;proxy=None;results=[]
    try:
        names,native=launch(tag,'cuda_ipc',fixed_reduction=True,deterministic_reductions=True,owned_names=names)
        stop_native_proxy(native);native=None;set_common_ipc_policy()
        reply=rpc(18840,'ec_ipc/inline_policy',dict(enabled=False));assert reply[0]['inline_role'] is None
        proxy=launch_whole(tag+'_proxy',names,'always_correctness',True)
        pool=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
        results.extend(asyncio.run(correctness_cohort(tag+'_N1',[pool[0]])))
        rows=[pool[i] for i in INDICES];random.Random(2026112404).shuffle(rows)
        results.extend(asyncio.run(correctness_cohort(tag+'_N8',rows)))
        write(ROOT/'analysis'/f'{tag}_summary.json',dict(time=time.time(),status='CORRECTNESS_PASS',cases=results,
            full_task_quality='PENDING',C3_performance='NOT_RUN in these captured cohorts'))
    except BaseException as error:
        write(ROOT/'state'/f'{tag}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_state_cases=len(results)))
        raise
    finally:
        if proxy is not None:stop_whole(proxy)
        if native is not None:stop_native_proxy(native)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])


async def measure(tag,method,url,main,fresh,counts):
    results=[]
    schedule=[('Main100','warmup',main,counts,2026112402),('Main100','natural',main,None,2026112402),
        ('Main100','replay',main,counts,2026112402)]
    if fresh:schedule += [('FreshHoldout','natural',fresh,None,2026112515),('FreshHoldout','replay',fresh,counts,2026112515)]
    for subset,mode,rows,output,seed in schedule:
        assert time.time()<experiment_deadline();run=f'{tag}_{subset}_{mode}'
        values=await execute(rows,[url],run,2.,seed,output,metadata=dict(method=method,subset=subset,
            evidence='DIAGNOSTIC_WARMUP' if mode=='warmup' else 'MEASURED_ONLINE runtime',
            quality='PENDING',no_token_or_layer_split=True,observer='OFF',
            policy='Deflect after actual encode only with P HTTP work and zero D streams. No future-output input.',
            baseline='Keep NativeStrong unchanged. NativeOnDemand shares the candidate EC transport and coordinator, without deflection.'))
        assert len(values)==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in values)
        ownership_empty(run,'cuda_ipc');metric=summarize(run)
        if mode!='warmup':results.append(dict(method=method,subset=subset,mode=mode,run=run,metrics=metric))
        print('WHOLE_DEFLECTION_METRICS',run,{k:metric[k] for k in ['TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput']},flush=True)
    return results


def run_arm(block,method,main,fresh,counts):
    tag=f'{TAG}_B{block}_{method}';names=[];native=None;proxy=None
    try:
        common=method!='NativeStrong'
        names,native=launch(tag,'cuda_ipc',fixed_reduction=common,deterministic_reductions=common,owned_names=names)
        set_common_ipc_policy()
        if common:
            stop_native_proxy(native);native=None
            assert rpc(18840,'ec_ipc/inline_policy',dict(enabled=False))[0]['inline_role'] is None
            proxy=launch_whole(tag+'_proxy',names,'never' if method=='NativeOnDemand' else 'idle_after_encode')
            url=URL
        else:url='http://127.0.0.1:18847'
        asyncio.run(smoke(tag,'NativeCommon' if common else 'NativeStrong',url))
        results=asyncio.run(measure(tag,method,url,main,fresh,counts))
        write(ROOT/'analysis'/f'{tag}_summary.json',dict(time=time.time(),status='COMPLETE',block=block,method=method,results=results,quality='PENDING'))
        return results
    except BaseException as error:
        write(ROOT/'state'/f'{tag}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',error=type(error).__name__+': '+str(error)))
        raise
    finally:
        if proxy is not None:stop_whole(proxy)
        if native is not None:stop_native_proxy(native)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])


def main():
    global TAG
    p=argparse.ArgumentParser();p.add_argument('--wait-for-c41',action='store_true')
    p.add_argument('--tag',default=TAG);p.add_argument('--predecessor-tag',default='suffix_QUANTA_C41')
    args=p.parse_args()
    assert re.fullmatch(r'whole_DEFLECTION_C43(?:R[0-9]+)?',args.tag)
    assert re.fullmatch(r'suffix_QUANTA_C41(?:R[0-9]+)?',args.predecessor_tag)
    TAG=args.tag
    preserve();assert time.time()+3600<experiment_deadline()
    path=resolve_manifest(None);main=json.loads(path.read_text())['requests'];random.Random(2026112404).shuffle(main)
    fresh,audit=fresh_holdout();random.Random(2026112516).shuffle(fresh)
    source=snapshot(TAG);plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    manifest=ROOT/'data/whole_deflection_fresh_holdout_v0.json'
    frozen_manifest=dict(name='Supplemental fresh-context serving holdout',requests=fresh,audit=audit,
        parent_manifest_SHA256=digest(ROOT/'data/main_manifest_template_v2.json'),main100_unchanged=True)
    if manifest.exists():assert json.loads(manifest.read_text())==frozen_manifest,'Do not replace the frozen supplemental holdout'
    else:write(manifest,frozen_manifest)
    reference=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl';counts=load_replay_counts(reference)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_WHOLE_BASELINE_GPU_RESULTS',source_archive_SHA256=source['archive_SHA256'],
        Main100_SHA256=digest(path),fresh_manifest_SHA256=digest(manifest),fresh_availability=audit,methods=METHODS,
        method_order_by_block=orders(),independent_initializations=3,GPU_group=[4,5,6],QPS=2.,
        algorithm='No tuned fraction. After original E completes, use D only if P HTTP work exists and D has no active or pending admission.',
        exact_model='Original BF16,full28 forward and decode,original vision/processor/MRoPE. No prefix KV for a deflected request.',
        safety='Wait for zero prior D streams before whole D prefill. Native admission resumes at first generated token, not role SSE.',
        transport='On-demand EC for candidate and NativeOnDemand. Keep original inline NativeStrong as the best-of-family control.',
        correctness='N1 and concurrent E N8 use the SAME actual E outputs for full P teacher vs full D. All28 strict before timing.',
        no_quality_gate_relaxation=True,no_encoder_rewrite=True,no_Kairos_reproduction_claim=True,
        heldout_caveat='Existing Main100 and its old heldout have prior campaign exposure. Supplemental fresh contexts are explicitly audited.',
        SLO='Unchanged5s/34ms/33ms. Relative ITLp99/TPOTp95 and TTFTp50 CI upper1.05. Serving gain>=5percent and quality required.',
        quality='PENDING; no unauthorized external extraction.',wait_for_c41=args.wait_for_c41,predecessor_tag=args.predecessor_tag))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_C41' if args.wait_for_c41 else 'STARTING'))
    if args.wait_for_c41:wait_previous(args.predecessor_tag)
    require_free_ports([18840,18841,18842,18847,18849])
    correctness();results=[]
    for block,methods in enumerate(orders()):
        for method in methods:
            assert time.time()+900<experiment_deadline()
            results.extend(run_arm(block,method,main,fresh,counts))
            write(ROOT/'analysis'/f'{TAG}_partial.json',dict(time=time.time(),results=results))
    write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE',results=results,
        quality='PENDING',evidence='MEASURED_ONLINE runtime',independent_starts_per_method=3))
    write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),status='COMPLETE'))
    preserve()


if __name__=='__main__':main()
