"""A bounded counter-only CPC audit, not a replacement for the closed N3."""
import asyncio
import json
import os
from pathlib import Path
import signal
import time

from . import baseline_confirmation as baseline
from . import token_confirmation_control as confirmation
from .client import execute,load_replay_counts
from .common import ROOT,digest,preserve,write,experiment_deadline
from .frontend_supply_control import retire
from .ipc_ec_control import stop_proxy,ownership_empty
from .llm_probe_control import rpc
from .metrics import summarize
from .runtime_endpoint_identity import require_free_ports
from .source_snapshot import snapshot

TAG='batch_COUNTER_C46'
METHODS=['NativeStrong','NativeCommon','Static10','Idle10','AggDP3','E_PD_DP2']
METRICS=['duration_seconds','TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms']


def qualify(stages,valid):
    comparisons={}
    for key in METRICS:
        left,profile,right=(stages[stage][key] for stage in ['clean_before','counter','clean_after'])
        center=(left+right)/2
        assert center>0
        difference=profile/center-1;spread=abs(right-left)/center
        comparisons[key]=dict(profile_relative_difference=difference,clean_relative_spread=spread,
            passed=abs(difference)<=.05 and spread<=.05)
    return dict(comparisons=comparisons,collector_valid=valid,
        observer_gate_passed=valid and all(row['passed'] for row in comparisons.values()))


def wait_previous():
    tag='ec_INLINE_CAUSAL_C44';value=json.loads((ROOT/'state'/f'{tag}_controller.json').read_text())
    proc=Path('/proc',str(value['pid']))
    while proc.exists():
        assert proc.stat().st_uid==os.getuid()
        assert b'mllm_epd_prefill_heavy_token_split.transport_policy_control' in (proc/'cmdline').read_bytes()
        assert time.time()+3600<experiment_deadline();time.sleep(1)
    for prefix in ['interference_PROBE_C40R4','suffix_QUANTA_C41R3',tag]:
        for path in (ROOT/'state').glob(f'server_{prefix}*.json'):
            if path.name.endswith('.source_snapshot.json'):continue
            lease=json.loads(path.read_text())
            assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()
    write(ROOT/'state'/f'{TAG}_resource_predecessor.json',dict(time=time.time(),
        predecessor=tag,completed=(ROOT/'state'/f'{tag}_controller_complete.json').exists(),
        all_owned_leases_closed=True,scope='Resource order only. Fresh correctness is mandatory.'))


async def measurements(tag,method,url,path,main,heldout,counts):
    # The same callable is used by the existing family lifecycle. Only this
    # measurement procedure changes. The model and candidate admission do not.
    rows=main;results=[];stages={};replies={};opened=[]
    metadata=dict(manifest_SHA256=digest(path),method=method,actual_independent_initializations=1,
        evidence='DIAGNOSTIC counter-only CPC; task quality PENDING',
        not_a_new_serving_method=True,not_an_independent_N3_or_primary_replacement=True,
        no_CUDA_events_or_synchronization=True,actual_batch_scope='Prepared logical query geometry, not graph-padded kernel shapes')
    async def run(stage):
        name=tag+'_'+stage
        values=await execute(rows,[url],name,2.,2026112402,counts,metadata=metadata)
        assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' and r['output_tokens']==counts[r['request_id']] for r in values)
        if method!='AggDP3':ownership_empty(name,'cuda_ipc')
        metric=summarize(name)
        results.append(dict(method=method,subset='Main100',mode=stage,run=name,metrics=metric))
        print('BATCH_COUNTER_COHORT',name,{key:metric[key] for key in METRICS},flush=True)
        return metric
    await run('warmup');stages['clean_before']=await run('clean_before')
    try:
        for port in [18840,18841,18842]:
            replies[str(port)]=dict(begin=rpc(port,'batch_counter/begin',dict(name=tag)))
            opened.append(port)
        stages['counter']=await run('counter')
    finally:
        for port in reversed(opened):replies[str(port)]['end']=rpc(port,'batch_counter/end',{})
        write(ROOT/'state'/f'{tag}_counter_RPC.json',replies)
    stages['clean_after']=await run('clean_after')
    valid=all(item['valid'] and item['original_methods_restored'] for reply in replies.values() for item in reply['end'])
    actual_images=actual_visual=0
    for reply in replies.values():
        for worker in reply['end']:
            folder=Path(worker['folder'])
            if not folder.name.startswith(('encoder_gpu','agg_gpu')):continue
            observed=json.loads((folder/'histograms.json').read_text())
            actual_images+=sum(r['output_images']*r['calls'] for r in observed['encoder_output_histogram'])
            actual_visual+=sum(r['output_visual_rows']*r['calls'] for r in observed['encoder_output_histogram'])
    conservation=dict(actual_images=actual_images,expected_images=sum(r['image_count'] for r in rows),
        actual_visual_rows=actual_visual,expected_visual_rows=sum(r['visual_tokens'] for r in rows))
    conservation['passed']=conservation['actual_images']==conservation['expected_images'] and conservation['actual_visual_rows']==conservation['expected_visual_rows']
    valid=valid and conservation['passed']
    result=qualify(stages,valid)
    write(ROOT/'analysis'/f'{tag}_counter_qualification.json',dict(time=time.time(),method=method,stages=stages,**result,
        encoder_output_conservation=conservation,
        independent_initializations=1,task_quality='PENDING',actual_batch_scope=metadata['actual_batch_scope'],
        full_GO=False,clean_N3_batch_backfill=False,
        scope='Frozen one-factor observer study. A failed gate remains diagnostic. No metric is waived or selectively removed.'))
    return results


def run_baseline(method,path,main,heldout,counts):
    family='agg' if method=='AggDP3' else 'e_pd';tag=f'{TAG}_B0_{method}'
    names=[tag+'_'+suffix for suffix in (['A4','A5','A6'] if family=='agg' else ['E4','P5','P6'])];proxy=None
    try:
        actual_names,proxy=baseline.launch(family,tag);assert actual_names==names
        asyncio.run(baseline.correctness(family,tag))
        return asyncio.run(measurements(tag,method,baseline.URL,path,main,heldout,counts))
    finally:
        if proxy is not None:stop_proxy(proxy)
        else:
            # The family launcher can fail after creating its proxy but before
            # returning the Popen handle. Resolve only this exact owned name.
            lease=ROOT/'state'/f'{tag}_proxy.json'
            if lease.exists():
                value=json.loads(lease.read_text());proc=Path('/proc',str(value['pid']))
                if proc.exists():
                    command=(proc/'cmdline').read_bytes()
                    assert proc.stat().st_uid==os.getuid() and os.getsid(value['pid'])==value['pid']
                    assert b'mllm_epd_prefill_heavy_token_split.proxy' in command and (tag+'_proxy').encode() in command
                    os.killpg(value['pid'],signal.SIGTERM)
                    until=time.time()+20
                    while proc.exists():
                        assert time.time()<until;time.sleep(.2)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])


def main():
    preserve();assert time.time()+5400<experiment_deadline()
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    source=snapshot(TAG);path,main_rows,heldout=confirmation.workload()
    counts=load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_COUNTER_GPU_RESULTS',source_archive_SHA256=source['archive_SHA256'],
        manifest_SHA256=digest(path),methods=METHODS,QPS=2.,GPU_group=[4,5,6],independent_initializations_per_method=1,
        purpose='The closed N3 lacks clean actual-batch measurements. Measure representative logical query and encoder output geometry without CUDA events or CUPTI.',
        sequence=['N1/N8 existing family correctness','Replay100 warmup','clean_before','counter','clean_after'],
        sampling='Same per-request R0 output token counts, original prompt/order/arrival. Not Natural EOS or new N3 confirmation.',
        actual_geometry='P/D prepared prompt versus generation query rows, mixed-request categories, encoder output image and visual-row counts.',
        scope_exclusions=['No exclusive kernel/transfer time','No SM/HBM counters','No queue timeline','No future safe window','No quality PASS','No backfill of old clean N3 batch shapes'],
        observer_gate=dict(metrics=METRICS,relative_difference_max=.05,clean_spread_max=.05),
        unchanged='Original model, BF16, compiler, chunked prefill, B8192/S1024, KV42GiB, Block128 and existing conservative Token admission.',
        no_active_decode_insertion=True,no_parameter_search=True,
        escalation='No automatic repeat or independent N3 expansion. All failed gates remain diagnostic.',
        dependency='Start only after owned C44 and all earlier owned GPU leases close. No correctness proof is borrowed.'))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_C44_RESOURCE'))
    wait_previous();results=[];original=confirmation.measurements;old_tag=confirmation.TAG
    confirmation.measurements=measurements;confirmation.TAG=TAG
    try:
        for method in METHODS:
            assert time.time()+600<experiment_deadline()
            require_free_ports([18840,18841,18842,18846,18847,18848,18849])
            if method in ['AggDP3','E_PD_DP2']:results.extend(run_baseline(method,path,main_rows,heldout,counts))
            else:results.extend(confirmation.run_arm(0,method,path,main_rows,heldout,counts))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            completed_cohorts=len(results),error=type(error).__name__+': '+str(error),restart_automatically=False))
        raise
    finally:confirmation.measurements=original;confirmation.TAG=old_tag
    write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE',results=results,
        independent_initializations_per_method=1,task_quality='PENDING',full_GO=False))
    write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),status='COMPLETE'));preserve()


if __name__=='__main__':main()
