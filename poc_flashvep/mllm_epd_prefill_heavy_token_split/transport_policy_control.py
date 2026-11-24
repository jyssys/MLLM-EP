"""One-factor Native inline/on-demand EC control. Never weaken NativeStrong."""
import asyncio
import json
import os
from pathlib import Path
import random
import time

from .client import execute,load_replay_counts,resolve_manifest
from .common import ROOT,digest,preserve,write,experiment_deadline
from .frontend_supply_control import retire
from .ipc_ec_control import launch,ownership_empty,stop_proxy
from .llm_probe_control import rpc
from .metrics import summarize
from .native_common_control import set_common_ipc_policy
from .runtime_endpoint_identity import require_free_ports
from .source_snapshot import snapshot
from .token_confirmation_control import smoke
from .whole_deflection_control import launch_whole,stop_whole

TAG='ec_INLINE_CAUSAL_C44'
DEPENDENCIES=[('whole_DEFLECTION_C43R4','whole_deflection_control'),
    ('interference_PROBE_C40R4','interference_probe_control'),('suffix_QUANTA_C41R3','suffix_quantum_cost_control')]


def policy_order(index):
    return [True,False,False,True] if index%2==0 else [False,True,True,False]


def wait_resources():
    records=[]
    for tag,module in DEPENDENCIES:
        value=json.loads((ROOT/'state'/f'{tag}_controller.json').read_text())
        proc=Path('/proc',str(value['pid']))
        while proc.exists():
            assert proc.stat().st_uid==os.getuid()
            command=(proc/'cmdline').read_bytes()
            assert ('mllm_epd_prefill_heavy_token_split.'+module).encode() in command and tag.encode() in command
            assert time.time()+1800<experiment_deadline();time.sleep(1)
        for path in (ROOT/'state').glob(f'server_{tag}*.json'):
            if path.name.endswith('.source_snapshot.json'):continue
            lease=json.loads(path.read_text())
            assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()
        records.append(dict(tag=tag,completed=(ROOT/'state'/f'{tag}_controller_complete.json').exists(),
            all_owned_leases_closed=True,dependency='Resource order only. No correctness proof borrowed.'))
    write(ROOT/'state'/f'{TAG}_predecessors.json',dict(time=time.time(),records=records))


def policy(tag,enabled):
    # This original guard stays unchanged. No private image or issued descriptor
    # can remain live while the global default changes.
    before=ownership_empty(tag+'_before_policy','cuda_ipc')
    reply=rpc(18840,'ec_ipc/inline_policy',dict(enabled=enabled))
    assert reply[0]['inline_role']==('prefill' if enabled else None)
    return dict(before=before,reply=reply)


def run_start(index,rows,counts):
    tag=f'{TAG}_B{index}';names=[];native=None;proxy=None;results=[]
    order=policy_order(index)
    try:
        names,native=launch(tag,'cuda_ipc',fixed_reduction=True,deterministic_reductions=True,owned_names=names)
        stop_proxy(native);native=None;set_common_ipc_policy()
        proxy=launch_whole(tag+'_policy_proxy',names,'never');url='http://127.0.0.1:18849'
        for enabled in [True,False]:
            label='inline' if enabled else 'on_demand'
            policy(tag+'_'+label,enabled)
            asyncio.run(smoke(tag+'_'+label,'NativeCommon',url))
            run=tag+'_'+label+'_warmup'
            values=asyncio.run(execute(rows,[url],run,2.,2026112402,counts,
                metadata=dict(evidence='DIAGNOSTIC_WARMUP',one_factor='EC descriptor policy only',task_quality='PENDING')))
            assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' for r in values)
            ownership_empty(run,'cuda_ipc')
        for period,enabled in enumerate(order):
            label='inline' if enabled else 'on_demand';change=policy(tag+f'_P{period}',enabled)
            for mode,output in [('natural',None),('replay',counts)]:
                assert time.time()+300<experiment_deadline();run=f'{tag}_P{period}_{label}_{mode}'
                values=asyncio.run(execute(rows,[url],run,2.,2026112402,output,
                    metadata=dict(evidence='MEASURED_ONLINE one-factor Native transport control',task_quality='PENDING',
                        one_factor='Original E inline P descriptor versus on-demand P descriptor. Same full P, original NIXL D and no deflection.',
                        descriptor_inline=enabled,block=index,period=period,
                        model_compiler_sampling_dataset_arrival_admission='Identical in both arms',
                        no_token_split=True,no_whole_deflection=True,GPU_observer='OFF',actual_batch='NOT_MEASURED in clean runs',
                        independent_start_unit='One initialization, with repeated ABBA periods. Not four independent starts.')))
                assert len(values)==100 and all(r['status']=='MEASURED_ONLINE' for r in values)
                ownership_empty(run,'cuda_ipc');metric=summarize(run)
                results.append(dict(block=index,period=period,inline=enabled,mode=mode,run=run,policy_change=change,metrics=metric))
                write(ROOT/'analysis'/f'{tag}_partial.json',dict(time=time.time(),results=results))
                print('TRANSPORT_ONE_FACTOR',run,{k:metric[k] for k in ['TTFT_p95_ms','ITL_p99_ms','SLO_goodput']},flush=True)
        write(ROOT/'analysis'/f'{tag}_summary.json',dict(time=time.time(),status='COMPLETE',results=results,
            independent_initializations=1,periods_are_not_independent_starts=True))
        return results
    except BaseException as error:
        write(ROOT/'state'/f'{tag}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_cohorts=len(results),restart_automatically=False))
        raise
    finally:
        if proxy is not None:stop_whole(proxy)
        if native is not None:stop_proxy(native)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])


def main():
    preserve();assert time.time()+3600<experiment_deadline()
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    source=snapshot(TAG);manifest=resolve_manifest(None)
    rows=json.loads(manifest.read_text())['requests'];random.Random(2026112404).shuffle(rows)
    counts=load_replay_counts(ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl')
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_ONE_FACTOR_NATIVE_TRANSPORT_RESULTS',
        source_archive_SHA256=source['archive_SHA256'],manifest_SHA256=digest(manifest),GPU_group=[4,5,6],QPS=2.,
        independent_initializations=3,periods=[policy_order(index) for index in range(3)],
        factor='Only E inline P descriptor versus on-demand P descriptor. Original tensors, clone ACKs and safe release unchanged.',
        controls='Both arms use the same original full28 Native E/P/D, common compiled BF16, no Token, no deflection and the same never-deflect coordinator.',
        correctness='Fresh serial N1/N8 outputs in both policies before timing. No active state when changing the default.',
        primary_evidence='MEASURED_ONLINE Native root-cause control, not a new candidate serving method.',
        motivation='A closed on-demand replay had148 P import groups/1251 images, RPC mean81.455ms/p95457.555ms/max1055.674ms. '
            'This is descriptive host evidence, not exclusive GPU time or a causal effect estimate.',
        strong_baseline='NativeStrong and the already inline Static Token remain unchanged. This cannot be advertised as a repair of their absent export RPC.',
        hypotheses=['Inline removes export RPC but only improves TTFT if that host dependency is critical.',
            'Clone/ready-stream waiting can dominate even with zero export RPC.',
            'Natural output-work effects can differ from equal-output replay; never pool them.'],
        observer='No GPU profiler or tensor capture during timing. Existing transport logs remain identical instrumentation in both arms.',
        uncertainty='Average the two periods of each policy within a start, then form paired independent N3 CI. Never use24 cohorts as24 starts.',
        dependencies=DEPENDENCIES,task_quality='PENDING',Dataset_SLO_unchanged=True,no_prefill_heavy_or_safe_window_claim=True))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),status='WAITING_FOR_OWNED_RESOURCE_DEPENDENCIES'))
    wait_resources();require_free_ports([18840,18841,18842,18847,18849]);results=[]
    for index in range(3):
        assert time.time()+900<experiment_deadline();results.extend(run_start(index,rows,counts))
    write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE',results=results,
        independent_initializations=3,one_factor='EC descriptor policy',task_quality='PENDING'))
    write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),status='COMPLETE'));preserve()


if __name__=='__main__':main()
