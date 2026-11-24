"""Secondary source-stratum Native study. Never replace or tune the frozen Main100."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from .client import execute,load_replay_counts
from .common import ROOT,digest,experiment_deadline,preserve,write
from .frontend_supply_control import retire
from .ipc_ec_control import launch,ownership_empty,stop_proxy
from .metrics import summarize
from .native_common_control import set_common_ipc_policy
from .profile_control import run as profile_run
from .source_snapshot import snapshot

TAG='ec_DOCUMENT_C39R1'


def source_rows():
    path=ROOT/'data/main_manifest_template_v2.json'
    parent=json.loads(path.read_text());discovery=set(parent['discovery_ids'])
    rows=[r for r in parent['requests'] if r['request_id'] in discovery and 'MMLongBench' in r['source']]
    assert len(rows)==37
    assert {r['request_id'] for r in rows}.isdisjoint(parent['heldout_ids'])
    return path,rows


def wait_for_confirmation():
    path=ROOT/'state/token_CONFIRM_C37_controller.json'
    value=json.loads(path.read_text());proc=Path('/proc',str(value['pid']))
    while proc.exists():
        assert proc.stat().st_uid==os.getuid()
        assert b'mllm_epd_prefill_heavy_token_split.token_confirmation_control' in (proc/'cmdline').read_bytes()
        assert time.time()<experiment_deadline()
        time.sleep(1)
    # A terminal confirmation failure does not invalidate this independent Native
    # source-stratum question. Its output is never called candidate confirmation.
    for path in (ROOT/'state').glob('server_token_CONFIRM_C37_*.json'):
        if path.name.endswith('.source_snapshot.json'):continue
        lease=json.loads(path.read_text())
        assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists(),path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-c37',action='store_true')
    args=parser.parse_args();preserve();assert time.time()<experiment_deadline()
    source,rows=source_rows()
    reference=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    counts=load_replay_counts(reference)
    manifest=ROOT/'data/secondary_document37_discovery_v0.json'
    plan=ROOT/'configs'/f'{TAG}.json';assert manifest.exists() and not plan.exists()
    archive=snapshot(TAG)
    frozen=json.loads(manifest.read_text())
    assert frozen['requests']==rows and frozen['source_manifest_SHA256']==digest(source)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_SECONDARY_STRATUM_PERFORMANCE',
        source_archive_SHA256=archive['archive_SHA256'],manifest_SHA256=digest(manifest),GPU_group=[4,5,6],
        Native='Common compiled full28 P and original NIXL D. No Token adapter, source credit or extra D cap.',
        primary_Natural_pilot_QPS=[.5,1.,2.,3.],CPC_replay_QPS=[1.,2.,3.],
        QPS_selection='All predeclared source-stratum points. No candidate or favorable-request selection.',
        profiling='Full CUPTI plus copies and scheduler state sampled at50ms. Clean/profile/clean with identical replay work.',
        profile_initializations='Each QPS has its own fresh E/P/D initialization. One collector begin/end per worker.',
        prior_unstarted_plan='ec_DOCUMENT_C39. Preserve its pre-GPU NOT_RUN closeout. Dataset and QPS are unchanged.',
        observer_gate='All5 metric differences and clean spreads<=5%; collector counters must pass. Otherwise DIAGNOSTIC.',
        distinction='Both workload and scheduler sampling differ from C27. Do not claim a one-factor cause for its observer overhead.',
        no_future_window_prediction=True,no_active_decode_insertion=True,service_proximity='NOT_MEASURED initially',
        SM_Tensor_HBM='NOT_MEASURED',task_quality='PENDING',main100_unchanged=True,
        wait_for_c37=args.wait_for_c37))
    write(ROOT/'state'/f'{TAG}_controller.json',dict(time=time.time(),pid=os.getpid(),
        status='WAITING_FOR_C37' if args.wait_for_c37 else 'STARTING'))
    if args.wait_for_c37:wait_for_confirmation()
    names=[];proxy=None;results=[]
    try:
        assert time.time()<experiment_deadline()
        names,proxy=launch(TAG+'_PILOT','cuda_ipc',fixed_reduction=True,deterministic_reductions=True,owned_names=names)
        set_common_ipc_policy()
        ordered=rows.copy();random.Random(2026112509).shuffle(ordered)
        for label,qps,output in [('warmup',2.,counts),('natural_q0.5',.5,None),
                ('natural_q1',1.,None),('natural_q2',2.,None),('natural_q3',3.,None)]:
            assert time.time()<experiment_deadline()
            run=TAG+'_'+label
            values=asyncio.run(execute(ordered,['http://127.0.0.1:18847'],run,qps,2026112510,output,
                metadata=dict(manifest=str(manifest),manifest_SHA256=digest(manifest),secondary_stratum=True,
                    evidence='DIAGNOSTIC_WARMUP' if label=='warmup' else 'MEASURED_ONLINE Native source-stratum pilot',
                    task_quality='PENDING',GPU_observer='OFF',not_Main100=True)))
            assert len(values)==37 and all(r['status']=='MEASURED_ONLINE' for r in values)
            ownership_empty(run,'cuda_ipc');metrics=summarize(run)
            results.append(dict(label=label,metrics=metrics))
            write(ROOT/'analysis'/f'{TAG}_partial.json',dict(time=time.time(),results=results))
            print('DOCUMENT_NATIVE_METRICS',label,{k:metrics[k] for k in
                ['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','gate_attainment']},flush=True)
        for ordinal,qps in enumerate([1.,2.,3.]):
            assert time.time()<experiment_deadline()
            if ordinal:
                stop_proxy(proxy);proxy=None
                retire([name for name in names if '_E4' not in name]);retire([name for name in names if '_E4' in name])
                names=[]
                names,proxy=launch(TAG+f'_PROFILE{ordinal}','cuda_ipc',fixed_reduction=True,
                    deterministic_reductions=True,owned_names=names)
                set_common_ipc_policy()
                run=TAG+f'_profile{ordinal}_warmup'
                values=asyncio.run(execute(ordered,['http://127.0.0.1:18847'],run,qps,2026112510,counts,
                    metadata=dict(manifest_SHA256=digest(manifest),evidence='DIAGNOSTIC_WARMUP',secondary_stratum=True)))
                assert len(values)==37 and all(r['status']=='MEASURED_ONLINE' for r in values)
                ownership_empty(run,'cuda_ipc')
            config=SimpleNamespace(manifest=manifest,replay_from=reference,server_names=','.join(names),
                order_seed=2026112509,qps=qps,seed=2026112510,name=TAG+f'_CPC_q{qps}_v0',
                url='http://127.0.0.1:18847',worker_urls=','.join(f'http://127.0.0.1:{port}' for port in [18840,18841,18842]),
                observer_kind='profile',profile_completion_roi=None,state_interval_ms=50.,
                observe_client_host=False,freeze_client_static_heap=False)
            asyncio.run(profile_run(config));ownership_empty(config.name,'cuda_ipc')
        write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),status='COMPLETE',results=results,
            source_stratum=True,model_quality='PENDING',evidence='Runtime pilots plus separately qualified/diagnostic profiling',
            Main100_replaced=False,serving_gain='NOT_TESTED',prefill_heavy='REQUIRES_TIMELINE_ANALYSIS'))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_pilots=len(results),restart_automatically=False))
        raise
    finally:
        if proxy is not None:stop_proxy(proxy)
        owned=[name for name in names if (ROOT/'state'/f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name]);retire([name for name in owned if '_E4' in name])
    write(ROOT/'state'/f'{TAG}_controller_complete.json',dict(time=time.time(),pid=os.getpid(),status='COMPLETE'))
    preserve()


if __name__=='__main__':main()
