"""One clean/event/clean Native control with no candidate insertion."""
import asyncio
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .client import resolve_manifest
from .common import ROOT,digest,preserve,write,experiment_deadline
from .frontend_supply_control import retire
from .ipc_ec_control import launch,stop_proxy,correctness,ownership_empty
from .llm_probe_control import rpc
from .native_inline_confirmation import measure,REFERENCE,URL
from .profile_control import run as profile_run
from .source_snapshot import snapshot

TAG='ec_GPU_ENCLOSURES_C19'


def main():
    import argparse
    global TAG
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default=TAG)
    parser.add_argument('--qps',type=float,choices=[.5,2,3],default=2)
    parser.add_argument('--observer-kind',choices=['gpu_enclosures','profile'],default='gpu_enclosures')
    parser.add_argument('--wait-for-tag')
    parser.add_argument('--profile-completion-roi',nargs=2,type=int)
    args=parser.parse_args()
    assert args.tag.isidentifier() and args.tag.startswith(('ec_GPU_ENCLOSURES_C19','ec_GPU_ENCLOSURES_C20','ec_LOW_CUPTI_C25','ec_ROI_CUPTI_C27'))
    if args.observer_kind=='profile':
        assert (args.tag.startswith('ec_LOW_CUPTI_C25') and args.qps==.5 and not args.profile_completion_roi) or (
            args.tag.startswith('ec_ROI_CUPTI_C27') and args.qps==3 and args.profile_completion_roi==[30,70])
    if args.profile_completion_roi:assert args.observer_kind=='profile'
    if args.wait_for_tag:assert args.wait_for_tag.isidentifier()
    TAG=args.tag;qps=args.qps
    preserve()
    path=ROOT/'configs'/f'{TAG}.json';assert not path.exists()
    source=snapshot(TAG)
    write(path,dict(time=time.time(),status='FROZEN_BEFORE_OBSERVER_RESULTS',
        source_archive_SHA256=source['archive_SHA256'],QPS=qps,warmup_QPS=2,requests=100,
        one_factor=('Bound CUPTI to the frozen completed-request interval30:70 at QPS3. Full-cohort metrics remain unchanged.'
            if args.profile_completion_roi else
            'Record full CUPTI kernel/copy activity at low load after the low-load event observer qualified.'
            if args.observer_kind=='profile' else
            'Replace full CUPTI activity recording with two CUDA events per original model-runner execute.'),
        observer_kind=args.observer_kind,wait_for_tag=args.wait_for_tag,
        profile_completion_roi=args.profile_completion_roi,
        unchanged=['Original stock compiled Native model','Main100','Natural-derived fixed replay counts',
                   'Arrival/order seeds','EC transport and ACK','B8192/S1024','KV42GiB','Frozen SLO'],
        evidence='DIAGNOSTIC until clean/profile/clean qualification',observer_gate_max=.05,
        limitation=('Every primary use still requires the frozen 5% observer and clean-repeatability gates.'
            if args.observer_kind=='profile' else
            'GPU stream enclosures are not kernel-exclusive busy. Extra-stream transfers may be uncovered.'),
        no_candidate_offloading=True,no_safe_headroom_claim=True))
    if args.wait_for_tag:
        while True:
            files=[p for p in (ROOT/'state').glob(f'server_{args.wait_for_tag}_*.json')
                if not p.name.endswith('.source_snapshot.json')]
            leases=[json.loads(p.read_text()) for p in files]
            assert len(leases)==3,'Expected the exact three preceding owned server leases'
            if all(r['finished'] and not Path('/proc',str(r['guardian'])).exists() for r in leases):break
            for r in leases:
                proc=Path('/proc',str(r['guardian']))
                if not r['finished']:assert proc.exists() and proc.stat().st_uid==os.getuid()
            assert time.time()<experiment_deadline();time.sleep(1)
    if TAG.startswith('ec_GPU_ENCLOSURES_C20'):
        prior=[json.loads((ROOT/'state'/f'server_ec_GPU_ENCLOSURES_C19_{role}{gpu}.json').read_text())
               for role,gpu in [('E',4),('P',5),('D',6)]]
        assert all(r['finished'] and not Path('/proc',str(r['guardian'])).exists() for r in prior)
    while True:
        leases=[json.loads((ROOT/'state'/f'server_token_CORRECTNESS_C15R2_{role}{gpu}.json').read_text())
                for role,gpu in [('E',4),('P',5),('D',6)]]
        if all(r['finished'] and not Path('/proc',str(r['guardian'])).exists() for r in leases):break
        for r in leases:
            proc=Path('/proc',str(r['guardian']))
            if proc.exists():
                assert proc.stat().st_uid==os.getuid()
                assert b'mllm_epd_prefill_heavy_token_split.servers' in (proc/'cmdline').read_bytes()
        assert time.time()<experiment_deadline()
        time.sleep(1)
    expected=[f'{TAG}_{role}{gpu}' for role,gpu in [('E',4),('P',5),('D',6)]]
    proxy=None
    try:
        names,proxy=launch(TAG,'cuda_ipc');assert names==expected
        for port in [18841,18842]:rpc(port,'ec_ipc/ack_policy',dict(mode='file'))
        rpc(18840,'ec_ipc/inline_policy',dict(enabled=True))
        asyncio.run(correctness(TAG,'cuda_ipc'))
        asyncio.run(measure(TAG,'warmup_replay100',2))
        run_args=SimpleNamespace(manifest=resolve_manifest(None),replay_from=REFERENCE,
            server_names=','.join(names),order_seed=2026112404,qps=qps,seed=2026112402,
            name=TAG+f'_CPC_q{qps:g}_v0',url=URL,
            worker_urls='http://127.0.0.1:18840,http://127.0.0.1:18841,http://127.0.0.1:18842',
            observer_kind=args.observer_kind,state_interval_ms=0 if args.observer_kind=='profile' else 50,
            profile_completion_roi=args.profile_completion_roi,
            observe_client_host=False,freeze_client_static_heap=False)
        asyncio.run(profile_run(run_args));ownership_empty(TAG,'cuda_ipc')
        write(ROOT/'state'/f'{TAG}_complete.json',dict(time=time.time(),status='COMPLETE',
            qualification_SHA256=digest(ROOT/'analysis'/f'{run_args.name}_qualification.json')))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error)))
        raise
    finally:
        if proxy is not None:stop_proxy(proxy)
        owned=[n for n in expected if (ROOT/'state'/f'server_{n}.json').exists()]
        if owned:
            retire([n for n in owned if '_E4' not in n]);retire([n for n in owned if '_E4' in n])
    preserve()


if __name__=='__main__':main()
