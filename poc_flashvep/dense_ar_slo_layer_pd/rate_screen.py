"""One resumable long-window rate test with a declared execution objective."""
import argparse,fcntl
from .common import *
from .sustainable import duration_config,assess
from .dynamic_discovery import settings
from .run import execute
from .qualification import against_previous
from .analyze import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['Native','Colocated','NATIVE','STATIC','DIRECT'])
    p.add_argument('rate',type=float);p.add_argument('--cap',type=int,default=16)
    p.add_argument('--seed',type=int,default=2026111451);p.add_argument('--measure',type=float,default=600.)
    p.add_argument('--budget',type=int,default=8192);p.add_argument('--slots',type=int,default=1)
    p.add_argument('--group',type=int,default=16);p.add_argument('--destination',type=int,default=8)
    p.add_argument('--layers',type=int,default=8);p.add_argument('--fraction',type=float,default=1.)
    p.add_argument('--full-graph',action='store_true');p.add_argument('--compile3',action='store_true')
    p.add_argument('--opaque-norm',action='store_true')
    p.add_argument('--async-decode',action='store_true')
    p.add_argument('--tag',default='v1');a=p.parse_args();assert a.rate>0
    options=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,nixl_manual_coalesce=True,
        nixl_descriptor_sort=False,phase='FIXED_WINDOW_RATE_DISCOVERY_NOT_HELD_OUT')
    suffix=''
    if a.arm=='Colocated':
        options.update(ar_async_scheduling=True,chunk=a.budget,token_budget=a.budget,D_slots=a.slots)
        suffix=f'_B{a.budget}_S{a.slots}_ASYNC'
        if a.full_graph:
            assert json.loads((ROOT/'analysis/full_graph_colocated_gate_v2.json').read_text())['passed']
            if a.budget!=512:
                assert json.loads((ROOT/'analysis'/f'full_graph_B{a.budget}_state_gate.json').read_text())['passed']
            options.update(direct_phase_norm=False,gpu_phase_mask_norm=True,ar_cudagraph_mode='FULL',
                explicit_cudagraph_capture_sizes=sorted(set([1,2,4,8,16,a.slots,a.budget//2,a.budget])))
            suffix+='_FULL'
        if a.compile3:
            gate='compiled_opaque_norm_gate.json' if a.opaque_norm else 'compiled_full_graph_colocated_gate.json'
            assert a.full_graph and json.loads((ROOT/'analysis'/gate).read_text())['passed']
            assert a.budget==512,'Qualify each compiled token budget before long timing'
            options.update(ar_compile_mode=3,startup_timeout=900);suffix+='_COMPILE3'
            if a.opaque_norm:
                options.update(opaque_unfused_norm=True,ar_execution_family='COMPILE3_OPAQUE_ORIGINAL_UNFUSED_NORM')
                suffix+='_OPAQUE_ORIGINAL_NORM'
        elif a.opaque_norm:raise AssertionError('Opaque original normalization is a qualified compiled-path option')
    elif a.arm in ('STATIC','DIRECT','NATIVE'):
        options.update(settings(a.group),D_admission_batch=a.destination,
            bounded_destination_batches=True,static_layers=a.layers,migration_fraction=a.fraction,
            static_request_block_size=a.group)
        suffix=f'_G{a.group}_D{a.destination}_L{a.layers}_F{a.fraction}'
        assert (ROOT/'configs/coalesced_cost_model_v1.json').exists()
        assert json.loads((ROOT/'analysis/source16_destination8_full_state_gate.json').read_text())['passed']
    else:options.update(D_admission_batch=a.destination)
    if a.async_decode:
        assert a.arm!='Colocated' and json.loads((ROOT/'analysis/async_D_online_work_gate.json').read_text())['passed']
        options['async_decode']=True;suffix+='_ASYNC_D'
    name=f'rate_{a.arm}{suffix}_O{a.cap}_q{a.rate}_s{a.seed}_W{a.measure}_{a.tag}'
    cfg=duration_config(name,a.arm,a.rate,cap=a.cap,seed=a.seed,measure=a.measure,follow=90.,**options)
    try:
        assert execute(save(cfg),restore=False)
        assert against_previous(name)['passed']
        value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
        append(ROOT/'state/rate_screens.jsonl',dict(time=time.time(),name=name,complete=True,
            qualified={k:v['qualified'] for k,v in value['SLO_regimes'].items()}))
        print('RATE_SCREEN',name,value['raw_completed_output_tokens_s'],
            {k:(v['qualified'],v['arrival_cohort_joint_attainment'],v['SLO_qualified_output_tokens_s'])
                for k,v in value['SLO_regimes'].items()},flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
