"""Validate the low-headroom result without opening a large dynamic grid."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,resources,write
from .flow_campaign import scenario_config,static_config
from .axis_configuration import base
from .discovery import job
from .layer_qualification import gate
from .replay import load_model,groups,oracle,evaluate
from .strong_static import frontier


def main():
    while not (ROOT/'analysis/flow_oracle_stop.json').exists():
        assert time.time()<json.loads((ROOT/'state/progress.json').read_text())['deadline_epoch']-7200
        time.sleep(15)
    screen=json.loads((ROOT/'analysis/flow_macro8_oracle_screen_v3.json').read_text())
    selected=max((r for r in screen if r['oracle']['Joint_attainment']>=.95),key=lambda r:r['ratio'])
    values=[]
    try:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            assert not resources().processes()
            resources().restore(list(range(8)))
            model=load_model(8,'v3');work=groups(selected['arrivals'],selected['caps'],8)
            static=frontier(model,work);dynamic=oracle(model,work,width=256);ideal=evaluate(dynamic)
            write(ROOT/'analysis/strong_per_cap_fraction_oracle.json',dict(static=static,oracle=ideal,
                oracle_plan=[d['layers'] for d in dynamic.decisions],
                ratio=ideal['SLO_output_tokens_s']/static['best']['metric']['SLO_output_tokens_s'],
                evidence='OFFLINE_ORACLE',scope='Same empirical state-updating replay. The bounded dynamic search is not an upper bound. Per-cap fixed fractions remove a pipeline-balance confound. No online gain claim.'))
        write(ROOT/'state/flow_low_headroom_diagnostic_selection.json',selected)
        cfg=base('flow8_GPU_full_state_qualification_N32_v1',method='Dynamic',count=32,cap=64,
            matched=True,P_group_size=8,D_admission_batch=8,flow_balance_policy=True,
            flow_horizon_groups=2,flow_cost_model='empirical_host_cost_model_discovery_v3.json',
            correctness_only=True,clean_timing=False,verify_layer_execution=True,
            full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill','decode'],
            hidden_verify_hash=True,verify_native_prefix_KV=True,negotiation_timeout_s=120.)
        assert job(cfg);gate(cfg['name'])
        homogeneous=[]
        for horizon in (2,3):
            cfg=base(f'flow8_H{horizon}_N64_O64_q4.8_v1',method='Dynamic',count=64,cap=64,
                matched=True,P_group_size=8,D_admission_batch=8,flow_balance_policy=True,
                flow_horizon_groups=horizon,flow_cost_model='empirical_host_cost_model_discovery_v3.json',
                hidden_source_slots=32,hidden_destination_slots=32)
            value=job(cfg)
            if value:
                homogeneous.append(value);write(ROOT/'analysis/flow_homogeneous64_discovery.json',homogeneous)
        cfg=static_config('flow_low_Static_fraction_N128_v1',selected,static['best']['plan'])
        cfg['static_policy_descriptor']=static['best']['policy']
        configs=[cfg]
        configs.append(static_config('flow_low_Static_fraction_P4_adaptiveD8_N128_v1',selected,static['best']['plan'],
            P_group_size=4,preplanned_source_groups=False,D_wait_for_source_macro=False,D_combine_ready_macros=True,
            static_policy_descriptor=static['best']['policy']))
        configs.append(scenario_config('flow_low_Online_H2_N128_v1',selected,method='Dynamic',
            flow_balance_policy=True,flow_horizon_groups=2,flow_cost_model='empirical_host_cost_model_discovery_v3.json'))
        configs.append(scenario_config('flow_low_ControllerZero_N128_v1',selected,method='Dynamic',
            flow_balance_policy=True,flow_horizon_groups=2,flow_controller_zero=True,
            flow_cost_model='empirical_host_cost_model_discovery_v3.json'))
        configs.append(scenario_config('flow_low_Native_default_N128_v1',selected,method='Native',
            P_group_size=4,preplanned_source_groups=False,D_wait_for_source_macro=False,D_combine_ready_macros=True,
            D_admission_batch=16))
        for cfg in configs:
            value=job(cfg)
            if value:
                values.append(value);write(ROOT/'analysis/flow_low_headroom_actual_pilot.json',values)
        write(ROOT/'state/flow_low_headroom_pilot_complete.json',dict(names=[r['name'] for r in values],
            scope='Small model-validation pilot after the original under5% screen. No large mixed-workload sweep is authorized by this controller.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
