"""Validate the new macro model, then test bounded online lookahead."""
import collections
import fcntl
import json
import random
import time
from .common import ROOT, QUIET_LOCK, resources, write
from .axis_configuration import base
from .discovery import job
from .replay import load_model, groups, oracle, static_frontier, evaluate
from ..duetserve_agg_decode.configuration import offsets


def wait_for_calibration():
    path=ROOT/'analysis/macro8_calibration_serving.json'
    while not path.exists() or len(json.loads(path.read_text()))<5:
        assert time.time()<json.loads((ROOT/'state/progress.json').read_text())['deadline_epoch']-7200
        time.sleep(15)


def select_model():
    from .cost_trace import fit
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not resources().processes()
        resources().restore(list(range(8)))
        fit('v3')
        for kind in ('P_source_ms','D_prepare_ms','negotiation_ms'):
            model=load_model(8,'v3')
            assert all(f'8:{m}' in model.tables[kind] for m in (0,2,4,8,12))
        cap_groups=[16]*8+[64]*5+[200]*3
        random.Random(2026111730).shuffle(cap_groups)
        caps=[cap for cap in cap_groups for _ in range(8)]
        values=[]
        for rate in (3.2,3.8,4.8,6.4):
            model=load_model(8,'v3')
            arrival=offsets(128,rate,2026111731)
            workload=groups(arrival,caps,8)
            best,frontier=static_frontier(model,workload)
            ideal=oracle(model,workload,width=128)
            observed=evaluate(ideal)
            ranked=sorted(frontier,key=lambda v:(v[0]['SLO_output_tokens_s'],-v[0]['TTFT_p95_ms']),reverse=True)
            unique=[]
            seen=set()
            for metric,policy,state in ranked:
                plan=tuple(d['layers'] for d in state.decisions)
                if plan in seen:
                    continue
                seen.add(plan)
                unique.append(dict(metric=metric,policy=policy,plan=list(plan)))
                if len(unique)==3:
                    break
            ratio=observed['SLO_output_tokens_s']/best[0]['SLO_output_tokens_s'] if best[0]['SLO_output_tokens_s'] else None
            record=dict(rate=rate,arrivals=arrival,caps=caps,static=best[0],
                static_policy=best[1],static_frontier=unique,oracle=observed,
                oracle_cuts=[d['layers'] for d in ideal.decisions],ratio=ratio,
                borrowed=sorted(model.borrowed),evidence='OFFLINE_ORACLE',
                scope='State-updating bounded beam with actual macro8 host calibration. Not an optimality proof, online gain or physical idle measurement. All static cap mappings and fixed-fraction candidates are screened.')
            values.append(record)
            print('V3_ORACLE',rate,ratio,observed['Joint_attainment'],flush=True)
        write(ROOT/'analysis/flow_macro8_oracle_screen_v3.json',values)
        return values


def scenario_config(name, selected, method='Static', **changes):
    cfg=base(name,method=method,count=128,cap=200,rate=selected['rate'],seed=2026111731,
        request_caps=selected['caps'],arrival_offsets=selected['arrivals'],matched=True,
        P_group_size=8,D_admission_batch=8,hidden_source_slots=32,hidden_destination_slots=32,
        D_combine_ready_macros=False,D_wait_for_source_macro=True,
        phase='DISCOVERY_VISIBLE_FIFO_MIXED_CAPS',workload_label='Mixed_output_grouped8')
    cfg.update(changes)
    return cfg


def static_config(name, selected, plan, **changes):
    return scenario_config(name,selected,static_layers_by_request=[m for m in plan for _ in range(8)],
        static_fraction_axis='REQUEST_ID_BLOCKS',static_request_block_size=8,**changes)


def main():
    wait_for_calibration()
    try:
        screen=select_model()
        eligible=[r for r in screen if r['ratio'] and r['ratio']>=1.05]
        if not eligible:
            write(ROOT/'analysis/flow_oracle_stop.json',dict(status='NO_PROMISING_MODELED_HEADROOM',
                evidence='OFFLINE_ORACLE',scope='Only the calibrated macro8 mixed-cap grid. This is not a universal Dynamic impossibility claim.'))
            return
        # Favor the largest ratio among candidates whose modeled attainment
        # clears95%. Do not relabel an overloaded failure as qualified capacity.
        qualified=[r for r in eligible if r['oracle']['Joint_attainment']>=.95]
        selected=max(qualified or eligible,key=lambda r:r['ratio'])
        write(ROOT/'state/flow_discovery_selection.json',selected)
        cfg=scenario_config('flow_GPU_qualification_N32_v1',selected,method='Dynamic',
            count=32,request_caps=selected['caps'][:32],arrival_offsets=[0.]*32,
            correctness_only=True,clean_timing=False,verify_layer_execution=True,
            full_prompt_state_witness=True,full_prompt_state_witness_roles=['prefill','decode'],
            verify_native_prefix_KV=True,hidden_verify_hash=True,
            flow_balance_policy=True,flow_cost_model='empirical_host_cost_model_discovery_v3.json',
            flow_horizon_groups=2,flow_completion_margin_ms=110.,negotiation_timeout_s=120.)
        assert job(cfg)
        from .layer_qualification import gate
        gate(cfg['name'])
        values=[]
        for horizon in (2,3):
            cfg=base(f'flow8_H{horizon}_N64_O64_q4.8_v1',method='Dynamic',count=64,
                cap=64,rate=4.8,matched=True,P_group_size=8,D_admission_batch=8,
                flow_balance_policy=True,flow_horizon_groups=horizon,
                flow_cost_model='empirical_host_cost_model_discovery_v3.json',
                hidden_source_slots=32,hidden_destination_slots=32)
            value=job(cfg)
            if value:
                values.append(value)
                write(ROOT/'analysis/flow_homogeneous64_discovery.json',values)
        for index,item in enumerate(selected['static_frontier']):
            cfg=static_config(f'flow_Static_frontier{index}_N128_v1',selected,item['plan'])
            value=job(cfg)
            if value:
                values.append(value)
                write(ROOT/'analysis/flow_discovery_GPU_results.json',values)
        # Stronger geometry controls are allowed to beat the fixed-macro study.
        best_mapping=selected['static_frontier'][0]['plan']
        for group in (4,):
            cfg=static_config(f'flow_Static_P{group}_adaptiveD8_N128_v1',selected,best_mapping,
                P_group_size=group,preplanned_source_groups=False,
                D_wait_for_source_macro=False,D_combine_ready_macros=True)
            value=job(cfg)
            if value:
                values.append(value)
                write(ROOT/'analysis/flow_discovery_GPU_results.json',values)
        for horizon in (2,3):
            cfg=scenario_config(f'flow_Online_H{horizon}_N128_v1',selected,method='Dynamic',
                flow_balance_policy=True,flow_cost_model='empirical_host_cost_model_discovery_v3.json',
                flow_horizon_groups=horizon,flow_completion_margin_ms=110.)
            value=job(cfg)
            if value:
                values.append(value)
                write(ROOT/'analysis/flow_discovery_GPU_results.json',values)
        cfg=scenario_config('flow_ControllerZero_N128_v1',selected,method='Dynamic',
            flow_balance_policy=True,flow_controller_zero=True,
            flow_cost_model='empirical_host_cost_model_discovery_v3.json',
            flow_horizon_groups=2,flow_completion_margin_ms=110.)
        value=job(cfg)
        if value:
            values.append(value)
        # A future-selected replay plan is a diagnostic, not an online policy.
        cfg=static_config('flow_Offline_selected_plan_GPU_diagnostic_N128_v1',selected,selected['oracle_cuts'],
            phase='OFFLINE_SELECTED_SCHEDULE_CORRECTNESS_AND_COST_DIAGNOSTIC',
            offline_selected_schedule=True)
        value=job(cfg)
        if value:
            values.append(value)
        write(ROOT/'analysis/flow_discovery_GPU_results.json',values)
        write(ROOT/'state/flow_discovery_complete.json',dict(time=time.time(),names=[v['name'] for v in values]))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
