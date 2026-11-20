"""Confirm the discovery crossover only if the reserved budget permits it."""
import collections
import json
import time
import numpy as np
from .common import ROOT,write
from .campaign import serving
from .axis_timeline import analyze
from .confirmation_analysis import paired_ratio,core_sources


def main():
    design=ROOT/'configs/frozen_cut12_root_confirmation_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),cut=12,N=200,QPS=2.35,seeds=[2026112051,2026112052,2026112053],
            reason='The largest predeclared cut reversed the discovery Layer/Token ordering. Test its reproducibility after all primary jobs.',
            minimum_remaining_dispatch_seconds=1200,
            scope='Exploratory condition selection followed by independent arrival starts. '
            'Do not replace the discovery-frozen best cuts8. Do not cancel a primary job to create time. '
            'Missing time means NOT_RUN, not a negative performance result.'))
    required=('paired_shared_KV128_confirmation_v2_complete.json','equal_output_control_complete.json',
        'continuation_costs_complete.json','fast_control_P_profile_complete.json')
    while not all((ROOT/'state'/marker).exists() for marker in required) or not any(
            (ROOT/'state'/marker).exists() for marker in ('static_QPS_extension_complete.json','static_QPS_extension_not_run.json')):
        assert not (ROOT/'state/STOP_REQUESTED.json').exists()
        for marker in ('confirmation_hold.json','equal_output_control_hold.json','continuation_costs_hold.json',
                       'fast_control_P_profile_hold.json','static_QPS_extension_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    budget=json.loads((ROOT/'state/budget.json').read_text())
    remaining=budget['last_large_dispatch_epoch']-time.time()
    if remaining<1200:
        write(ROOT/'state/cut12_root_confirmation_not_run.json',dict(time=time.time(),status='NOT_RUN',remaining_seconds=remaining,
            reason='Preserve the predeclared final-hour closeout budget.'));return
    results=[]
    for seed in (2026112051,2026112052,2026112053):
        for axis in ('Layer','Token'):
            name=f'RootCut12_{axis}_N200_q2.35_s{seed}_BI0_v1'
            value=serving(name,axis,'longbench_shortanswer200_v2.json',200,2.35,seed,
                controlled_pipeline=True,source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',async_scheduling=True,
                KV_block_tokens=128,restore_official_UCX_init=True,quiescent_witness=True,
                axis_kind=axis,axis_cut=12,axis_state_capture=False,hidden_source_slots=4,
                qualified_controlled_performance=True,runtime_qualification_tag='fast128_v3',token_teacher_gate_tag='fast128_v3',
                burst_gate_tag='fast128_v3',qualified_async_runtime=True,axis_mode='STATIC',phase='EXPLORATORY_CROSSOVER_INDEPENDENT_CONFIRMATION')
            assert analyze(name)['all_prompt_work_accounted']
            results.append(dict(name=name,axis=axis,cut=12,seed=seed,metrics=value))
            write(ROOT/'analysis/BI0_cut12_root_confirmation_v1.json',dict(results=results,evidence='MEASURED_ONLINE',
                scope='Exploratory discovery-cell selection. Independent held-out arrival starts. Frozen best cuts remain8.'))
    groups=collections.defaultdict(dict)
    for row in results:groups[row['axis']][row['seed']]=row
    fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second')
    ratios={field:paired_ratio([groups['Layer'][s]['metrics'][field] for s in groups['Layer']],
        [groups['Token'][s]['metrics'][field] for s in groups['Layer']]) for field in fields}
    matches=all(core_sources(groups['Layer'][s]['name'])==core_sources(groups['Token'][s]['name']) for s in groups['Layer'])
    native_gates={}
    for axis in ('Layer','Token'):
        measures={}
        for field in ('ITL_p99_ms','TPOT_p95_ms'):
            controls=[json.loads((ROOT/'analysis'/f'Confirmation_ControlledNative0_N200_q2.35_s{s}_BI0_v2_metrics.json').read_text())[field] for s in groups[axis]]
            measures[field]=paired_ratio([groups[axis][s]['metrics'][field] for s in groups[axis]],controls)
        native_gates[axis]=dict(ratios=measures,passed=all(r['CI95'][1]<=1.05 for r in measures.values()))
    write(ROOT/'analysis/BI0_cut12_root_confirmation_statistics_v1.json',dict(ratios=ratios,
        core_runtime_sources_equal=matches,native_relative_decode_5pct=native_gates,
        all_quality_CI_passed=all(r['metrics']['quality_gate']['CI_passed'] for r in results),
        means={axis:{field:float(np.mean([r['metrics'][field] for r in values.values()])) for field in fields} for axis,values in groups.items()},
        evidence='MEASURED_ONLINE',scope='Three paired starts per axis at cut12. Exploratory condition selection remains explicit. '
        'This does not retune the main static winner or establish an online dynamic policy.'))
    write(ROOT/'state/cut12_root_confirmation_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/cut12_root_confirmation_hold.json',dict(time=time.time(),error=repr(error)));raise
