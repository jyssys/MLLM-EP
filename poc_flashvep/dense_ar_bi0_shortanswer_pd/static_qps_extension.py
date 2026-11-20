"""Reuse discovery-frozen cuts at new rates without per-rate retuning."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze
from .axis_timeline import analyze as axis_analyze


def main():
    design=ROOT/'configs/frozen_static_QPS_extension_v2.json'
    if not design.exists():
        write(design,dict(time=time.time(),QPS=[1.2,1.65],N=200,arrival_seed=2026112051,
            arms=['ContinuousNative','Agg','ControlledNative','Layer','Token'],
            selection='The same q2.35 discovery-frozen cut at every rate. No per-QPS retuning.',
            actual_control_batches=[1,1,1],source_credit=4,KV_block_tokens=128,B=8192,S=1024,
            scope='One independent start per additional cell. Start3 confidence comes only from q2.35 confirmation. '
            'Budget rule: run both rates with at least45 minutes remaining. Run1.65 only with25–45 minutes. '
            'Otherwise leave the extension NOT_RUN. Freeze this rule before discovery performance.'))
    while not (ROOT/'state/paired_shared_KV128_confirmation_v2_complete.json').exists():
        for marker in ('STOP_REQUESTED.json','confirmation_hold.json','controlled_discovery_hold.json'):
            assert not (ROOT/'state'/marker).exists(),marker
        time.sleep(5)
    budget=json.loads((ROOT/'state/budget.json').read_text())
    remaining=budget['last_large_dispatch_epoch']-time.time()
    if remaining<25*60:
        write(ROOT/'state/static_QPS_extension_not_run.json',dict(time=time.time(),status='NOT_RUN',reason='Preserve the final validation/publication budget.'));return
    rates=(1.2,1.65) if remaining>=45*60 else (1.65,)
    write(ROOT/'state/static_QPS_budget_selection_v2.json',dict(time=time.time(),remaining_seconds=remaining,
        selected_QPS=list(rates),omitted_QPS=[q for q in (1.2,1.65) if q not in rates],
        omitted_status='NOT_RUN_BUDGET',scope='Only the predeclared remaining-time rule selects rates. No measured winner is used.'))
    choices=json.loads((ROOT/'configs/frozen_static_axis_selection_v2.json').read_text())['winners']
    results=[]
    for rate in rates:
        arms=[('ContinuousNative','Native',0),('Agg','Agg',0),('ControlledNative','Native',0)]
        arms.extend((axis,axis,r['cut']) for axis,r in choices.items() if 'cut' in r)
        for label,method,cut in arms:
            name=f'FixedQPS_{label}{cut}_N200_q{rate:g}_s2026112051_BI0_v1'
            extra={}
            if label not in ('ContinuousNative','Agg'):
                extra.update(controlled_pipeline=True,source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',async_scheduling=True)
            if cut:extra.update(axis_kind=method,axis_cut=cut,axis_state_capture=False,hidden_source_slots=4,
                qualified_controlled_performance=True,runtime_qualification_tag='fast128_v3',
                token_teacher_gate_tag='fast128_v3',burst_gate_tag='fast128_v3',qualified_async_runtime=True,axis_mode='STATIC')
            value=serving(name,method,'longbench_shortanswer200_v2.json',200,rate,2026112051,
                KV_block_tokens=128,restore_official_UCX_init=True,phase='FROZEN_CUT_NEW_QPS',**extra)
            analyze(name)
            if label not in ('ContinuousNative','Agg'):assert axis_analyze(name)['all_prompt_work_accounted']
            results.append(dict(name=name,label=label,cut=cut,QPS=rate,seed=2026112051,metrics=value))
            write(ROOT/'analysis/BI0_static_QPS_extension_v1.json',dict(results=results,
                evidence='MEASURED_ONLINE',scope='Frozen cut across QPS. Actual-batch1 PD mechanism controls and ordinary continuous external baselines remain separate. Natural EOS finite cohorts.'))
    write(ROOT/'state/static_QPS_extension_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/static_QPS_extension_hold.json',dict(time=time.time(),error=repr(error)));raise
