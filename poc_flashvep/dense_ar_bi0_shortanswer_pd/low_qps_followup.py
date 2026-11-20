"""Use a short remaining budget for a predeclared low-load control."""
import json
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze
from .axis_timeline import analyze as axis_analyze


def main():
    declaration=ROOT/'configs/frozen_late_low_QPS_control_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),QPS=1.2,N=200,seed=2026112051,
            methods=['ControlledNative','Layer','Token'],cut='Same discovery-frozen best cut per axis.',
            selection='Run only after all primary jobs and the optional cut12 decision. Require14 minutes before the dispatch cutoff.',
            scope='Time-only fallback for a budget too small for the20-minute cut12 study. '
            'Do not cancel primary jobs or change the frozen all-method QPS extension rule. '
            'One start per actual-batch1 control. Do not fabricate missing continuous baselines at this shared configuration.'))
    while not any((ROOT/'state'/marker).exists() for marker in ('cut12_root_confirmation_complete.json','cut12_root_confirmation_not_run.json')):
        assert not (ROOT/'state/STOP_REQUESTED.json').exists()
        assert not (ROOT/'state/cut12_root_confirmation_hold.json').exists()
        time.sleep(5)
    budget=json.loads((ROOT/'state/budget.json').read_text());remaining=budget['last_large_dispatch_epoch']-time.time()
    existing=ROOT/'analysis/BI0_static_QPS_extension_v1.json'
    if existing.exists() and any(row['QPS']==1.2 for row in json.loads(existing.read_text())['results']):
        write(ROOT/'state/late_low_QPS_control_not_run.json',dict(time=time.time(),status='NOT_RUN',reason='QPS1.2 already measured by the primary extension.'));return
    if remaining<840:
        write(ROOT/'state/late_low_QPS_control_not_run.json',dict(time=time.time(),status='NOT_RUN',remaining_seconds=remaining,reason='Preserve the final-hour budget.'));return
    choices=json.loads((ROOT/'configs/frozen_static_axis_selection_v2.json').read_text())['winners']
    arms=[('ControlledNative','Native',0)]+[(axis,axis,r['cut']) for axis,r in choices.items() if 'cut' in r]
    rows=[]
    for label,method,cut in arms:
        name=f'LowQPSControl_{label}{cut}_N200_q1.2_s2026112051_BI0_v1'
        extra=dict(axis_kind=method,axis_cut=cut,axis_state_capture=False,hidden_source_slots=4,
            qualified_controlled_performance=True,runtime_qualification_tag='fast128_v3',
            token_teacher_gate_tag='fast128_v3',burst_gate_tag='fast128_v3',qualified_async_runtime=True,axis_mode='STATIC') if cut else {}
        value=serving(name,method,'longbench_shortanswer200_v2.json',200,1.2,2026112051,
            controlled_pipeline=True,source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',async_scheduling=True,
            KV_block_tokens=128,restore_official_UCX_init=True,quiescent_witness=True,
            phase='LATE_TIME_ONLY_LOW_LOAD_CONTROL',**extra)
        analyze(name);assert axis_analyze(name)['all_prompt_work_accounted']
        rows.append(dict(name=name,label=label,cut=cut,QPS=1.2,seed=2026112051,metrics=value))
        write(ROOT/'analysis/BI0_controlled_low_QPS_followup_v1.json',dict(results=rows,evidence='MEASURED_ONLINE',
            scope='One independent start per cell. Same discovery-frozen cuts. Actual batches1/1/1. No start-level CI or dynamic gain.'))
    write(ROOT/'state/late_low_QPS_control_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/late_low_QPS_control_hold.json',dict(time=time.time(),error=repr(error)));raise
