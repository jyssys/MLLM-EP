"""Run memory-qualified credit controls, then a bounded discovery matrix."""
import json
import time
import numpy as np
from .common import ROOT,PREVIOUS,SOURCE,write,digest,initialize
from .campaign import cell


def wait_for(*flags):
    while not all((ROOT/'state'/flag).exists() for flag in flags):
        assert not (ROOT/'state/STOP_REQUESTED.json').exists()
        for marker in ('last12_confirmation_hold.json','serial_online_causal_control_hold.json'):
            if (ROOT/'state'/marker).exists() and not (ROOT/'state/new_overlay_runtime_smoke_complete.json').exists():
                raise AssertionError(marker)
        time.sleep(5)


def pilot_data():
    target=ROOT/'data/discovery50.json'
    if target.exists():return
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text());rng=np.random.default_rng(2026112084)
    selected=[]
    for task in sorted({r['task'] for r in data}):
        indices=[i for i,r in enumerate(data) if r['task']==task]
        selected.extend(map(int,rng.choice(indices,10,replace=False)))
    selected.sort();rows=[data[i] for i in selected]
    write(target,rows)
    write(ROOT/'data/discovery50_manifest.json',dict(seed=2026112084,indices=selected,
        prompt_ids=[r['prompt_id'] for r in rows],SHA256=digest(target),N=50,
        selection='Ten requests per task, sampled without output or performance filtering. Preserve original order and original inputs.',
        scope='Pilot discovery only. Full200 independent starts confirm selected conditions. No N50 cell is labeled a full200 main result.'))


def repair_native():
    old='Last12Confirm_Native_q2.35_c4_s2026112071_N200_v1'
    hashes=json.loads((ROOT/'raw'/old/'source_hashes.json').read_text())
    path='poc_flashvep/dense_ar_bi0_axis_rootcause/serving.py'
    if hashes[path]!=digest(SOURCE/'serving.py'):
        name='Last12Confirm_Native_q2.35_c4_s2026112071_N200_buffered_v2'
        row=cell(name,seed=2026112071,phase='SOURCE_TRACE_CONTRACT_REPAIR')
        assert row['status']!='FAILED'
        write(ROOT/'analysis/last12_native_reference_repair.json',dict(original=old,replacement=row,
            reason='The first Native source snapshot predates buffered tracing. Replace it by source contract only, not by measured outcome. Original data remains visible.',
            evidence='MEASURED_ONLINE'))
    write(ROOT/'state/last12_native_reference_repair_complete.json',dict(time=time.time(),complete=True))


def credits():
    results=[];capacity=None
    for slots in (32,16,8):
        name=f'HiddenCapacity_Layer8_slots{slots}_BoundaryN3_v1'
        row=cell(name,'Layer',8,dataset='context_boundary3.json',count=3,credit=slots,
            serial=True,correctness_only=True,qualified_controlled_performance=False,
            axis_state_capture=True,hidden_source_slots=slots,phase='HIDDEN_CAPACITY_CORRECTNESS_PREFLIGHT')
        results.append(row)
        write(ROOT/'analysis/hidden_capacity_preflight.json',dict(results=results))
        if row['status']=='FAILED':continue
        folder=ROOT/'raw'/name
        p=json.loads((folder/'axis_events_prefill.json').read_text())['states']
        d=json.loads((folder/'axis_events_decode.json').read_text())['states']
        assert all(p[f'q{i}']['KV']==d[f'import_q{i}']['KV'] for i in range(3))
        assert all(json.loads(path.read_text())['passed'] for path in folder.glob('*lifetime*.json'))
        capacity=slots;break
    assert capacity is not None,'No fixed hidden pool passed memory and lifetime checks'
    write(ROOT/'configs/frozen_credit_pool_capacity.json',dict(time=time.time(),hidden_source_slots=capacity,
        scope='Selected using memory/state feasibility only. The same fixed Layer pool is used across source-credit arms.'))
    rows=[]
    for index,credit in enumerate((1,4,8,16,32)):
        order=('Native','Layer','Token') if index%2==0 else ('Token','Layer','Native')
        for axis in order:
            name=f'CreditSweep_{axis}8_q2.35_c{credit}_h{capacity}_s2026112083_N200_v1'
            if axis=='Layer' and credit>capacity:
                row=dict(name=name,axis=axis,credit=credit,status='NOT_RUN',reason='Source window exceeds the certified fixed hidden capacity.')
            else:
                row=cell(name,axis,0 if axis=='Native' else 8,seed=2026112083,credit=credit,
                    hidden_source_slots=capacity,phase='SOURCE_CREDIT_ONE_FACTOR_SWEEP')
            rows.append(row);write(ROOT/'analysis/credit_sweep_v1.json',dict(results=rows,
                evidence='MEASURED_ONLINE',scope='One independent start per cell. Fixed hidden pool across Layer credit arms. Not a confirmed sweet spot.'))
            if row['status']=='FAILED':
                write(ROOT/'state/credit_sweep_runtime_failure.json',dict(time=time.time(),name=name))
                raise AssertionError(name)
    write(ROOT/'state/credit_sweep_complete.json',dict(time=time.time(),complete=True))


def discovery():
    pilot_data();rows=[]
    path=ROOT/'configs/frozen_pilot50_grid_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),N=50,QPS=[1.2,2.35,.4,.7,1.,1.65,2.],cuts=[2,4,8,12],seed=2026112081,
            output='Natural EOS cap2048',source_credit=4,hidden_source_slots=4,
            scope='Budget allocation changed after the user prioritized the serial/online causal audit. Full200 cut confirmations stay separate. Do not relabel this pilot as the requested full200 matrix.'))
    budget=json.loads((ROOT/'state/budget.json').read_text());scheduled=[]
    for q in (1.2,2.35,.4,.7,1.,1.65,2.):
        scheduled.append(('Native',0,q))
        for cut in (2,4,8,12):
            order=('Layer','Token') if int(100*q+cut)%2==0 else ('Token','Layer')
            scheduled.extend((axis,cut,q) for axis in order)
    for axis,cut,q in scheduled:
        name=f'Pilot50_{axis}{cut}_q{q:g}_c4_s2026112081_v1'
        if budget['last_large_dispatch_epoch']-time.time()<90*60:
            rows.append(dict(name=name,axis=axis,cut=cut,QPS=q,N=50,status='NOT_RUN',reason='Preserve independent confirmation and closeout budget.'))
        else:
            row=cell(name,axis,cut,rate=q,seed=2026112081,dataset='discovery50.json',count=50,
                phase='PILOT50_FIXED_AXIS_QPS_DISCOVERY')
            rows.append(row)
            assert row['status']!='FAILED',name
        write(ROOT/'analysis/pilot50_all_cuts_qps_v1.json',dict(results=rows,
            evidence='MEASURED_ONLINE',scope='Small discovery cohort. No point has independent-start CI. Full200 confirmations are separate.'))
    write(ROOT/'state/pilot50_all_cuts_complete.json',dict(time=time.time(),complete=True))


def main():
    initialize();pilot_data()
    wait_for('last12_confirmation_complete.json','serial_online_causal_control_complete.json')
    repair_native();credits();discovery()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/study_hold.json',dict(time=time.time(),error=repr(error)));raise
