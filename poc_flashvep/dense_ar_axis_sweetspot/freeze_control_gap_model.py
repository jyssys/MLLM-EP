"""Freeze a declared host-gap estimator before any fresh confirmation arrival."""
import json
import statistics
import time
from .common import ROOT,DEADLINE,write,digest,preserve
from .reservation_feedback import analyze


def main():
    preserve()
    protocol=ROOT/'state/control_ACK_gap_estimator_protocol_before_new_calibration.json'
    assert protocol.exists()
    while not (ROOT/'state/service_model_v3_frozen.json').exists():
        if (ROOT/'state/batch4_cost_confirmation_deferred.json').exists():
            write(ROOT/'state/control_gap_model_deferred.json',dict(time=time.time(),status='NOT_RUN'))
            return
        assert not (ROOT/'state/batch4_cost_confirmation_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(.25)
    design=json.loads((ROOT/'state/new_batch_cost_calibration_design.json').read_text())
    tests=json.loads((ROOT/'state/batch4_knob_confirmation_design.json').read_text())['names']
    assert not any((ROOT/'raw'/name/'measurement_begin.json').exists() for name in tests), 'Do not freeze after evaluation begins'
    grouped={a:[] for a in ('Layer','Token')};provenance=[]
    for name in design['names']:
        row=analyze(name)
        assert row['eligible_backlogged_nonwindow_samples']>=10
        grouped[row['method']].append(row['mean_backlogged_nonwindow_P_gap_ms']/1000)
        provenance.append(dict(name=name,feedback_SHA256=digest(ROOT/'analysis'/(name+'_reservation_feedback.json')),
            mean_gap_ms=row['mean_backlogged_nonwindow_P_gap_ms'],gap_parts_ms=row['mean_backlogged_nonwindow_gap_parts_ms']))
    base=ROOT/'configs/frozen_action_service_costs_v3.json'
    model=json.loads(base.read_text())['model']
    for row in model['models']:
        if row['layers']==8 and row['batch']==4 and row['output']==16:
            row['old_v3_median_below50ms_P_gap']=row['cost_seconds']['P_gap']
            row['cost_seconds']['P_gap']=statistics.median(grouped[row['method']])
            row['status']='BATCH4_AND_UNCENSORED_BACKLOGGED_CONTROL_GAP_CALIBRATION'
    model['scope']='Keep v3 calibration starts and scalar score equations. Replace only Last8/O16/Decode4 P_gap with the median of three per-start means over already-arrived, source-window-unblocked transitions. Do not discard50ms-or-longer reservation waits. Actual old/new model errors are evaluated on fresh1970/1971/1972 seeds. No GPU or SLO change.'
    target=ROOT/'configs/frozen_action_service_costs_v4.json'
    assert not target.exists()
    epoch=time.time();monotonic=time.monotonic()
    write(target,dict(frozen_epoch=epoch,frozen_monotonic=monotonic,based_on_v3_SHA256=digest(base),model=model,
        gap_estimator_training=provenance,protocol_SHA256=digest(protocol)))
    assert not any((ROOT/'raw'/name/'measurement_begin.json').exists() for name in tests), 'Evaluation raced the model freeze'
    write(ROOT/'state/service_model_v4_frozen.json',dict(time=epoch,monotonic=monotonic,path=str(target),SHA256=digest(target),
        confirmed_before_any_evaluation_measurement_begin=True,
        scope='CPU calibration helper read only the six declared calibration traces. Worker process startup, if already underway, is not a measured arrival. No evaluation result was read.'))
    preserve();print('CONTROL_ACK_GAP_MODEL_V4_FROZEN',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/control_gap_model_failure.json',dict(time=time.time(),status='FAILED',error=repr(error),
            scope='CPU cost-estimator failure. Preserve the other models and GPU evidence. Do not count this model as held-out qualified.'))
        raise
