"""Freeze the final structural replay before fresh confirmation measurements."""
import json
import statistics
import time
from .common import ROOT,DEADLINE,write,digest,preserve


def main():
    preserve();protocol=ROOT/'state/control_boundary_replay_protocol.json';assert protocol.exists()
    while not (ROOT/'state/service_model_v4_frozen.json').exists():
        if (ROOT/'state/control_gap_model_deferred.json').exists():return
        assert not (ROOT/'state/control_gap_model_failure.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(.25)
    tests=json.loads((ROOT/'state/batch4_knob_confirmation_design.json').read_text())['names']
    assert not any((ROOT/'raw'/n/'measurement_begin.json').exists() for n in tests)
    v3=ROOT/'configs/frozen_action_service_costs_v3.json';v4=ROOT/'configs/frozen_action_service_costs_v4.json'
    costs={r['method']:dict(r['cost_seconds']) for r in json.loads(v3.read_text())['model']['models']
        if r['layers']==8 and r['batch']==4 and r['output']==16}
    grouped={a:[] for a in costs};idle=[];provenance=[]
    for r in json.loads(v4.read_text())['gap_estimator_training']:
        feedback=ROOT/'analysis'/(r['name']+'_reservation_feedback.json')
        row=json.loads(feedback.read_text());parts=row['mean_backlogged_nonwindow_gap_parts_ms']
        grouped[row['method']].append((parts['before_reservation_ms']+parts['post_reservation_ms'])/1000)
        idle.append(row['reservation_RTT_by_host_phase']['HOST_IDLE_NO_ACTIVE_COHORT']['mean_ms']/1000)
        provenance.append(dict(name=r['name'],SHA256=digest(feedback)))
    for a in costs:costs[a]['P_gap']=statistics.median(grouped[a])
    target=ROOT/'configs/frozen_control_feedback_replay_inputs_v1.json';assert not target.exists()
    now=time.time();mono=time.monotonic()
    write(target,dict(frozen_epoch=now,frozen_monotonic=mono,costs=costs,idle_RTT_seconds=statistics.median(idle),
        model_v3_SHA256=digest(v3),model_v4_SHA256=digest(v4),training=provenance,
        protocol_SHA256=digest(protocol),evidence='STRUCTURAL_ACCOUNTING',
        scope='Only six new calibration traces. Separate non-ACK P gap from the mandatory control round trip. D prompt preparation gives one modeled boundary per request. Decode gives one modeled boundary per output step. No fresh confirmation result is read. This is a host-step model, not exclusive GPU timing.'))
    assert not any((ROOT/'raw'/n/'measurement_begin.json').exists() for n in tests)
    write(ROOT/'state/control_feedback_replay_inputs_frozen.json',dict(time=now,monotonic=mono,
        SHA256=digest(target),before_confirmation_measurements=True))
    preserve();print('CONTROL_BOUNDARY_REPLAY_INPUTS_FROZEN',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/control_replay_freeze_failure.json',dict(time=time.time(),status='FAILED',error=repr(error)))
        raise
