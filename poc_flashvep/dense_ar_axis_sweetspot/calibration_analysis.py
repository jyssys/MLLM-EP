"""Freeze low-load action costs before held-out performance starts."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,write,digest,bind_legacy,preserve
from .model_validation import extract
from .online_audit import audit


def main():
    while not (ROOT/'state/initial_complete.json').exists():
        assert not (ROOT/'state/initial_stopped_v2.json').exists()
        time.sleep(10)
    bind_legacy();preserve()
    manifest=json.loads((ROOT/'state/initial_design_v2_before_performance.json').read_text())
    results=[]
    # Analysis holds the same lease. It cannot overlap a GPU measurement.
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for name in manifest['calibration']:
            value=audit(name)
            cfg,rows,costs=extract(ROOT,name)
            results.append(dict(name=name,method=cfg['evaluation_arm'],layers=cfg['static_layers'],
                output=cfg['cap'],batch=cfg['D_admission_batch'],cost_seconds=costs,
                measured_client_metrics=value['metric_summary'],
                common_Balanced=value['common_Balanced'],
                configuration_SHA256=digest(ROOT/'configs'/(name+'.json')),
                request_log_SHA256=digest(ROOT/'raw'/name/'requests.jsonl')))
        target=ROOT/'configs/low_load_action_costs_v1.json'
        data=dict(results=results,
            scope='One independent server start per action. These are calibrated host service costs, not exclusive kernels. Low offered load does not prove an empty queue. Keep observed waiting in the evidence. No held-out performance enters this lookup.',
            evidence='MEASURED_ONLINE')
        if target.exists():
            assert json.loads(target.read_text())['calibration']==data
        else:
            write(target,dict(frozen_epoch=time.time(),calibration=data))
    write(ROOT/'state/calibration_analysis_complete.json',dict(time=time.time(),
        path=str(target),SHA256=digest(target)))
    preserve()
    print('LOW_LOAD_COSTS_FROZEN',len(results),flush=True)


if __name__=='__main__':
    main()
