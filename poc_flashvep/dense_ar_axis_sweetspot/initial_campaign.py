"""Qualify Last2 and acquire low-load effective service demand first."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,bind_legacy,preserve,resources,save,write
from .configuration import base,state
from .run import execute


def main():
    bind_legacy();preserve()
    from ..dense_ar_matched_axis_pd.qualification import gate
    from ..dense_ar_matched_axis_pd.analysis import metrics
    from ..dense_ar_matched_axis_pd.batch_matched import audit
    qualification=[state(a,m,n) for n in (8,32) for a,m in (('Native',0),('Layer',2),('Token',2))]
    calibration=[]
    for a,m in [('Native',0)]+[(a,m) for m in (2,4,8,12) for a in ('Layer','Token')]:
        name=f'calibration_{a}_Last{m}_N36_O16_q0p5_s2026111900_B1_W32_v2'
        calibration.append(base(name,a,m,36,16,.5,2026111900,
            phase='LOW_LOAD_ACTION_COST_CALIBRATION',matrix_stage='calibration'))
    manifest=ROOT/'state/initial_design_v2_before_performance.json'
    design=dict(qualification=[c['name'] for c in qualification],calibration=[c['name'] for c in calibration],
        calibration_seed=2026111900,B=8192,S=1024,actual_batches=[1,1,1],
        method_scope='Fixed actions. No online dynamic scheduler. N8/32 state tests do not match aggregate rows. N36 timing matches exact rows.',
        next_decision='Use measured effective P/D demand and guard decomposition to select discriminating low/high-load and WIP counterfactuals.')
    if manifest.exists():
        assert json.loads(manifest.read_text())==design
    else:
        write(manifest,design)
    for cfg in qualification+calibration:
        save(cfg)
    try:
        for index,cfg in enumerate(qualification+calibration):
            write(ROOT/'state/initial_progress_v2.json',dict(time=time.time(),status='RUNNING',
                completed=index,total=len(qualification+calibration),name=cfg['name']))
            assert execute(cfg['name']), cfg['name']
            if cfg['correctness_only']:
                reference=next(c['name'] for c in qualification if c['evaluation_arm']=='Native' and c['count']==cfg['count'])
                gate(cfg['name'],reference)
            else:
                metrics(cfg['name']);audit(cfg['name'])
            print('INITIAL_PROGRESS',index+1,len(qualification+calibration),cfg['name'],flush=True)
        write(ROOT/'state/initial_complete.json',dict(time=time.time(),qualification=6,calibration=9))
    except BaseException as error:
        write(ROOT/'state/initial_stopped_v2.json',dict(time=time.time(),error=repr(error)))
        raise
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            owner=resources()
            if not owner.processes():
                owner.restore(list(range(8)))
        preserve()


if __name__=='__main__':
    main()
