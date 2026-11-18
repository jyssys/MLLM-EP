"""Monitor the revised scope without marking deferred matrices complete."""
import fcntl
import json
import time
from pathlib import Path
from .common import ROOT, QUIET_LOCK, preserve, resources, write
from .monitor import snapshot as original_snapshot


def snapshot():
    value = original_snapshot()
    design = json.loads((ROOT/'state/batch_matched_design_20261118.json').read_text())
    done = []
    failed = []
    for name in design['names']:
        path = ROOT/'raw'/name/'dispatcher_result.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        (done if result.get('complete') else failed).append(name)
    controllers = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if (b'poc_flashvep.dense_ar_matched_axis_pd.batch_matched' in args and b'--prepare' not in args
                or b'poc_flashvep.dense_ar_matched_axis_pd.fixed_cohort_online' in args):
            controllers.append(int(path.name))
    stopped = ROOT/'state/batch1_stopped.json'
    critical = value['critical_failures']+failed
    if stopped.exists():
        critical.append(json.loads(stopped.read_text())['error'])
    operator_stopped = ROOT/'state/serving_projection_API_control_stopped.json'
    if operator_stopped.exists():
        critical.append('PROJECTION_API_CONTROL: '+json.loads(operator_stopped.read_text())['error'])
    complete = (ROOT/'state/batch1_complete.json').exists()
    online={}
    design2=ROOT/'state/cohort2_online_design.json'
    if design2.exists():
        value2=json.loads(design2.read_text())
        done2=0
        for name in value2['names']:
            p=ROOT/'raw'/name/'dispatcher_result.json'
            if p.exists() and json.loads(p.read_text()).get('complete'):
                done2+=1
        online=dict(completed=done2,total=value2['total'],complete=(ROOT/'state/cohort2_online_complete.json').exists())
        stopped2=ROOT/'state/cohort2_online_stopped.json'
        if stopped2.exists():
            online['stopped']=json.loads(stopped2.read_text())
    all_complete=complete and (not online or online['complete'])
    prototype = {}
    for key, name in [('design','fixed_cohort2_qualification_design.json'),
                      ('progress','fixed_cohort2_qualification_progress.json'),
                      ('complete','fixed_cohort2_qualification_complete.json'),
                      ('stopped','fixed_cohort2_qualification_stopped.json')]:
        path = ROOT/'state'/name
        if path.exists():
            prototype[key] = json.loads(path.read_text())
    version2 = {}
    for key, suffix in [('progress','progress'),('complete','complete'),('stopped','stopped')]:
        path = ROOT/'state'/('fixed_cohort2_v2_qualification_'+suffix+'.json')
        if path.exists():
            version2[key] = json.loads(path.read_text())
    version3 = {}
    for key, suffix in [('progress','progress'),('complete','complete'),('stopped','stopped')]:
        path = ROOT/'state'/('fixed_cohort2_v3_qualification_'+suffix+'.json')
        if path.exists():
            version3[key] = json.loads(path.read_text())
    lease = value.get('GPU_lease') or {}
    log = ROOT/'logs'/(lease.get('name', '')+'.log')
    age = time.time()-log.stat().st_mtime if log.exists() else None
    job_age = time.time()-lease.get('start_epoch', time.time())
    stalled = (lease.get('status') == 'GPU_JOB_RUNNING' and
               job_age > 240 and age is not None and age > 180)
    return dict(epoch=time.time(), protocol='ACTUAL_BATCH_ONE', completed=len(done),
        total=design['total_starts'], qualification_complete=(ROOT/'state/batch1_full_state_qualified.json').exists(),
        complete=complete, controllers=controllers, critical=sorted(set(critical)),
        progress_stalled=stalled, job_log_age_s=age, job_age_s=job_age,
        lease=lease, original_counts=value['counts'], fixed_cohort2=prototype,
        fixed_cohort2_v2=version2,
        fixed_cohort2_v3=version3,
        cohort2_online=online,all_scopes_complete=all_complete,
        original_remaining_scope='DEFERRED_BY_USER_REQUEST_NOT_COMPLETED')


def main():
    while True:
        value = snapshot()
        write(ROOT/'state/batch1_monitor_progress.json', value)
        print(json.dumps(value), flush=True)
        assert not value['critical'], value
        assert value['all_scopes_complete'] or value['controllers'], value
        if value['all_scopes_complete']:
            preserve()
            with QUIET_LOCK.open('a+') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX)
                resources().restore(list(range(8)))
            preserve()
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
