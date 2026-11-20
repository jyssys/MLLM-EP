"""Account for dispatcher time without calling it GPU compute time."""
import json
import time
from .common import ROOT,START,write


def main():
    rows=[]
    for path in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        receipt=json.loads(path.read_text());cfg=json.loads((path.parent/'configuration.json').read_text())
        gpus=cfg.get('gpus',[]);elapsed=receipt['elapsed_seconds']
        rows.append(dict(name=path.parent.name,normal_dispatcher_completion=receipt['complete'],
            returncode=receipt['returncode'],dispatcher_elapsed_seconds=elapsed,
            assigned_GPUs=gpus,assigned_GPU_hours=elapsed*len(gpus)/3600))
    lease=json.loads((ROOT/'state/GPU_lease_current.json').read_text())
    assert lease['status']=='RELEASED'
    elapsed=lease['time']-START
    write(ROOT/'analysis/experiment_budget_accounting.json',dict(time=time.time(),rows=rows,
        dispatched_commands=len(rows),normal_completions=sum(r['normal_dispatcher_completion'] for r in rows),
        failed_dispatches=sum(not r['normal_dispatcher_completion'] for r in rows),
        start_epoch=START,last_GPU_release_epoch=lease['time'],
        campaign_wall_hours_until_last_GPU_release=elapsed/3600,
        summed_dispatcher_elapsed_hours=sum(r['dispatcher_elapsed_seconds'] for r in rows)/3600,
        summed_assigned_GPU_hours=sum(r['assigned_GPU_hours'] for r in rows),
        evidence='STRUCTURAL_ACCOUNTING',scope='Dispatcher elapsed time includes engine startup,warmup,measurement and teardown. '
        'This sum excludes intervals between jobs and burner restoration. Assigned GPU-hours are elapsed times device count,not kernel-exclusive compute. '
        'Correctness jobs,diagnostics and failed commands are included. This is not the number of independent serving confirmation starts.'))
    print(json.dumps(dict(dispatched_commands=len(rows),normal_completions=sum(r['normal_dispatcher_completion'] for r in rows),
        wall_hours=elapsed/3600,assigned_GPU_hours=sum(r['assigned_GPU_hours'] for r in rows))))


if __name__=='__main__':main()
