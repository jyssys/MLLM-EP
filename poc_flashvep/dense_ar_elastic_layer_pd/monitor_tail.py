"""Finish only predeclared cells after the administrative time limit."""
import argparse,fcntl
from .common import *
from .run import execute
from .analyze import compare,metrics,records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--driver-pid',type=int,required=True)
    args=parser.parse_args()
    selection=json.loads((ROOT/'configs/heldout_selection.json').read_text())
    deadline=selection['experiment_deadline_epoch']+3600
    write(ROOT/'state/monitor_resumption.json',dict(time=time.time(),driver_pid=args.driver_pid,
        original_deadline=selection['experiment_deadline_epoch'],administrative_extension_deadline=deadline,
        reason='The user authorized ample time and requested continued monitoring. Finish only immutable, predeclared unstarted cells. Do not tune runtime, selection, SLO, prompts, or predictors.'))
    while Path(f'/proc/{args.driver_pid}').exists():
        progress=json.loads((ROOT/'state/confirmation_progress.json').read_text())
        write(ROOT/'state/monitor_heartbeat.json',dict(time=time.time(),completed=len(progress['results']),
            not_run=len(progress['not_run']),failed=progress.get('failed'),status='ORIGINAL_DRIVER_ACTIVE'))
        if time.time()>deadline:
            raise RuntimeError('The original driver exceeded the extended administrative limit')
        time.sleep(20)
    path=ROOT/'state/confirmation_progress.json'
    state=json.loads(path.read_text())
    if state.get('failed'):
        write(ROOT/'state/monitor_terminal.json',dict(time=time.time(),status='FAILED_ORIGINAL_DRIVER',failed=state['failed']))
        return
    initial=ROOT/'analysis/confirmation_results.json'
    if initial.exists():write(ROOT/'analysis/initial_deadline_confirmation_results.json',json.loads(initial.read_text()))
    present={r['name'] for r in state['results']}
    plan=json.loads((ROOT/'state/confirmation_plan.json').read_text())['jobs']
    reference=next(r['name'] for r in plan if r['arm']=='Reference')
    state['administrative_extension']=dict(start=time.time(),deadline=deadline,
        originally_not_run=state['not_run'],scope='Unchanged predeclared grid. No result-dependent setting selection.')
    try:
        for item in plan:
            name=item['name']
            if name in present:continue
            if time.time()>=deadline:break
            if not execute(name,restore=False):
                state['failed']=name;write(path,state);return
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            gate=compare(reference,name,True)
            requests=records(ROOT/'raw'/name/'requests.jsonl')
            state['results'].append(dict(name=name,arm=item['arm'],cap=cfg['cap'],rate=cfg['arrival_rate'],
                seed=cfg['arrival_seed'],output=gate,metrics=metrics(name),
                completed_layer_distribution={str(n):sum(r.get('completed_D_prefill_layers',0)==n for r in requests)
                    for n in (0,2,4,8,12)},administrative_extension=True))
            present.add(name)
            state['not_run']=[r for r in state['not_run'] if r['name']!=name]
            write(path,state)
            print('RESUMED_FROZEN_CELL',name,gate['passed'],flush=True)
            if not gate['passed']:return
        state['not_run']=[dict(name=r['name'],status='NOT_RUN',reason='EXTENDED_ADMINISTRATIVE_DEADLINE')
            for r in plan if r['name'] not in present]
        write(path,state)
        write(initial,dict(state,evidence='MEASURED',
            scope='Original frozen choices and prompts. The administrative extension completed predeclared cells only. Throughput includes cohort drain.'))
        write(ROOT/'state/monitor_terminal.json',dict(time=time.time(),status='GPU_GRID_FINISHED',
            completed=len(state['results']),not_run=len(state['not_run'])))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
